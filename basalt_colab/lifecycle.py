"""Colab G4 VM lifecycle manager, tunnel repair, watchdog, and cached Basalt telemetry."""
from __future__ import annotations

import http.client
import os
import re
import subprocess
import sys
import threading
import time
from typing import Any, Callable

from basalt_colab.colab import (
    CHILD_FLAGS,
    Logger,
    alive,
    boot_cmd,
    build_bundle,
    child_env,
    cleanup_local,
    colab_run,
    colab_usage,
    http_call,
    kill_rec,
    kill_tree,
    load_session,
    me,
    missing_vm_files,
    pick_port,
    proc_start,
    read_api_key,
    save_session,
    session_alive,
    spawn_detached,
    ssh_cmd,
    vm_timeline,
    vmx,
)
from basalt_colab.config import Config, ConfigError, diff_hot_cold
from basalt_colab.manifest import Manifest
from basalt_colab.paths import IS_WIN, Paths
from basalt_colab.usage import UsageLedger

BOOT_TIMEOUT = 1500
FAIL_COOLDOWN = 60
KEEPALIVE_S = 5
UPSTREAM_TIMEOUT = 3600

STEPS = (
    ("plan", r"storage plan mode=", "storage layout planned"),
    ("patch", r"abliteration edit fetched", "abliteration edit downloaded (43 KB)"),
    ("model", r"backbone fetched \([^)]+\)", "model backbone downloaded"),
    ("build", r"basalt built", "Basalt built / ready"),
    ("abliterate", r"abliteration applied", "abliteration baked into weights (62/62)"),
    ("loaded", r"weights loaded", "model loaded onto GPU + RAM"),
    ("pinned", r"pin: ", "backbone storage punched & reclaimed"),
    ("engram", r"engram done", "Engram table loaded"),
    ("warm", r"first token", "warmup answer done"),
)
LATER = (
    ("loaded", "loading the model onto GPU + RAM"),
    ("pinned", "reclaiming backbone storage"),
    ("engram", "loading the Engram table"),
    ("warm", "warming up"),
)


class StartError(Exception):
    pass


def mmss(seconds: float) -> str:
    return "%d:%02d" % divmod(max(0, int(float(seconds))), 60)


def poll_interval() -> float:
    return float(os.environ.get("BASALT_COLAB_POLL_S") or os.environ.get("LLM_POLL_S") or "15")


def tunnel_health(paths: Paths, port: int, timeout: float = 10) -> bool:
    try:
        conn, r = http_call(port, "GET", "/health", headers={"Authorization": "Bearer " + read_api_key(paths)}, timeout=timeout)
        ok = r.status == 200 and b"ok" in r.read()
        conn.close()
        return ok
    except (OSError, http.client.HTTPException):
        return False


def spawn_watchdog(paths: Paths, session: str, t0: float, max_hours: float, idle_min: float) -> dict[str, Any]:
    deadline = t0 + max_hours * 3600 + min(120.0, max_hours * 180)
    args = [
        sys.executable, "-m", "basalt_colab", "_watchdog",
        session, repr(t0), repr(deadline), repr(idle_min), repr(max_hours),
    ]
    extra = {"BASALT_COLAB_HOME": str(paths.config_dir)} if os.environ.get("BASALT_COLAB_HOME") else None
    p = spawn_detached(args, extra_env=extra)
    return {"pid": p.pid, "start": proc_start(p.pid)}


def run_watchdog(paths: Paths, session: str, t0: float, deadline: float, idle_min: float, max_hours: float) -> None:
    """Detached guard: stops the VM at hard deadline, or after idle_min once the proxy process dies."""
    logger = Logger(paths, echo=False)
    poll, orphan = max(1.0, min(30.0, idle_min * 6)), None
    while paths.session_file.exists():
        s = load_session(paths)
        if s is not None and (s.get("name") != session or abs(s.get("t0", 0) - t0) > 1e-3):
            return
        now, why = time.time(), None
        if now >= deadline:
            why = f"hard stop after {max_hours:.2f} h"
        elif s is not None:
            orphan = None if alive(s.get("serve")) else (orphan or now)
            if orphan and now - orphan >= idle_min * 60:
                why = f"proxy gone for {idle_min:.1f} min"
        if why:
            logger.log(f"watchdog {session}: {why}, stopping the VM")
            kill_rec((s or {}).get("tunnel"))
            for _ in range(3):
                colab_run(["stop", "-s", session], 120)
                if not session_alive(session):
                    break
                time.sleep(5)
            logger.log(f"watchdog {session}: session {'STILL ACTIVE' if session_alive(session) else 'stopped'}")
            return
        time.sleep(max(0.2, min(poll, deadline - now)))


def drain_stderr(p: subprocess.Popen[Any], logger: Logger, on_create: Callable[[], None] | None = None) -> None:
    assert p.stderr is not None
    for raw in p.stderr:
        line = raw.decode("utf-8", "replace").strip()
        if line and not re.search(r"open failed: connect failed", line):
            logger.log("tunnel: " + line[-300:], echo=False)
            if on_create and re.search(r"\[colab\] Creating (runtime|session)", line):
                on_create()


def stop_cloud(
    paths: Paths,
    cfg: Config,
    manifest: Manifest,
    ledger: UsageLedger,
    logger: Logger,
    s: dict[str, Any] | None,
    reason: str,
    served: int = 0,
) -> tuple[bool, float | None, float | None]:
    """Kills tunnel, wipes VM secrets, stops Colab session, records usage.jsonl; returns (still_alive, cu, ran_s)."""
    session = (s or {}).get("name") or cfg.colab.session
    s = s or {"name": session}
    kill_rec(s.get("tunnel"))
    was_alive = session_alive(session)
    if was_alive:
        vmx(session, "rm -rf /content/secrets; echo CLEANED", 30)
        for _ in range(3):
            colab_run(["stop", "-s", session], 120)
            if not session_alive(session):
                break
            time.sleep(5)
    still = session_alive(session)
    bal, _ = colab_usage()
    alloc_t0 = s.get("allocated_ts") or s.get("t0")
    billed = bool(alloc_t0) and (was_alive or (s.get("stages") or {}).get("allocated") is not None)
    ran = max(0.0, time.time() - float(alloc_t0)) if billed else None
    cu = (ran / 3600.0 * manifest.g4_cu_per_hour) if ran is not None else None
    cost_str = f", ran {mmss(ran)}, about {cu:.2f} CU" if (ran is not None and cu is not None) else ""
    logger.log(
        f"down ({reason}): session {'STILL ACTIVE, run: basalt-colab down' if still else 'stopped'}, "
        f"balance {bal} CU{cost_str}"
    )
    if billed and ran is not None and cu is not None:
        try:
            ledger.record(
                session=session,
                started_ts=float(alloc_t0),
                reason=reason,
                ran_s=ran,
                cu_est=cu,
                balance_start=s.get("balance_start"),
                balance_end=bal,
                model=s.get("model") or cfg.basalt.effective_model_alias,
                basalt_ref=s.get("basalt_ref") or cfg.basalt.ref or manifest.basalt_tag,
                served=served,
            )
        except OSError:
            pass
    if still:
        s["stopping"] = True
        save_session(paths, s)
    else:
        cleanup_local(paths, s)
    return still, cu, ran


class Manager:
    """Colab G4 lifecycle manager inside serve: wake on demand, hold waiters, repair tunnel, auto-stop."""

    def __init__(self, paths: Paths, cfg: Config, manifest: Manifest, ledger: UsageLedger, logger: Logger) -> None:
        self.paths = paths
        self.cfg = cfg
        self.manifest = manifest
        self.ledger = ledger
        self.logger = logger
        self.cond = threading.Condition()
        self.state = "off"
        self.steps: list[tuple[str, float, str | None]] = []
        self.attempt = 0
        self.errors: dict[int, str] = {}
        self.begun = 0.0
        self.fail_until = 0.0
        self.fail_msg = ""
        self.stop_reason: str | None = None
        self.active = 0
        self.last_activity = time.time()
        self.served = 0
        self.waiters = 0
        self.waiter_lost_at: float | None = None
        self.explicit_up = False
        self.alloc_note = ""
        self.tunnel: subprocess.Popen[Any] | None = None
        self.port: int | None = None
        self.tunnel_ok = False
        self.repairing = False
        self.sess: dict[str, Any] | None = None
        self.closing = False
        self.last_retry = 0.0
        self.tlock = threading.Lock()
        self.no_tunnel = False
        self.last_stop: dict[str, Any] | None = None
        self._last_stopped_pair: tuple[float | None, float | None] = (None, None)
        self._stats_lock = threading.Lock()
        self._stats_at = 0.0
        self._stats_cache: dict[str, Any] | None = None
        self._override_wait_gpu: bool | None = None
        self._override_ref: str | None = None
        self._override_quant: str | None = None
        self._override_pack_url: str | None = None
        self._override_uncensor_scale: float | None = None

    def ready(self) -> bool:
        return self.state == "ready" and self.tunnel_ok

    def enter(self) -> bool:
        with self.cond:
            if self.ready():
                self.active += 1
                return True
        return False

    def leave(self, served: int = 1) -> None:
        with self.cond:
            self.active -= 1
            self.served += served
            self.last_activity = time.time()

    def reload_config(self, new_cfg: Config, prev_cfg: Config | None = None) -> tuple[list[str], list[str]]:
        with self.cond:
            hot, cold = diff_hot_cold(prev_cfg or self.cfg, new_cfg)
            self.cfg.colab.idle_min = new_cfg.colab.idle_min
            self.cfg.colab.max_hours = new_cfg.colab.max_hours
            self.cfg.colab.wait_gpu = new_cfg.colab.wait_gpu
            self.cfg.colab.orphan_boot_s = new_cfg.colab.orphan_boot_s
            self.cfg.proxy.strict_model = new_cfg.proxy.strict_model
            self.cfg.basalt.effort = new_cfg.basalt.effort
            # If VM is off, cold basalt settings take effect immediately for the next boot too!
            if self.state == "off":
                self.cfg.basalt = new_cfg.basalt
                cold = []
        return hot, cold

    def acquire(
        self,
        on_tick: Callable[[int], None] | None = None,
        every: float = KEEPALIVE_S,
        explicit: bool = False,
        is_gone: Callable[[], bool] | None = None,
        wait_gpu: bool | None = None,
        basalt_ref: str | None = None,
        quant: str | None = None,
        pack_url: str | None = None,
        uncensor_scale: float | None = None,
    ) -> str | None:
        t0 = time.time()
        tick, mine = t0 + every, None
        with self.cond:
            self.waiters += 1
            self.waiter_lost_at = None
            if explicit:
                self.explicit_up = True
        try:
            while True:
                if is_gone and is_gone():
                    return "client disconnected"
                with self.cond:
                    if self.ready():
                        self.active += 1
                        return None
                    if mine is not None and mine in self.errors:
                        return self.errors[mine]
                    if mine is None and self.state in ("off", "stopping") and time.time() < self.fail_until:
                        return self.fail_msg
                    if self.state == "off":
                        if self.closing:
                            return "proxy is shutting down"
                        self._override_wait_gpu = wait_gpu
                        self._override_ref = basalt_ref
                        self._override_quant = quant
                        self._override_pack_url = pack_url
                        self._override_uncensor_scale = uncensor_scale
                        self._begin(True, explicit=explicit)
                    if self.state == "starting":
                        mine = self.attempt
                        if explicit:
                            self.explicit_up = True
                    wait_s = max(0.05, tick - time.time())
                    self.cond.wait(min(0.5, wait_s) if is_gone else wait_s)
                if time.time() >= tick:
                    tick += every
                    if on_tick:
                        on_tick(int(time.time() - t0))
        finally:
            with self.cond:
                self.waiters = max(0, self.waiters - 1)
                if self.waiters == 0 and self.state == "starting" and not self.explicit_up:
                    self.waiter_lost_at = time.time()

    def wait_off(self, on_tick: Callable[[int], None] | None = None, limit: float = 900, every: float = KEEPALIVE_S) -> bool:
        t0 = time.time()
        tick = t0 + every
        while time.time() - t0 < limit:
            with self.cond:
                if self.state == "off":
                    return self.paths.session_file.exists()
                self.cond.wait(max(0.05, tick - time.time()))
            if time.time() >= tick:
                tick += every
                if on_tick:
                    on_tick(int(time.time() - t0))
        return True

    def adopt(self) -> None:
        with self.cond:
            if self.state == "off":
                self._begin(False, explicit=True)

    def _begin(self, new: bool, explicit: bool = False) -> None:
        self.attempt += 1
        self.state, self.steps, self.stop_reason, self.no_tunnel, self.begun = "starting", [], None, False, time.time()
        self.explicit_up, self.waiter_lost_at, self.alloc_note = explicit, None, ""
        threading.Thread(target=self._run_start, args=(self.attempt, new), daemon=True).start()

    def _check_stop(self) -> None:
        if self.stop_reason:
            raise StartError(f"start cancelled ({self.stop_reason})")

    def _run_start(self, attempt: int, new: bool) -> None:
        try:
            ok, err = self._start(new), None
        except (StartError, ConfigError) as e:
            ok, err = False, str(e)
        except Exception as e:
            ok, err = False, f"internal error: {e!r}"
        with self.cond:
            self.alloc_note = ""
            if ok and not self.stop_reason:
                self.state, self.tunnel_ok, self.last_activity = "ready", True, time.time()
                self.cond.notify_all()
                return
            if ok:
                err = f"start cancelled ({self.stop_reason})"
            if err is None:
                self.state = "off"
                self.cond.notify_all()
                return
            self.errors[attempt], self.fail_msg, self.fail_until = err, err, time.time() + FAIL_COOLDOWN
            self.state = "stopping"
            self.cond.notify_all()
        self.logger.log("start failed: " + err)
        self._stop("start failed")

    def request_stop(self, reason: str, only_idle: bool = False) -> None:
        with self.cond:
            if only_idle and not (
                self.state == "ready"
                and self.active == 0
                and time.time() - self.last_activity > self.cfg.colab.idle_min * 60
            ):
                return
            if self.state == "starting":
                self.stop_reason = reason
                return
            if self.state == "stopping" or (self.state == "off" and not self.paths.session_file.exists()):
                return
            self.state = "stopping"
            self.cond.notify_all()
        self.logger.log(f"stopping the VM ({reason})")
        threading.Thread(target=self._stop, args=(reason,), daemon=True).start()

    def pop_last_stopped(self) -> tuple[float | None, float | None]:
        pair = self._last_stopped_pair
        self._last_stopped_pair = (None, None)
        return pair

    def _stop(self, reason: str) -> None:
        with self.tlock:
            self.no_tunnel = True
            self._kill_tunnel()
        try:
            _, cu, ran = stop_cloud(
                self.paths, self.cfg, self.manifest, self.ledger, self.logger,
                load_session(self.paths), reason, served=self.served,
            )
            self._last_stopped_pair = (cu, ran)
            self.last_stop = {"reason": reason, "cu": cu, "ran": ran, "at": time.time()}
        except Exception as e:
            self.logger.log(f"stop failed: {e!r}")
        with self.cond:
            self.state, self.tunnel_ok, self.sess, self.alloc_note = "off", False, None, ""
            self._stats_cache = None
            self.cond.notify_all()

    def _stage(self, s: dict[str, Any], key: str, t0: float) -> None:
        s["stages"][key] = round(time.time() - t0, 1)
        save_session(self.paths, s)

    def activity(self) -> str:
        done = {key for key, _, _ in self.steps}
        if "allocated" not in done:
            return self.alloc_note or "allocating a G4 GPU"
        if "uploaded" not in done:
            return "uploading the boot files"
        todo = [w for k, w in (("model", "downloading the model"), ("build", "preparing Basalt")) if k not in done]
        rest = [w for k, w in LATER if k not in done]
        todo = [] if len(rest) < len(LATER) else todo
        return " and ".join(todo) if todo else rest[0] if rest else "opening the endpoint"

    def _start(self, new: bool) -> bool:
        session = self.cfg.colab.session
        s = load_session(self.paths)
        if session_alive(session):
            self.logger.log(f"session {session} is running, adopting it")
            s = s or {
                "name": session,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "t0": time.time(),
                "allocated_ts": time.time(),
                "stages": {},
            }
            s.pop("stopping", None)
            s["serve"] = me()
            if not alive(s.get("watchdog")):
                s["watchdog"] = spawn_watchdog(
                    self.paths, session, float(s["t0"]), self.cfg.colab.max_hours, self.cfg.colab.idle_min
                )
            save_session(self.paths, s)
            self.sess = s
            self.steps += [("allocated", time.time(), None), ("uploaded", time.time(), None)]
            self._await_ready(s, adopted=True)
        elif not new:
            if s:
                cleanup_local(self.paths, s)
                self.logger.log(f"cleaned up state of the stopped session {session}")
            return False
        else:
            if s:
                cleanup_local(self.paths, s)
            s = self._launch()
            self._await_ready(s, adopted=False)
        return True

    def _sleep_cancelable(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self._check_stop()
            time.sleep(min(0.5, max(0.05, end - time.time())))

    def _launch(self) -> dict[str, Any]:
        cfg = self.cfg
        session = cfg.colab.session
        if self._override_quant is not None:
            from basalt_colab.config import quant_value
            q = quant_value(self._override_quant)
            if not q:
                raise StartError(f"invalid --quant value: {self._override_quant!r}")
            cfg.basalt.quant = q
        if self._override_pack_url is not None:
            cfg.basalt.pack_url = self._override_pack_url.strip()
        if self._override_uncensor_scale is not None:
            cfg.basalt.uncensor_scale = float(self._override_uncensor_scale)
            cfg.basalt.uncensor = cfg.basalt.uncensor_scale > 0.0
        if self._override_ref is not None:
            cfg.basalt.ref = self._override_ref.strip()
        try:
            from basalt_colab.config import validate_config
            validate_config(cfg)
            env_vars = cfg.boot_vars(manifest=self.manifest)
        except ConfigError as e:
            raise StartError(str(e))
        try:
            missing = missing_vm_files(self.paths.vm_dir)
        except OSError as e:
            raise StartError(f"cannot read vm/ ({self.paths.vm_dir}): {e}")
        if missing:
            raise StartError("vm/ is incomplete, missing: " + ", ".join(missing))
        for p in (self.paths.key_file, self.paths.ssh_key):
            if not p.exists():
                raise StartError(f"missing {p}; run: basalt-colab setup")
        bal, _ = colab_usage()
        if bal is None:
            raise StartError("the Colab CLI cannot read your balance (login expired?); run: basalt-colab setup")
        if bal < cfg.colab.min_balance:
            raise StartError(f"Colab balance {bal:.2f} compute units is below min_balance {cfg.colab.min_balance:.0f}")
        self._check_stop()
        t0 = time.time()
        model_id = cfg.basalt.effective_model_alias
        s: dict[str, Any] = {
            "name": session,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "balance_start": bal,
            "t0": t0,
            "stages": {},
            "serve": me(),
            "model": model_id,
            "quant": cfg.basalt.quant,
            "basalt_ref": env_vars.get("BASALT_REF") or self.manifest.basalt_tag,
            "cold_signature": cfg.cold_signature(),
        }
        save_session(self.paths, s)
        s["watchdog"] = spawn_watchdog(self.paths, session, t0, cfg.colab.max_hours, cfg.colab.idle_min)
        save_session(self.paths, s)
        self.sess = s
        self.logger.log(
            f"up: {model_id} ({cfg.describe_engine()}, {cfg.describe_uncensor()}), balance {bal:.2f} CU, "
            f"hard stop {cfg.colab.max_hours:.1f} h, idle stop {cfg.colab.idle_min:.1f} min"
        )
        wait_gpu = cfg.colab.wait_gpu if self._override_wait_gpu is None else self._override_wait_gpu
        deadline = time.time() + (1800 if wait_gpu else 300)
        max_attempts = 40 if wait_gpu else 3
        delay_scale = min(1.0, poll_interval() / 15.0)
        out = ""
        for attempt in range(max_attempts):
            rc, out = colab_run(["new", "-s", session, "--gpu", "G4"], 300)
            if rc == 0 and "READY" in out:
                self.alloc_note = ""
                break
            self.logger.log(f"allocation attempt {attempt + 1} failed: {out.strip()[-300:]}")
            self._check_stop()
            if attempt + 1 >= max_attempts or time.time() >= deadline:
                brief = " ".join(re.sub(r"[╭╮╰╯─│]+", " ", out).split())[-200:]
                raise StartError("no G4 GPU from Colab (none free, or plan has no compute units): " + brief)
            wait_s = min(60.0, 10.0 * (2 ** min(attempt, 3))) * max(0.02, delay_scale)
            self.alloc_note = f"waiting for G4 GPU (retry {attempt + 2} in {max(1, int(wait_s))}s)"
            self._sleep_cancelable(wait_s)
        s["allocated_ts"] = time.time()
        self._stage(s, "allocated", t0)
        self.logger.log(f"G4 allocated after {s['stages']['allocated']:.0f} s")
        self.steps.append(("allocated", time.time(), "G4 GPU allocated"))
        bundle = self.paths.state_dir / "bundle.tgz"
        try:
            build_bundle(self.paths, bundle)
            rc, out = colab_run(["upload", "-s", session, str(bundle), "/content/bundle.tgz"], 120)
        finally:
            try:
                bundle.unlink()
            except OSError:
                pass
        if rc != 0 or "Upload failed" in out:
            raise StartError("bundle upload failed: " + out.strip()[-200:])
        self._check_stop()
        rc, out = vmx(session, boot_cmd(env_vars), 60)
        if "BOOT_STARTED" not in out:
            raise StartError("boot start failed: " + out.strip()[-300:])
        self._stage(s, "boot_started", t0)
        self.steps.append(("uploaded", time.time(), None))
        self.logger.log(f"boot started on GPU VM after {s['stages']['boot_started']:.0f} s; ready in ~2-3 minutes")
        return s

    def _await_ready(self, s: dict[str, Any], adopted: bool) -> None:
        session = self.cfg.colab.session
        poll_s = poll_interval()
        health_s = min(1.0, poll_s)
        t0, seen, fails, spawns, next_tl, box = time.time(), 0, 0, 0, time.time(), None
        base = float(s.get("t0", t0)) + float(s.get("stages", {}).get("boot_started", 0))
        kill_rec(s.get("tunnel"))
        while time.time() - t0 < BOOT_TIMEOUT:
            if box is None and time.time() >= next_tl:
                next_tl, box = time.time() + poll_s, []
                threading.Thread(target=lambda b=box: b.append(vm_timeline(session)), daemon=True).start()
            if box:
                (rc, out), box = box[0], None
                if rc != 0:
                    self.logger.log(f"vm timeline read failed (rc {rc})", echo=False)
                lines = [l for l in out.splitlines() if re.match(r"^\d+s ", l)]
                if adopted and rc == 0 and not lines:
                    raise StartError(f"running session {session} has no boot timeline")
                adopted, fails = False, (0 if rc == 0 else fails + 1)
                if fails >= 5 and not session_alive(session):
                    raise StartError("the VM disappeared during boot")
                seen = self._take(lines, seen, base)
                bad = [l for l in lines if re.search(r"FAILED|died|GUARD|\[PATCH FAIL\]", l)]
                if bad:
                    raise StartError("boot failed on VM: " + bad[0])
            if self.tunnel is None or self.tunnel.poll() is not None:
                if spawns and (spawns >= 5 or not session_alive(session)):
                    raise StartError("tunnel kept failing or the VM disappeared during boot")
                self._kill_tunnel()
                self._spawn_tunnel(s)
                spawns += 1
            elif self.port is not None and tunnel_health(self.paths, self.port, timeout=5):
                rc, out = vm_timeline(session)
                self._take([l for l in out.splitlines() if re.match(r"^\d+s ", l)], seen, base)
                self._stage(s, "endpoint_ready", float(s.get("t0", t0)))
                self.logger.log(f"READY after {time.time() - float(s.get('t0', t0)):.0f} s (stages {s['stages']})")
                return
            self._check_stop()
            time.sleep(health_s)
        raise StartError(f"boot did not finish in {BOOT_TIMEOUT // 60} min")

    def _take(self, lines: list[str], seen: int, base: float) -> int:
        for l in lines[seen:]:
            self.logger.log("  vm " + l)
            m = re.match(r"^(\d+)s \S+ (.*)$", l)
            for key, rx, label in (STEPS if m else ()):
                if re.match(rx, m.group(2)):
                    self.steps.append((key, min(base + int(m.group(1)), time.time()), label))
        return max(seen, len(lines))

    def _spawn_tunnel(self, s: dict[str, Any]) -> None:
        port = pick_port()
        with self.tlock:
            if self.no_tunnel:
                raise StartError("stop in progress")
            try:
                p = subprocess.Popen(
                    ssh_cmd(self.cfg.colab.session, self.paths.ssh_key, port),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=child_env(),
                    creationflags=CHILD_FLAGS,
                    start_new_session=not IS_WIN,
                )
            except OSError as e:
                raise StartError(f"cannot run ssh: {e}")
            self.tunnel, self.port = p, port
        threading.Thread(target=drain_stderr, args=(p, self.logger, self._auto_created), daemon=True).start()
        s["tunnel"] = {"pid": p.pid, "start": proc_start(p.pid), "port": port}
        save_session(self.paths, s)

    def _auto_created(self) -> None:
        session = self.cfg.colab.session
        self.logger.log(f"ALERT: colab ssh --proxy-mode created a new session {session}; stopping it")
        threading.Thread(target=colab_run, args=(["stop", "-s", session], 120), daemon=True).start()
        self.request_stop("tunnel created a session")

    def _kill_tunnel(self) -> None:
        p, self.tunnel = self.tunnel, None
        if p and p.poll() is None:
            kill_tree(p.pid)
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                kill_tree(p.pid, hard=True)

    def _tunnel_up(self, s: dict[str, Any]) -> bool:
        self._spawn_tunnel(s)
        t1 = time.time()
        while time.time() - t1 < 60:
            if self.port is not None and tunnel_health(self.paths, self.port):
                return True
            if self.tunnel is None or self.tunnel.poll() is not None:
                break
            self._check_stop()
            time.sleep(1)
        self._kill_tunnel()
        return False

    def _repair(self) -> None:
        try:
            self.logger.log("tunnel exited, checking the session")
            session = self.cfg.colab.session
            s = self.sess or load_session(self.paths) or {"name": session}
            reason = "session gone"
            if session_alive(session):
                reason = "tunnel failed 3 times"
                for _ in range(3):
                    if self.state != "ready":
                        return
                    if self._tunnel_up(s):
                        with self.cond:
                            self.tunnel_ok = True
                            self.cond.notify_all()
                        self.logger.log("tunnel restored")
                        return
            self.logger.log(f"gpu vm unreachable ({reason})")
            self.request_stop(reason)
        except Exception as e:
            self.logger.log(f"tunnel repair failed: {e!r}")
            self.request_stop("tunnel repair failed")
        finally:
            self.repairing = False

    def tunnel_failed(self) -> bool:
        p = self.tunnel
        if p is None or p.poll() is None:
            return False
        self._start_repair()
        t0 = time.time()
        with self.cond:
            while self.state == "ready" and not self.tunnel_ok and time.time() - t0 < 90:
                self.cond.wait(1)
            return self.ready()

    def _start_repair(self) -> None:
        with self.cond:
            if self.state != "ready" or self.repairing:
                return
            self.repairing, self.tunnel_ok = True, False
        threading.Thread(target=self._repair, daemon=True).start()

    def monitor(self) -> None:
        while True:
            time.sleep(0.5)
            now = time.time()
            with self.cond:
                st, s, p = self.state, self.sess, self.tunnel
                max_h = self.cfg.colab.max_hours
                idle_m = self.cfg.colab.idle_min
                orphan_s = self.cfg.colab.orphan_boot_s
                orphan_boot = (
                    st == "starting"
                    and not self.explicit_up
                    and self.waiters == 0
                    and self.waiter_lost_at is not None
                    and now - self.waiter_lost_at > orphan_s
                )
            if st in ("starting", "ready") and s and now > float(s.get("t0", now)) + max_h * 3600:
                self.request_stop(f"hard stop after {max_h:.2f} h")
            elif orphan_boot:
                self.request_stop("all waiting clients disconnected")
            elif st == "ready":
                self.request_stop(f"idle {idle_m:.1f} min", only_idle=True)
                if p is not None and p.poll() is not None:
                    self._start_repair()
            elif st == "off" and self.paths.session_file.exists() and now - self.last_retry > 60:
                self.last_retry = now
                self.request_stop("retrying the stop")

    def shutdown(self) -> None:
        with self.cond:
            self.closing = True
            busy = self.state != "off" or self.paths.session_file.exists()
        if busy:
            self.logger.log("proxy exiting: stopping the VM first (Ctrl+C again leaves it to the watchdog)")
            self.request_stop("proxy exit")
            self.wait_off()

    def basalt_stats(self) -> dict[str, Any] | None:
        """Cached live snapshot from Basalt /status and /metrics (never blocks inference lock)."""
        if not self.ready() or not self.port:
            return None
        with self._stats_lock:
            now = time.time()
            if self._stats_cache is not None and now - self._stats_at < 0.8:
                return self._stats_cache
            port = self.port
            try:
                hdrs = {"Authorization": "Bearer " + read_api_key(self.paths)}
            except OSError:
                return None
            out: dict[str, Any] = {}
            for ep in ("/status", "/metrics"):
                try:
                    conn, r = http_call(port, "GET", ep, headers=hdrs, timeout=1.2)
                    raw = r.read() if r.status == 200 else b""
                    conn.close()
                    if raw:
                        import json

                        data = json.loads(raw)
                        if ep == "/status" and isinstance(data, dict):
                            out["busy"] = bool(data.get("busy"))
                            if data.get("tokens_per_s") is not None:
                                out["live_tok_s"] = data["tokens_per_s"]
                            elif data.get("tokens_per_s_mean") is not None:
                                out["live_tok_s"] = data["tokens_per_s_mean"]
                        elif ep == "/metrics" and isinstance(data, dict):
                            live = data.get("live") or {}
                            if live.get("queued") is not None:
                                out["queued"] = live["queued"]
                            reqs = data.get("requests") or []
                            last = next(
                                (x for x in reqs if isinstance(x, dict) and x.get("decode_tok_s") is not None),
                                None,
                            )
                            if last:
                                out["last_tok_s"] = last.get("decode_tok_s")
                                out["hit_rate"] = last.get("hit_rate")
                                out["pcie_share"] = last.get("pcie_share")
                                out["draft_len"] = last.get("draft_len")
                except (OSError, ValueError, http.client.HTTPException):
                    pass
            self._stats_at = time.time()
            self._stats_cache = out or None
            return self._stats_cache

    def status(self) -> dict[str, Any]:
        basalt = self.basalt_stats()
        with self.cond:
            s = self.sess or {}
            state = "ready (tunnel reconnecting)" if self.state == "ready" and not self.tunnel_ok else self.state
            alloc_ts = s.get("allocated_ts")
            ran = (time.time() - float(alloc_ts)) if (alloc_ts and self.state != "off") else None
            cu = (ran / 3600.0 * self.manifest.g4_cu_per_hour) if ran is not None else None
            steps_view = [
                {"key": k, "elapsed_s": max(0, int(at - self.begun)), "label": lbl}
                for k, at, lbl in self.steps
                if lbl
            ]
            return {
                "state": state,
                "stage": self.activity() if self.state == "starting" else "",
                "steps": steps_view,
                "session": self.cfg.colab.session,
                "model": self.cfg.basalt.effective_model_alias,
                "quant": self.cfg.basalt.quant,
                "uncensor": self.cfg.describe_uncensor(),
                "basalt_ref": self.cfg.basalt.ref or self.manifest.basalt_tag,
                "engine_summary": self.cfg.describe_engine(),
                "active": self.active,
                "served": self.served,
                "ran": ran,
                "cu": cu,
                "last_stop": self.last_stop,
                "idle_s": round(time.time() - self.last_activity),
                "idle_stop_min": self.cfg.colab.idle_min,
                "max_hours": self.cfg.colab.max_hours,
                "effort": self.cfg.basalt.effort,
                "wait_gpu": self.cfg.colab.wait_gpu,
                "strict_model": self.cfg.proxy.strict_model,
                "started": s.get("started"),
                "stages": s.get("stages"),
                "last_error": self.fail_msg or None,
                "pid": os.getpid(),
                "basalt": basalt,
            }

    def usage_summary(self, days: int = 30) -> dict[str, Any]:
        with self.cond:
            s = self.sess or {}
            alloc_ts = s.get("allocated_ts")
            ran = (time.time() - float(alloc_ts)) if (alloc_ts and self.state != "off") else None
            cu = (ran / 3600.0 * self.manifest.g4_cu_per_hour) if ran is not None else None
        return self.ledger.summary(days=days, active_ran_s=ran, active_cu=cu)
