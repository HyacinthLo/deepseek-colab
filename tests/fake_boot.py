"""Stand-in for vm/boot.sh: validates bundle, writes timeline, runs mock_basalt.py and real vm/gate.py."""
import json, os, subprocess, sys, time, urllib.request

C = os.getcwd()
R, V = os.path.join(C, "res"), os.path.join(C, "vm")
TESTS = os.path.dirname(os.path.abspath(__file__))
T0 = time.time()
MODE = os.environ.get("FAKE_MODE", "ok")
BOOT_S = float(os.environ.get("FAKE_BOOT_S", "2"))


def st(msg):
    with open(os.path.join(R, "timeline.txt"), "a") as f:
        f.write("%ds %s %s\n" % (time.time() - T0, time.strftime("%H:%M:%S"), msg))


def main():
    with open(os.path.join(R, "boot.pgid"), "w") as f:
        f.write(str(os.getpgid(0)))
    keep = (
        "MODEL_ALIAS", "CTX", "PARALLEL", "KV_TYPE", "VISION_MODE", "SPEC_MAX", "DRAFT_EXPERTS",
        "VRAM_RESERVE", "RAM_BUDGET", "UNCENSOR", "MAX_HOURS", "CONV_CACHE_MIB", "BASALT_REF",
        "BASALT_SAMPLING_JSON",
    )
    env = {k: v for k, v in os.environ.items() if k in keep or k.startswith("X_")}
    env["FAKE_MODE"] = MODE
    files = [os.path.relpath(os.path.join(d, f), C) for d, _, fs in os.walk(V) for f in fs]
    want = [
        "secrets/key", "vm/manifest.json", "vm/boot.sh", "vm/serve.sh", "vm/build.sh",
        "vm/patch_basalt.py", "vm/basalt_jobs.py", "vm/abliterate_basalt.py",
        "vm/abliterate.c", "vm/pfetch.py", "vm/gate.py",
    ]
    check = {
        "missing": [p for p in want if not os.path.exists(os.path.join(C, p))],
        "crlf": [f for f in files + ["secrets/key"] if b"\r" in open(os.path.join(C, f), "rb").read()],
        "secrets": sorted(os.listdir(os.path.join(C, "secrets"))),
        "secret_modes": sorted({oct(os.stat(os.path.join(C, "secrets", f)).st_mode & 0o777) for f in os.listdir(os.path.join(C, "secrets"))}),
        "vm_files": sorted(files),
        "env": env,
    }
    with open(os.path.join(R, "boot_check.json"), "w") as f:
        json.dump(check, f)
    st("boot start engine=basalt model=%s ref=%s mode=%s" % (env.get("MODEL_ALIAS", "DeepSeek-V4.1-Flash-1017-UD-IQ1_M"), env.get("BASALT_REF", "pinned"), MODE))
    time.sleep(BOOT_S / 2)
    if MODE == "fail":
        st("backbone FAILED")
        st("fetch/build FAILED, stopping")
        return 1
    st("basalt built")
    if env.get("UNCENSOR", "1") == "1":
        st("abliteration edit fetched")
    st("backbone fetched (basalt iq1_m)")
    if env.get("UNCENSOR", "1") == "1":
        st("abliteration applied")
    port = os.environ["FAKE_VM_INNER_PORT"]
    with open(os.path.join(R, "server.log"), "w") as log:
        p = subprocess.Popen(
            [sys.executable, os.path.join(TESTS, "mock_basalt.py"), port, "0.4"],
            env=dict(os.environ, MOCK_KEY=open("secrets/key").read().strip(), PYTHONDONTWRITEBYTECODE="1"),
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    with open(os.path.join(R, "server.pid"), "w") as f:
        f.write(str(p.pid))
    st("server started")
    for _ in range(100):
        try:
            with urllib.request.urlopen("http://127.0.0.1:%s/health" % port, timeout=2) as r:
                if r.status == 200:
                    st("basalt http ready")
                    break
        except OSError:
            pass
        if p.poll() is not None:
            st("server died")
            return 1
        time.sleep(0.2)
    st("weights loaded")
    st("pin: VRAM 95727 MiB, model 1.6G")
    st("engram done")
    time.sleep(BOOT_S / 2)
    open(os.path.join(R, "READY"), "w").close()
    with open(os.path.join(R, "first_token_at"), "w") as f:
        f.write(repr(time.time()))
    st("first token (warmup done)")
    with open(os.path.join(R, "gate_open_at"), "w") as f:
        f.write(repr(time.time()))
    g = subprocess.Popen(
        [sys.executable, os.path.join(V, "gate.py"), os.environ["FAKE_VM_PORT"], port],
        stdout=subprocess.PIPE,
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
    )
    st("gate open 18080" if "gate open" in g.stdout.readline().decode() else "gate FAILED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
