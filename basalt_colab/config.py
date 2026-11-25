"""Layered declarative configuration (config.toml) for basalt-colab."""
from __future__ import annotations

import copy
import json
import os
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from basalt_colab.manifest import Manifest
from basalt_colab.paths import Paths

CONTEXT_ALIASES = {
    "32k": 32768,
    "64k": 65536,
    "128k": 131072,
    "131k": 131072,
    "256k": 262144,
    "262k": 262144,
    "512k": 524288,
    "524k": 524288,
    "1m": 1048576,
    "1024k": 1048576,
    "1048k": 1048576,
}

QUANT_ALIASES = {
    "iq1_m": "ud-iq1_m",
    "ud-iq1_m": "ud-iq1_m",
    "ud_iq1_m": "ud-iq1_m",
    "iq2_xxs": "ud-iq2_xxs",
    "ud-iq2_xxs": "ud-iq2_xxs",
    "ud_iq2_xxs": "ud-iq2_xxs",
    "q2_k_xl": "ud-q2_k_xl",
    "ud-q2_k_xl": "ud-q2_k_xl",
    "ud_q2_k_xl": "ud-q2_k_xl",
    "custom": "custom",
}

QUANT_DEFAULT_MODEL_ALIAS = {
    "ud-iq1_m": "DeepSeek-V4.1-Flash-1017-UD-IQ1_M",
    "ud-iq2_xxs": "DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS",
    "ud-q2_k_xl": "DeepSeek-V4.1-Flash-1017-UD-Q2_K_XL",
    "custom": "DeepSeek-V4.1-Flash-1017-Custom",
}

DEFAULT_QUANT = "ud-iq1_m"
DEFAULT_MODEL_ALIAS = QUANT_DEFAULT_MODEL_ALIAS[DEFAULT_QUANT]
VALID_QUANTS = ("ud-iq1_m", "ud-iq2_xxs", "ud-q2_k_xl", "custom")
VALID_DRAFT_EXPERTS = (0, 32, 64, 96, 128)
HOT_FIELDS = frozenset({
    "colab.idle_min",
    "colab.max_hours",
    "colab.wait_gpu",
    "colab.orphan_boot_s",
    "proxy.strict_model",
    "basalt.effort",
})


class ConfigError(ValueError):
    """Raised when configuration values fail schema or range validation."""


def quant_value(text: str) -> str | None:
    """Normalizes quantization preset names (ud-iq1_m, ud-iq2_xxs, ud-q2_k_xl, custom)."""
    return QUANT_ALIASES.get(str(text).strip().lower())


def context_value(text: str | int) -> int | None:
    """Parses per-slot context: 32K..1M or integer 4096..1048576 (rounded up to 256)."""
    if isinstance(text, int) and not isinstance(text, bool):
        return -(-text // 256) * 256 if 4096 <= text <= 1048576 else None
    t = str(text).strip().lower().replace(",", "").replace("_", "")
    if t in CONTEXT_ALIASES:
        return CONTEXT_ALIASES[t]
    if t.isdigit():
        v = int(t)
        return -(-v // 256) * 256 if 4096 <= v <= 1048576 else None
    return None


def short_context(n: int) -> str:
    preferred = {32768: "32K", 65536: "64K", 131072: "131K", 262144: "262K", 524288: "524K", 1048576: "1M"}
    return preferred.get(n, f"{n:,}")


def effort_value(text: str | int) -> str | None:
    """Validates DeepSeek V4.1 reasoning effort: 1-100, low, high, or max."""
    t = str(text).strip().lower()
    if t in ("low", "high", "max"):
        return t
    if t.isdigit() and 1 <= int(t) <= 100:
        return t
    return None


def _parse_bool(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on", "y"):
        return True
    if s in ("0", "false", "no", "off", "n"):
        return False
    raise ConfigError(f"invalid boolean value: {val!r}")


@dataclass(slots=True)
class ColabConfig:
    session: str = "basalt-ds41"
    idle_min: float = 10.0
    max_hours: float = 6.0
    min_balance: float = 5.0
    wait_gpu: bool = False
    orphan_boot_s: float = 30.0


@dataclass(slots=True)
class ProxyConfig:
    bind: str = "127.0.0.1"
    port: int = 18080
    strict_model: bool = False

    @property
    def host(self) -> str:
        return self.bind if self.bind not in ("", "0.0.0.0", "::") else "127.0.0.1"

    @property
    def loopback(self) -> bool:
        return self.bind in ("127.0.0.1", "localhost", "::1")


@dataclass(slots=True)
class SamplingConfig:
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None

    def to_basalt_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.temperature is not None:
            out["temperature"] = float(self.temperature)
        if self.top_p is not None:
            out["top-p"] = float(self.top_p)
        if self.top_k is not None:
            out["top-k"] = int(self.top_k)
        if self.min_p is not None:
            out["min-p"] = float(self.min_p)
        if self.presence_penalty is not None:
            out["presence-penalty"] = float(self.presence_penalty)
        return out


@dataclass(slots=True)
class BasaltConfig:
    quant: str = DEFAULT_QUANT
    pack_url: str = ""
    ref: str = ""
    model_alias: str = ""
    max_context: int = 262144
    parallel: int = 2
    kv: str = "fp8"
    vision: str = "dynamic"
    spec: int = 6
    draft_experts: int = 32
    conv_cache_mib: int = 32768
    vram_reserve_mib: int = 700
    resident_budget_gib: int = 0
    uncensor: bool = True
    uncensor_scale: float = 1.0
    effort: str = "50"
    workers: int = 64
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    @property
    def effective_model_alias(self) -> str:
        raw = self.model_alias.strip()
        if raw and raw.lower() != "auto":
            return raw
        return QUANT_DEFAULT_MODEL_ALIAS.get(self.quant, DEFAULT_MODEL_ALIAS)

    def resolved_pack_url(self, manifest: Manifest | None = None) -> str:
        if self.pack_url.strip():
            return self.pack_url.strip()
        if manifest is not None:
            spec = manifest.pack(self.quant)
            if spec and spec.url:
                return spec.url
            if self.quant == DEFAULT_QUANT and manifest.pack_url:
                return manifest.pack_url
        return ""


@dataclass(slots=True)
class Config:
    colab: ColabConfig = field(default_factory=ColabConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)
    basalt: BasaltConfig = field(default_factory=BasaltConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def clone(self) -> Config:
        return copy.deepcopy(self)

    def describe_engine(self) -> str:
        ref = f"ref={self.basalt.ref} · " if self.basalt.ref else ""
        return (
            f"{ref}{self.basalt.quant.upper()} · {short_context(self.basalt.max_context)}×{self.basalt.parallel} "
            f"{self.basalt.kv.upper()} KV · vision {self.basalt.vision}"
        )

    def describe_uncensor(self) -> str:
        if not self.basalt.uncensor or self.basalt.uncensor_scale <= 0:
            return "original"
        if abs(self.basalt.uncensor_scale - 1.0) < 1e-6:
            return "uncensored"
        return f"uncensored ×{self.basalt.uncensor_scale:g}"

    def summary(self) -> str:
        return " · ".join((
            self.basalt.effective_model_alias,
            self.describe_engine(),
            f"effort {self.basalt.effort}",
            f"idle {self.colab.idle_min:g} min",
            f"cap {self.colab.max_hours:g} h",
            self.describe_uncensor(),
        ))

    def cold_signature(self) -> dict[str, Any]:
        """Fields that require a VM restart if changed while G4 is running."""
        b = self.basalt
        return {
            "quant": b.quant,
            "pack_url": b.pack_url,
            "ref": b.ref,
            "model_alias": b.effective_model_alias,
            "max_context": b.max_context,
            "parallel": b.parallel,
            "kv": b.kv,
            "vision": b.vision,
            "spec": b.spec,
            "draft_experts": b.draft_experts,
            "conv_cache_mib": b.conv_cache_mib,
            "vram_reserve_mib": b.vram_reserve_mib,
            "resident_budget_gib": b.resident_budget_gib,
            "uncensor": b.uncensor,
            "uncensor_scale": b.uncensor_scale,
            "sampling": b.sampling.to_basalt_dict(),
        }

    def boot_vars(self, extra_env: dict[str, str] | None = None, manifest: Manifest | None = None) -> dict[str, str]:
        """Builds shell-safe environment variables passed to vm/boot.sh."""
        b = self.basalt
        pack_url = b.resolved_pack_url(manifest)
        if b.quant in ("ud-q2_k_xl", "custom") and not pack_url:
            raise ConfigError(
                f"basalt.quant={b.quant!r} requires basalt.pack_url pointing to your .basalt pack "
                f"(build UD-Q2_K_XL with scripts/build_q2k_xl_pack.py)"
            )
        uncensor_active = b.uncensor and b.uncensor_scale > 0.0
        env: dict[str, str] = {
            "MODEL_QUANT": b.quant,
            "MODEL_ALIAS": b.effective_model_alias,
            "CTX": str(b.max_context),
            "PARALLEL": str(b.parallel),
            "KV_TYPE": b.kv,
            "VISION_MODE": b.vision,
            "SPEC_MAX": str(b.spec),
            "DRAFT_EXPERTS": str(b.draft_experts),
            "CONV_CACHE_MIB": str(b.conv_cache_mib),
            "VRAM_RESERVE": str(b.vram_reserve_mib),
            "RAM_BUDGET": str(b.resident_budget_gib),
            "UNCENSOR": "1" if uncensor_active else "0",
            "UNCENSOR_SCALE": f"{b.uncensor_scale:g}",
            "WORKERS": str(b.workers),
            "MAX_HOURS": str(self.colab.max_hours),
        }
        if pack_url:
            env["MODEL_URL"] = pack_url
        if b.ref:
            env["BASALT_REF"] = b.ref
        sampling_dict = b.sampling.to_basalt_dict()
        if sampling_dict:
            env["BASALT_SAMPLING_JSON"] = json.dumps(sampling_dict, separators=(",", ":"))
        for k in ("BASALT_PREBUILT_URL", "BASALT_PREBUILT_SHA256"):
            if os.environ.get(k):
                env[k] = os.environ[k]
        if extra_env:
            env.update(extra_env)
        for kv in os.environ.get("BASALT_COLAB_BOOT_ENV", "").split():
            k, sep, v = kv.partition("=")
            if not sep:
                raise ConfigError(f'BASALT_COLAB_BOOT_ENV entry without "=": {kv}')
            env[k] = v
        for k, v in env.items():
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", k) or "'" in v:
                raise ConfigError(f"bad boot env {k}: names must be shell words, values must not contain single quotes")
        return env

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        if not isinstance(data, dict):
            raise ConfigError("config root must be a table/dict")
        c_raw = data.get("colab") or {}
        p_raw = data.get("proxy") or {}
        b_raw = data.get("basalt") or {}
        s_raw = b_raw.get("sampling") or {}
        if not all(isinstance(x, dict) for x in (c_raw, p_raw, b_raw, s_raw)):
            raise ConfigError("config sections [colab], [proxy], [basalt], [basalt.sampling] must be tables")

        colab = ColabConfig(
            session=str(c_raw.get("session", ColabConfig.session)).strip(),
            idle_min=float(c_raw.get("idle_min", ColabConfig.idle_min)),
            max_hours=float(c_raw.get("max_hours", ColabConfig.max_hours)),
            min_balance=float(c_raw.get("min_balance", ColabConfig.min_balance)),
            wait_gpu=_parse_bool(c_raw.get("wait_gpu", ColabConfig.wait_gpu)),
            orphan_boot_s=float(c_raw.get("orphan_boot_s", ColabConfig.orphan_boot_s)),
        )
        proxy = ProxyConfig(
            bind=str(p_raw.get("bind", ProxyConfig.bind)).strip(),
            port=int(p_raw.get("port", ProxyConfig.port)),
            strict_model=_parse_bool(p_raw.get("strict_model", ProxyConfig.strict_model)),
        )
        sampling = SamplingConfig(
            temperature=None if s_raw.get("temperature") in (None, "") else float(s_raw["temperature"]),
            top_p=None if s_raw.get("top_p") in (None, "") else float(s_raw["top_p"]),
            top_k=None if s_raw.get("top_k") in (None, "") else int(s_raw["top_k"]),
            min_p=None if s_raw.get("min_p") in (None, "") else float(s_raw["min_p"]),
            presence_penalty=None if s_raw.get("presence_penalty") in (None, "") else float(s_raw["presence_penalty"]),
        )
        q_in = b_raw.get("quant", BasaltConfig.quant)
        q_parsed = quant_value(q_in)
        if q_parsed is None:
            raise ConfigError(f"basalt.quant must be one of {VALID_QUANTS}, got {q_in!r}")
        ctx_in = b_raw.get("max_context", BasaltConfig.max_context)
        ctx_parsed = context_value(ctx_in)
        if ctx_parsed is None:
            raise ConfigError(f"basalt.max_context must be 4096..1048576 (or 32K..1M), got {ctx_in!r}")
        basalt = BasaltConfig(
            quant=q_parsed,
            pack_url=str(b_raw.get("pack_url", BasaltConfig.pack_url)).strip(),
            ref=str(b_raw.get("ref", BasaltConfig.ref)).strip(),
            model_alias=str(b_raw.get("model_alias", BasaltConfig.model_alias)).strip(),
            max_context=ctx_parsed,
            parallel=int(b_raw.get("parallel", BasaltConfig.parallel)),
            kv=str(b_raw.get("kv", BasaltConfig.kv)).strip().lower(),
            vision=str(b_raw.get("vision", BasaltConfig.vision)).strip().lower(),
            spec=int(b_raw.get("spec", BasaltConfig.spec)),
            draft_experts=int(b_raw.get("draft_experts", BasaltConfig.draft_experts)),
            conv_cache_mib=int(b_raw.get("conv_cache_mib", BasaltConfig.conv_cache_mib)),
            vram_reserve_mib=int(b_raw.get("vram_reserve_mib", BasaltConfig.vram_reserve_mib)),
            resident_budget_gib=int(b_raw.get("resident_budget_gib", BasaltConfig.resident_budget_gib)),
            uncensor=_parse_bool(b_raw.get("uncensor", BasaltConfig.uncensor)),
            uncensor_scale=float(b_raw.get("uncensor_scale", BasaltConfig.uncensor_scale)),
            effort=str(b_raw.get("effort", BasaltConfig.effort)).strip().lower(),
            workers=int(b_raw.get("workers", BasaltConfig.workers)),
            sampling=sampling,
        )
        cfg = cls(colab=colab, proxy=proxy, basalt=basalt)
        validate_config(cfg)
        return cfg


def validate_config(cfg: Config) -> None:
    if not re.match(r"^[A-Za-z0-9._-]+$", cfg.colab.session):
        raise ConfigError(f"invalid colab.session: {cfg.colab.session!r}")
    if not (0.01 <= cfg.colab.idle_min <= 120):
        raise ConfigError("colab.idle_min must be between 0.01 and 120")
    if not (0.001 <= cfg.colab.max_hours <= 24):
        raise ConfigError("colab.max_hours must be between 0.001 and 24")
    if cfg.colab.min_balance < 0:
        raise ConfigError("colab.min_balance must be >= 0")
    if cfg.colab.orphan_boot_s < 0.5:
        raise ConfigError("colab.orphan_boot_s must be >= 0.5")

    if not (1 <= cfg.proxy.port <= 65535):
        raise ConfigError("proxy.port must be 1..65535")

    b = cfg.basalt
    if b.quant not in VALID_QUANTS:
        raise ConfigError(f"basalt.quant must be one of {VALID_QUANTS}, got {b.quant!r}")
    if b.pack_url:
        if "'" in b.pack_url or not re.match(r"^https?://", b.pack_url):
            raise ConfigError(f"basalt.pack_url must be an http(s) URL without single quotes: {b.pack_url!r}")
    if b.model_alias and "'" in b.model_alias:
        raise ConfigError(f"invalid basalt.model_alias: {b.model_alias!r}")
    if b.ref and not re.match(r"^[A-Za-z0-9._/@-]+$", b.ref):
        raise ConfigError(f"invalid basalt.ref: {b.ref!r}")
    if not (4096 <= b.max_context <= 1048576):
        raise ConfigError("basalt.max_context must be 4096..1048576")
    if not (1 <= b.parallel <= 4):
        raise ConfigError("basalt.parallel must be 1..4")
    if b.kv not in ("fp8", "fp16"):
        raise ConfigError(f"basalt.kv must be 'fp8' or 'fp16', got {b.kv!r}")
    if b.vision not in ("dynamic", "off"):
        raise ConfigError(f"basalt.vision must be 'dynamic' or 'off', got {b.vision!r}")
    if not (0 <= b.spec <= 8):
        raise ConfigError("basalt.spec must be 0..8")
    if b.draft_experts not in VALID_DRAFT_EXPERTS:
        raise ConfigError(f"basalt.draft_experts must be one of {VALID_DRAFT_EXPERTS}, got {b.draft_experts!r}")
    if not (0 <= b.conv_cache_mib <= 131072):
        raise ConfigError("basalt.conv_cache_mib must be 0..131072")
    if not (256 <= b.vram_reserve_mib <= 16384):
        raise ConfigError("basalt.vram_reserve_mib must be 256..16384")
    if b.resident_budget_gib != 0 and not (16 <= b.resident_budget_gib <= 160):
        raise ConfigError("basalt.resident_budget_gib must be 0 (auto) or 16..160")
    if not (0.0 <= b.uncensor_scale <= 2.0):
        raise ConfigError("basalt.uncensor_scale must be between 0.0 and 2.0")
    if effort_value(b.effort) is None:
        raise ConfigError(f"basalt.effort must be 1-100, low, high, or max; got {b.effort!r}")
    if not (4 <= b.workers <= 128):
        raise ConfigError("basalt.workers must be 4..128")


def apply_env_overrides(cfg: Config) -> Config:
    """Applies BASALT_COLAB_* environment variable overrides on top of config.toml."""
    c = cfg.clone()
    env = os.environ
    if env.get("BASALT_COLAB_SESSION"):
        c.colab.session = env["BASALT_COLAB_SESSION"].strip()
    if env.get("BASALT_COLAB_IDLE_MIN"):
        c.colab.idle_min = float(env["BASALT_COLAB_IDLE_MIN"])
    if env.get("BASALT_COLAB_MAX_HOURS"):
        c.colab.max_hours = float(env["BASALT_COLAB_MAX_HOURS"])
    if env.get("BASALT_COLAB_MIN_BALANCE"):
        c.colab.min_balance = float(env["BASALT_COLAB_MIN_BALANCE"])
    if env.get("BASALT_COLAB_WAIT_GPU"):
        c.colab.wait_gpu = _parse_bool(env["BASALT_COLAB_WAIT_GPU"])
    if env.get("BASALT_COLAB_ORPHAN_BOOT_S"):
        c.colab.orphan_boot_s = float(env["BASALT_COLAB_ORPHAN_BOOT_S"])
    if env.get("BASALT_COLAB_BIND"):
        c.proxy.bind = env["BASALT_COLAB_BIND"].strip()
    if env.get("BASALT_COLAB_PORT"):
        c.proxy.port = int(env["BASALT_COLAB_PORT"])
    if env.get("BASALT_COLAB_STRICT_MODEL"):
        c.proxy.strict_model = _parse_bool(env["BASALT_COLAB_STRICT_MODEL"])
    if env.get("BASALT_COLAB_QUANT"):
        q = quant_value(env["BASALT_COLAB_QUANT"])
        if q is None:
            raise ConfigError(f"invalid BASALT_COLAB_QUANT: {env['BASALT_COLAB_QUANT']!r}")
        c.basalt.quant = q
    if env.get("BASALT_COLAB_PACK_URL"):
        c.basalt.pack_url = env["BASALT_COLAB_PACK_URL"].strip()
    if env.get("BASALT_COLAB_REF"):
        c.basalt.ref = env["BASALT_COLAB_REF"].strip()
    if env.get("BASALT_COLAB_MODEL_ALIAS"):
        c.basalt.model_alias = env["BASALT_COLAB_MODEL_ALIAS"].strip()
    if env.get("BASALT_COLAB_MAX_CONTEXT"):
        parsed = context_value(env["BASALT_COLAB_MAX_CONTEXT"])
        if parsed is None:
            raise ConfigError(f"invalid BASALT_COLAB_MAX_CONTEXT: {env['BASALT_COLAB_MAX_CONTEXT']!r}")
        c.basalt.max_context = parsed
    if env.get("BASALT_COLAB_PARALLEL"):
        c.basalt.parallel = int(env["BASALT_COLAB_PARALLEL"])
    if env.get("BASALT_COLAB_KV"):
        c.basalt.kv = env["BASALT_COLAB_KV"].strip().lower()
    if env.get("BASALT_COLAB_VISION"):
        c.basalt.vision = env["BASALT_COLAB_VISION"].strip().lower()
    if env.get("BASALT_COLAB_UNCENSOR"):
        c.basalt.uncensor = _parse_bool(env["BASALT_COLAB_UNCENSOR"])
    if env.get("BASALT_COLAB_UNCENSOR_SCALE"):
        c.basalt.uncensor_scale = float(env["BASALT_COLAB_UNCENSOR_SCALE"])
    if env.get("BASALT_COLAB_EFFORT"):
        c.basalt.effort = env["BASALT_COLAB_EFFORT"].strip().lower()
    validate_config(c)
    return c


def parse_toml(text: str) -> Config:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"TOML parse error: {e}") from e
    return Config.from_dict(raw)


def _toml_val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v == int(v) else f"{v:g}"
    return json.dumps(str(v))


def dump_toml(cfg: Config) -> str:
    """Serializes Config to a clean, human-readable TOML document."""
    c, p, b, s = cfg.colab, cfg.proxy, cfg.basalt, cfg.basalt.sampling
    lines = [
        "# basalt-colab configuration (Basalt v0.5.7 + DeepSeek-V4.1-Flash on Colab G4)",
        "",
        "[colab]",
        f"session = {_toml_val(c.session)}",
        f"idle_min = {_toml_val(c.idle_min)}",
        f"max_hours = {_toml_val(c.max_hours)}",
        f"min_balance = {_toml_val(c.min_balance)}",
        f"wait_gpu = {_toml_val(c.wait_gpu)}",
        f"orphan_boot_s = {_toml_val(c.orphan_boot_s)}",
        "",
        "[proxy]",
        f"bind = {_toml_val(p.bind)}",
        f"port = {_toml_val(p.port)}",
        f"strict_model = {_toml_val(p.strict_model)}",
        "",
        "[basalt]",
        f"quant = {_toml_val(b.quant)}",
        f"pack_url = {_toml_val(b.pack_url)}",
        f"ref = {_toml_val(b.ref)}",
        f"model_alias = {_toml_val(b.model_alias)}",
        f"max_context = {_toml_val(b.max_context)}",
        f"parallel = {_toml_val(b.parallel)}",
        f"kv = {_toml_val(b.kv)}",
        f"vision = {_toml_val(b.vision)}",
        f"spec = {_toml_val(b.spec)}",
        f"draft_experts = {_toml_val(b.draft_experts)}",
        f"conv_cache_mib = {_toml_val(b.conv_cache_mib)}",
        f"vram_reserve_mib = {_toml_val(b.vram_reserve_mib)}",
        f"resident_budget_gib = {_toml_val(b.resident_budget_gib)}",
        f"uncensor = {_toml_val(b.uncensor)}",
        f"uncensor_scale = {_toml_val(b.uncensor_scale)}",
        f"effort = {_toml_val(b.effort)}",
        f"workers = {_toml_val(b.workers)}",
        "",
        "[basalt.sampling]",
    ]
    samp_fields = (
        ("temperature", s.temperature),
        ("top_p", s.top_p),
        ("top_k", s.top_k),
        ("min_p", s.min_p),
        ("presence_penalty", s.presence_penalty),
    )
    has_samp = False
    for k, v in samp_fields:
        if v is not None:
            lines.append(f"{k} = {_toml_val(v)}")
            has_samp = True
    if not has_samp:
        lines.append("# temperature = 1.0")
        lines.append("# top_p = 0.95")
    lines.append("")
    return "\n".join(lines)


def load_config(paths: Paths, apply_env: bool = True) -> Config:
    if paths.config_file.is_file():
        cfg = parse_toml(paths.config_file.read_text(encoding="utf-8"))
    else:
        cfg = Config()
    return apply_env_overrides(cfg) if apply_env else cfg


def save_config(cfg: Config, paths: Paths) -> None:
    validate_config(cfg)
    paths.ensure_dirs()
    tmp = paths.config_file.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(dump_toml(cfg), encoding="utf-8")
    os.replace(tmp, paths.config_file)


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def diff_hot_cold(old: Config, new: Config) -> tuple[list[str], list[str]]:
    """Returns (hot_changed_keys, cold_changed_keys) between two Config objects."""
    a, b = _flatten(old.to_dict()), _flatten(new.to_dict())
    hot: list[str] = []
    cold: list[str] = []
    for k in sorted(set(a) | set(b)):
        if a.get(k) != b.get(k):
            (hot if k in HOT_FIELDS else cold).append(k)
    return hot, cold
