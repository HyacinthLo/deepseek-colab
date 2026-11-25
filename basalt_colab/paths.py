"""XDG config/state path resolution and one-time legacy state/ migration."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

IS_WIN = os.name == "nt"
PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True, slots=True)
class Paths:
    config_dir: Path
    state_dir: Path
    vm_dir: Path

    @property
    def config_file(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def key_file(self) -> Path:
        return self.state_dir / "api_key"

    @property
    def ssh_key(self) -> Path:
        return self.state_dir / "id_ed25519"

    @property
    def session_file(self) -> Path:
        return self.state_dir / "session.json"

    @property
    def lock_file(self) -> Path:
        return self.state_dir / "serve.lock"

    @property
    def log_file(self) -> Path:
        return self.state_dir / "basalt-colab.log"

    @property
    def usage_file(self) -> Path:
        return self.state_dir / "usage.jsonl"

    @property
    def opencode_file(self) -> Path:
        return self.state_dir / "opencode.json"

    @property
    def serve_out_file(self) -> Path:
        return self.state_dir / "serve.out"

    def ensure_dirs(self) -> None:
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if not IS_WIN:
            try:
                os.chmod(self.state_dir, 0o700)
            except OSError:
                pass


def get_paths(home_override: str | Path | None = None, vm_override: str | Path | None = None) -> Paths:
    """Resolves config, state, and VM directories (respecting BASALT_COLAB_HOME and XDG vars)."""
    env_home = home_override if home_override is not None else os.environ.get("BASALT_COLAB_HOME")
    if env_home:
        base = Path(env_home).expanduser().resolve()
        config_dir = base
        state_dir = base / "state"
    elif IS_WIN:
        appdata = Path(os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming"))
        local = Path(os.environ.get("LOCALAPPDATA") or appdata)
        config_dir = appdata / "basalt-colab"
        state_dir = local / "basalt-colab"
    else:
        xdg_cfg = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
        xdg_state = Path(os.environ.get("XDG_STATE_HOME") or (Path.home() / ".local" / "state"))
        config_dir = xdg_cfg / "basalt-colab"
        state_dir = xdg_state / "basalt-colab"

    env_vm = vm_override if vm_override is not None else os.environ.get("BASALT_COLAB_VM_DIR")
    vm_dir = Path(env_vm).expanduser().resolve() if env_vm else (PACKAGE_DIR / "vm")
    return Paths(config_dir=config_dir, state_dir=state_dir, vm_dir=vm_dir)


def migrate_legacy_state(paths: Paths, legacy_dir: Path | None = None) -> list[str]:
    """One-time safe copy of legacy ./state/{api_key,id_ed25519,id_ed25519.pub} into XDG state dir."""
    if legacy_dir is None:
        candidates = [Path.cwd() / "state", PACKAGE_DIR.parent / "state"]
        legacy_dir = next((c for c in candidates if c.is_dir()), None)
    if not legacy_dir or not legacy_dir.is_dir():
        return []
    try:
        if legacy_dir.resolve() == paths.state_dir.resolve():
            return []
    except OSError:
        return []

    paths.ensure_dirs()
    migrated: list[str] = []
    for name, mode in (("api_key", 0o600), ("id_ed25519", 0o600), ("id_ed25519.pub", 0o644)):
        src = legacy_dir / name
        dst = paths.state_dir / name
        if src.is_file() and not dst.exists():
            shutil.copy2(src, dst)
            if not IS_WIN:
                try:
                    os.chmod(dst, mode)
                except OSError:
                    pass
            migrated.append(name)
    return migrated
