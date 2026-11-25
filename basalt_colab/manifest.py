"""Loads the pinned Basalt & model manifest (basalt_colab/vm/manifest.json)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PackSpec:
    quant: str
    url: str
    model_alias: str
    size_gb: int
    kld: float | None = None
    top1: float | None = None
    estimated_speed_tok_s: float = 150.0
    description: str = ""


@dataclass(frozen=True, slots=True)
class Manifest:
    basalt_repo: str
    basalt_tag: str
    basalt_commit: str
    prebuilt_url: str
    prebuilt_sha256: str
    pack_url: str
    pack_model_id: str
    abliteration_url: str
    abliteration_sha256: str
    colab_cli: str
    g4_cu_per_hour: float
    estimated_speed_tok_s: float
    packs: dict[str, PackSpec] = field(default_factory=dict)

    def pack(self, quant: str) -> PackSpec | None:
        return self.packs.get(quant.strip().lower())


def load_manifest(vm_dir: Path) -> Manifest:
    path = vm_dir / "manifest.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    packs: dict[str, PackSpec] = {}
    for q, info in (raw.get("packs") or {}).items():
        packs[q.lower()] = PackSpec(
            quant=q.lower(),
            url=str(info.get("url", "")),
            model_alias=str(info.get("model_alias", "")),
            size_gb=int(info.get("size_gb", 0)),
            kld=None if info.get("kld") is None else float(info["kld"]),
            top1=None if info.get("top1") is None else float(info["top1"]),
            estimated_speed_tok_s=float(info.get("estimated_speed_tok_s", raw.get("estimated_speed_tok_s", 160.4))),
            description=str(info.get("description", "")),
        )
    return Manifest(
        basalt_repo=raw["basalt_repo"],
        basalt_tag=raw["basalt_tag"],
        basalt_commit=raw["basalt_commit"],
        prebuilt_url=raw.get("prebuilt_url", ""),
        prebuilt_sha256=raw.get("prebuilt_sha256", ""),
        pack_url=raw["pack_url"],
        pack_model_id=raw["pack_model_id"],
        abliteration_url=raw["abliteration_url"],
        abliteration_sha256=raw["abliteration_sha256"],
        colab_cli=raw.get("colab_cli", "google-colab-cli==0.7.5"),
        g4_cu_per_hour=float(raw.get("g4_cu_per_hour", 8.9)),
        estimated_speed_tok_s=float(raw.get("estimated_speed_tok_s", 160.4)),
        packs=packs,
    )
