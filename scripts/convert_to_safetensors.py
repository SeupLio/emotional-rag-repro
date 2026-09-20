"""Convert legacy ``pytorch_model.bin`` checkpoints to safetensors.

transformers >= 5.x refuses to ``torch.load`` a ``.bin`` checkpoint with
torch < 2.6 (CVE-2025-32434).  Rather than downgrading transformers or
upgrading torch (both heavy), we convert once and load the safetensors.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "models" / "bge-base-zh-v1.5",
    ROOT / "models" / "bge-base-en-v1.5",
    ROOT / "models" / "hubert-base-superb-er",
]


def convert(d: Path) -> bool:
    dst = d / "model.safetensors"
    src = d / "pytorch_model.bin"
    if dst.exists() and dst.stat().st_size > 1024:
        print(f"[skip ] {d.name}: already converted")
        return True
    if not src.exists():
        print(f"[miss ] {d.name}: no pytorch_model.bin")
        return False
    sd = torch.load(str(src), map_location="cpu", weights_only=True)
    sd = {k: (v.contiguous() if hasattr(v, "contiguous") else v) for k, v in sd.items()}
    save_file(sd, str(dst), metadata={"format": "pt"})
    print(f"[ok   ] {d.name}: {len(sd)} tensors -> {dst.stat().st_size/1e6:.1f} MB")
    return True


if __name__ == "__main__":
    ok = 0
    for t in TARGETS:
        try:
            ok += convert(t)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail ] {t.name}: {type(exc).__name__}: {exc}")
    print("converted", ok)
    sys.exit(0)
