"""Fetch BAAI/bge-base-en-v1.5 with integrity verification.

InCharacter contains both English and Chinese characters; ``bge-base-zh-v1.5``
serves the Chinese half and this encoder the English half.  Both are 768-d BERT
models with identical pooling, so the retrieval geometry is unchanged.
"""
from __future__ import annotations

import os
import sys
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_fetch import download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "models", "bge-base-en-v1.5")
STALE = os.path.join(ROOT, "raw", "_stale")
REPO = "BAAI/bge-base-en-v1.5"
SMALL = ["config.json", "special_tokens_map.json", "tokenizer.json",
         "tokenizer_config.json", "vocab.txt"]
EXPECT = 437997357


def fetch_plain(url: str, dst: str) -> bool:
    import urllib.request
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)
    return True


def main() -> int:
    os.makedirs(D, exist_ok=True)
    os.makedirs(STALE, exist_ok=True)
    for f in SMALL:
        try:
            fetch_plain(f"https://hf-mirror.com/{REPO}/resolve/main/{f}",
                        os.path.join(D, f))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f}: {exc}", flush=True)

    out = os.path.join(D, "model.safetensors")
    if os.path.exists(out) and os.path.getsize(out) > 400e6:
        print("[skip] already converted", flush=True)
        return 0

    for attempt in range(1, 5):
        dst = os.path.join(D, "pytorch_model.bin")
        if os.path.exists(dst):
            os.replace(dst, os.path.join(STALE, f"bgeen_bad_{attempt}.bin"))
        t0 = time.time()
        download(f"https://hf-mirror.com/{REPO}/resolve/main/pytorch_model.bin",
                 dst, nthreads=8, expect=EXPECT)
        print(f"[try {attempt}] {time.time()-t0:.0f}s", flush=True)
        try:
            sd = torch.load(dst, map_location="cpu", weights_only=True)
            sd = {k: v.contiguous() for k, v in sd.items()}
            save_file(sd, out, metadata={"format": "pt"})
            print(f"[ok  ] {len(sd)} tensors -> {out}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[bad ] {type(exc).__name__}: {exc}", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
