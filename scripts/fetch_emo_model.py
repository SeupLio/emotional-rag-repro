"""Fetch the multilingual Go-Emotions classifier used as the text emotion channel.

``AnasAlokla/multilingual_go_emotions`` is a multilingual BERT fine-tuned on the
28-label Go Emotions taxonomy; those labels map cleanly onto Plutchik's wheel,
and one model serves both the English and the Chinese characters of
InCharacter.  It ships ``model.safetensors``, so transformers >= 5.x loads it
without touching ``torch.load``.
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_fetch import download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = "https://hf-mirror.com"
REPO = "AnasAlokla/multilingual_go_emotions"
SMALL = ["config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]


def fetch_plain(url: str, dst: str) -> bool:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)
    print(f"[ok  ] {os.path.basename(dst)} ({len(data)/1e6:.2f} MB) in {time.time()-t0:.0f}s",
          flush=True)
    return True


def main() -> int:
    d = os.path.join(ROOT, "models", "multilingual_go_emotions")
    os.makedirs(d, exist_ok=True)
    for f in SMALL:
        try:
            fetch_plain(f"{MIRROR}/{REPO}/resolve/main/{f}", os.path.join(d, f))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] {f}: {exc}", flush=True)
    dst = os.path.join(d, "model.safetensors")
    if not (os.path.exists(dst) and os.path.getsize(dst) > 100e6):
        download(f"{MIRROR}/{REPO}/resolve/main/model.safetensors", dst, nthreads=8)
    print("size", os.path.getsize(dst) if os.path.exists(dst) else 0, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
