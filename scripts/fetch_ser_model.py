"""Fetch the speech-emotion-recognition backbone used by the multimodal extension.

``superb/hubert-base-superb-er`` (HuBERT-base fine-tuned on IEMOCAP for the
SUPERB Emotion Recognition task, 4 classes: neutral / happy / sad / angry).
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
REPO = "superb/hubert-base-superb-er"
SMALL = ["config.json", "preprocessor_config.json", "vocab.json", "tokenizer_config.json",
         "special_tokens_map.json"]
BIG = ["pytorch_model.bin"]


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
    print(f"[ok    ] {os.path.basename(dst)} ({len(data)/1e6:.2f} MB) in {time.time()-t0:.0f}s",
          flush=True)
    return True


def main() -> int:
    dst_dir = os.path.join(ROOT, "models", "hubert-base-superb-er")
    os.makedirs(dst_dir, exist_ok=True)
    for f in SMALL:
        try:
            fetch_plain(f"{MIRROR}/{REPO}/resolve/main/{f}", os.path.join(dst_dir, f))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip  ] {f}: {exc}", flush=True)
    for f in BIG:
        dst = os.path.join(dst_dir, f)
        if os.path.exists(dst) and os.path.getsize(dst) > 100 * 1e6:
            continue
        download(f"{MIRROR}/{REPO}/resolve/main/{f}", dst, nthreads=8)
    # list what we got
    for n in sorted(os.listdir(dst_dir)):
        print("  ", n, os.path.getsize(os.path.join(dst_dir, n)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
