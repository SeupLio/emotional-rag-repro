"""Download one shard of the RAVDESS parquet mirror (audio bytes + labels).

``xbgoose/ravdess`` stores RAVDESS as two parquet shards (~167 MB each); one
shard is enough to validate the acoustic emotion channel.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_fetch import download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = ("https://hf-mirror.com/datasets/xbgoose/ravdess/resolve/main/data/"
       "train-00000-of-00002-94d632c9f1f51bbe.parquet")


def main() -> int:
    dst_dir = os.path.join(ROOT, "raw", "ravdess_parquet")
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "train-00000.parquet")
    if not (os.path.exists(dst) and os.path.getsize(dst) > 100 * 1e6):
        download(URL, dst, nthreads=8, expect=166845666)
    print("size", os.path.getsize(dst) if os.path.exists(dst) else 0, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
