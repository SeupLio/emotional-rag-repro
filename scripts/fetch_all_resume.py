"""Download every remaining artefact with the resumable fetcher.

Run it as many times as needed -- completed chunk ranges are recorded on disk,
so each invocation only pulls what is still missing.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from resume_fetch import download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = "https://hf-mirror.com"

JOBS = [
    # (dst, url, expect, threads)  -- ordered by how much the report needs them
    (os.path.join(ROOT, "raw", "ravdess_parquet", "train-00000.parquet"),
     f"{M}/datasets/xbgoose/ravdess/resolve/main/data/"
     "train-00000-of-00002-94d632c9f1f51bbe.parquet", 166845666, 8),
    (os.path.join(ROOT, "models", "bge-base-en-v1.5", "pytorch_model.bin"),
     f"{M}/BAAI/bge-base-en-v1.5/resolve/main/pytorch_model.bin", 437997357, 8),
    (os.path.join(ROOT, "models", "multilingual_go_emotions", "model.safetensors"),
     f"{M}/AnasAlokla/multilingual_go_emotions/resolve/main/model.safetensors", 711523440, 10),
]


def main() -> int:
    for dst, url, expect, n in JOBS:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == expect:
            print(f"[skip ] {os.path.basename(dst)}", flush=True)
            continue
        for attempt in range(1, 7):
            print(f"[get  ] {os.path.basename(dst)} try{attempt}", flush=True)
            t0 = time.time()
            try:
                ok = download(url, dst, nthreads=n, expect=expect, max_seconds=1500)
            except Exception as exc:  # noqa: BLE001
                print(f"   exc {type(exc).__name__}: {exc}", flush=True)
                ok = False
            print(f"   -> {ok} in {time.time()-t0:.0f}s", flush=True)
            if ok:
                break
            time.sleep(3)
    return 0


if __name__ == "__main__":
    sys.exit(main())
