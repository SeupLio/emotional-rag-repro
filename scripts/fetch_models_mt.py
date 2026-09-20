"""Download the two checkpoints the replication actually needs.

The paper uses a single encoder, ``BAAI/bge-base-zh-v1.5``, for every dataset,
so no English twin is fetched.  Downloads go through hf-mirror with parallel
byte-range segments because single-connection throughput here is ~40 KB/s.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_fetch import download, head  # noqa: E402

BASE = "https://hf-mirror.com"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

TARGETS = [
    ("BAAI/bge-base-zh-v1.5", ["config.json", "special_tokens_map.json", "tokenizer.json",
                               "tokenizer_config.json", "vocab.txt", "pytorch_model.bin"]),
    ("Qwen/Qwen2.5-1.5B-Instruct", ["config.json", "generation_config.json", "tokenizer.json",
                                    "tokenizer_config.json", "vocab.json", "merges.txt",
                                    "model.safetensors"]),
]

MIN_SIZE = {"pytorch_model.bin": 300 << 20, "model.safetensors": 2 << 30}
THREADS = {"pytorch_model.bin": 8, "model.safetensors": 12}


def main() -> int:
    for repo, files in TARGETS:
        sub = os.path.join(ROOT, repo.split("/")[-1])
        os.makedirs(sub, exist_ok=True)
        print(f"### {repo}", flush=True)
        for fname in files:
            dst = os.path.join(sub, fname)
            if os.path.exists(dst):
                sz = os.path.getsize(dst)
                if sz >= MIN_SIZE.get(fname, 1):
                    print(f"  [skip] {fname} ({sz/1e6:.1f} MB)", flush=True)
                    continue
                print(f"  [bad ] {fname} truncated -> redownload", flush=True)
                os.remove(dst)
            url = f"{BASE}/{repo}/resolve/main/{fname}"
            try:
                total, _ = head(url)
            except Exception as e:
                print(f"  [FAIL] head {fname}: {e}", flush=True)
                continue
            print(f"  [get ] {fname} ({total/1e6:.1f} MB)", flush=True)
            t0 = time.time()
            ok = download(url, dst, nthreads=THREADS.get(fname, 4), expect=total)
            print(f"  [{'ok  ' if ok else 'FAIL'}] {fname} in {time.time()-t0:.0f}s", flush=True)
    print("ALLDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
