"""Re-fetch the HuBERT SER checkpoint with integrity verification.

``parallel_fetch`` pre-allocates the file, so a segment that fails after all
retries leaves a hole of zeros and the size check still passes.  We therefore
verify by actually loading the checkpoint, and retry until it parses.
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
URL = "https://hf-mirror.com/superb/hubert-base-superb-er/resolve/main/pytorch_model.bin"
D = os.path.join(ROOT, "models", "hubert-base-superb-er")
STALE = os.path.join(ROOT, "raw", "_stale")
EXPECT = 378360273


def main() -> int:
    os.makedirs(STALE, exist_ok=True)
    for attempt in range(1, 5):
        dst = os.path.join(D, "pytorch_model.bin")
        if os.path.exists(dst):
            os.replace(dst, os.path.join(STALE, f"hubert_bad_{attempt}.bin"))
        t0 = time.time()
        download(URL, dst, nthreads=6, expect=EXPECT)
        print(f"[try {attempt}] downloaded in {time.time()-t0:.0f}s", flush=True)
        try:
            sd = torch.load(dst, map_location="cpu", weights_only=True)
            sd = {k: v.contiguous() for k, v in sd.items()}
            out = os.path.join(D, "model.safetensors")
            save_file(sd, out, metadata={"format": "pt"})
            print(f"[ok  ] {len(sd)} tensors -> {out}", flush=True)
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"[bad ] {type(exc).__name__}: {exc}", flush=True)
    print("[fail] could not obtain a valid checkpoint")
    return 1


if __name__ == "__main__":
    sys.exit(main())
