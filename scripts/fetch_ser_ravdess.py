"""Download a RAVDESS-trained speech-emotion-recognition checkpoint.

Why this exists
---------------
The first version of the multimodal acoustic channel used
``superb/hubert-base-superb-er`` (fine-tuned on **IEMOCAP, 4 classes**) and
scored it against **RAVDESS (8 classes)**.  The two label spaces do not line
up and, in practice, the checkpoint predicted ``anger`` for *every* clip
(acc = 0.174, macro-F1 = 0.053) -- see the reproduction report, section 5.2.

``ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`` is fine-tuned on
RAVDESS itself and exposes the 8 classes we actually evaluate, so it is the
drop-in replacement for the acoustic channel.

Notes
-----
* The HuggingFace main site is unreachable from this machine; we pull from the
  ``hf-mirror.com`` mirror.  (The ``audeering/...`` checkpoint that the report
  originally proposed simply returns 404 on the mirror, which is why the first
  attempt produced 0-byte files.)
* This repo ships ``model.safetensors``, so we avoid ``pytorch_model.bin``
  entirely -- transformers >= 5 refuses to load ``.bin`` (CVE-2025-32434).
"""
from __future__ import annotations

import os
import sys
import time

BASE = "https://hf-mirror.com"
REPO = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DEST = os.path.join(ROOT, "ser_ravdess")

FILES = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",
]

MIN_SIZE = {"model.safetensors": 1 * 1024 * 1024 * 1024}  # ~1.2 GB


def _get(url: str, dst: str) -> bool:
    import urllib.request

    tmp = dst + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as r, \
                    open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                got = 0
                t0 = time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got % (64 << 20) < (1 << 20):
                        print(f"    {got/1e6:.0f}/{total/1e6:.0f} MB "
                              f"{got/max(time.time()-t0,1)/1e6:.1f} MB/s",
                              flush=True)
            break
        except Exception as e:
            print(f"    retry {attempt+1}: {type(e).__name__}: {e}", flush=True)
            time.sleep(4)
    else:
        return False

    size = os.path.getsize(tmp)
    if size < MIN_SIZE.get(os.path.basename(dst), 1):
        print(f"    too small ({size} B), discarding", flush=True)
        os.remove(tmp)
        return False
    if os.path.exists(dst):
        os.remove(dst)
    os.rename(tmp, dst)
    return True


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    print(f"### {REPO} -> {DEST}", flush=True)
    for fname in FILES:
        dst = os.path.join(DEST, fname)
        if os.path.exists(dst) and \
                os.path.getsize(dst) >= MIN_SIZE.get(fname, 1):
            print(f"  [skip] {fname} ({os.path.getsize(dst)/1e6:.1f} MB)",
                  flush=True)
            continue
        url = f"{BASE}/{REPO}/resolve/main/{fname}"
        print(f"  [get ] {fname}", flush=True)
        ok = _get(url, dst)
        print(f"  [{'ok  ' if ok else 'FAIL'}] {fname}", flush=True)
        if not ok:
            return 1
    print("ALLDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
