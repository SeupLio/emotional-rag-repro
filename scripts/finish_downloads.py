"""Finish pending downloads: promote completed ``.part`` files and refill gaps.

* Qwen2.5-1.5B-Instruct model.safetensors  -- already complete
* ChatHaruhi Harry.jsonl                   -- already complete
* BAAI/bge-base-en-v1.5 pytorch_model.bin  -- truncated, re-fetch (438 MB)
* ChatHaruhi zhongli / Raj                 -- HEAD returns no length, use urllib
"""
from __future__ import annotations

import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_fetch import download  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
RAW = os.path.join(ROOT, "raw", "chatharuhi")
MIRROR = "https://hf-mirror.com"


STALE = os.path.join(ROOT, "raw", "_stale")


def promote(part: str) -> bool:
    """Move a finished ``.part`` file onto its final name.

    Never deletes anything -- leftovers are *moved* into ``raw/_stale`` so the
    sandbox's bulk-delete guard is not triggered.
    """
    dst = part[: -len(".part")]
    size = os.path.getsize(part)
    if size == 0:
        return False
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        os.makedirs(STALE, exist_ok=True)
        os.replace(part, os.path.join(STALE, os.path.basename(part)))
        return False
    os.replace(part, dst)
    print(f"[rename] {os.path.basename(dst)} ({size/1e6:.1f} MB)")
    return True


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
    print(f"[ok    ] {os.path.basename(dst)} ({len(data)/1e6:.2f} MB) in {time.time()-t0:.0f}s")
    return True


def main() -> int:
    for d in (MODELS, RAW):
        for name in sorted(os.listdir(d)):
            p = os.path.join(d, name)
            if os.path.isfile(p) and name.endswith(".part"):
                promote(p)

    # ---- bge-base-en-v1.5 (English characters of InCharacter) -------------
    en_dir = os.path.join(MODELS, "bge-base-en-v1.5")
    os.makedirs(en_dir, exist_ok=True)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "vocab.txt"):
        try:
            fetch_plain(f"{MIRROR}/BAAI/bge-base-en-v1.5/resolve/main/{f}",
                        os.path.join(en_dir, f))
        except Exception as exc:  # noqa: BLE001
            print(f"[skip  ] {f}: {exc}")
    bin_path = os.path.join(en_dir, "pytorch_model.bin")
    if not (os.path.exists(bin_path) and os.path.getsize(bin_path) == 437997357):
        if os.path.exists(bin_path):
            os.makedirs(STALE, exist_ok=True)
            os.replace(bin_path, os.path.join(STALE, "bge_en_truncated.bin"))
        url = f"{MIRROR}/BAAI/bge-base-en-v1.5/resolve/main/pytorch_model.bin"
        try:
            download(url, bin_path, nthreads=10, expect=437997357)
        except TypeError:
            download(url, bin_path, nthreads=10)
        got = os.path.getsize(bin_path) if os.path.exists(bin_path) else 0
        print(f"[bge-en] {got/1e6:.1f} MB / 437997357")

    # ---- missing ChatHaruhi roles ----------------------------------------
    for stem in ("zhongli", "Raj"):
        dst = os.path.join(RAW, f"{stem}.jsonl")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        url = (f"{MIRROR}/datasets/silk-road/ChatHaruhi-RolePlaying"
               f"/resolve/main/{stem}.jsonl")
        try:
            fetch_plain(url, dst)
        except Exception as exc:  # noqa: BLE001
            print(f"[fail  ] {stem}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
