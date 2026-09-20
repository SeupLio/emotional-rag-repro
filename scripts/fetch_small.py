"""Fetch small model files (config / tokenizer) with a plain urllib fallback.

hf-mirror does not answer HEAD for every file, which breaks the ranged
downloader in ``parallel_fetch.py``.  Small files do not need range requests
anyway, so we just stream them with urllib.
"""
import os
import sys
import time
import urllib.request

MIRROR = "https://hf-mirror.com"

TARGETS = {
    "Qwen/Qwen2.5-1.5B-Instruct": [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ],
}

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def fetch(repo: str, fname: str) -> bool:
    dst_dir = os.path.join(ROOT, repo.split("/")[-1])
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, fname)
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return True
    url = f"{MIRROR}/{repo}/resolve/main/{fname}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {fname}: {exc}")
        return False
    if not data:
        print(f"[FAIL] {fname}: empty body")
        return False
    tmp = dst + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, dst)
    print(f"[ok  ] {fname} ({len(data)/1024:.1f} KB) in {time.time()-t0:.0f}s")
    return True


def main() -> int:
    ok = 0
    for repo, files in TARGETS.items():
        print(f"--- {repo} ---")
        for fname in files:
            ok += fetch(repo, fname)
    print(f"done: {ok} files ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
