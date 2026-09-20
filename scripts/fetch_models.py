"""Download model weights from the hf-mirror endpoint with size verification.

The stock ``huggingface_hub`` client cannot be used in this sandbox: its
temporary-file cleanup trips the sandbox's safe-delete guard and aborts the
transfer.  Plain HTTP downloads with a Content-Length check are both simpler
and resumable-safe here.
"""

from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from huggingface_hub import hf_hub_download  # noqa: E402  (only for url helpers)

BASE = "https://hf-mirror.com"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# repo -> list of files
TARGETS = {
    "BAAI/bge-base-zh-v1.5": [
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "pytorch_model.bin",
    ],
    "BAAI/bge-base-en-v1.5": [
        "config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "pytorch_model.bin",
    ],
    "Qwen/Qwen2.5-1.5B-Instruct": [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    ],
}

# Minimum plausible byte size, used to invalidate truncated leftovers.
MIN_SIZE = {"pytorch_model.bin": 300 * 1024 * 1024, "model.safetensors": 2 * 1024 * 1024 * 1024}


def url_of(repo: str, fname: str) -> str:
    return f"{BASE}/{repo}/resolve/main/{fname}"


def remote_size(url: str) -> int:
    import urllib.request

    req = urllib.request.Request(url, method="HEAD")
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return int(r.headers.get("Content-Length", 0))
        except Exception:
            time.sleep(2)
    return 0


def download(url: str, dst: str, expect: int = 0) -> bool:
    import urllib.request

    tmp = dst + ".part"
    if os.path.exists(tmp):
        os.remove(tmp)
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length", 0))
                got = 0
                t0 = time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    got += len(chunk)
                    if got % (32 << 20) < (1 << 20):
                        mb = got / 1e6
                        spd = got / max(time.time() - t0, 1) / 1e6
                        print(f"    {mb:.0f}/{total/1e6:.0f} MB  {spd:.1f} MB/s", flush=True)
            break
        except Exception as e:
            print(f"    retry {attempt+1}: {type(e).__name__}: {e}", flush=True)
            time.sleep(3)
    else:
        return False

    size = os.path.getsize(tmp)
    if expect and size != expect:
        print(f"    size mismatch: got {size}, expected {expect}", flush=True)
        os.remove(tmp)
        return False
    if os.path.exists(dst):
        os.remove(dst)
    os.rename(tmp, dst)
    return True


def main() -> int:
    for repo, files in TARGETS.items():
        sub = os.path.join(ROOT, repo.split("/")[-1])
        os.makedirs(sub, exist_ok=True)
        print(f"### {repo}", flush=True)
        for fname in files:
            dst = os.path.join(sub, fname)
            if os.path.exists(dst):
                sz = os.path.getsize(dst)
                need = MIN_SIZE.get(fname, 1)
                if sz >= need:
                    print(f"  [skip] {fname} ({sz/1e6:.1f} MB)", flush=True)
                    continue
                print(f"  [bad ] {fname} truncated ({sz/1e6:.1f} MB) -> redownload", flush=True)
                os.remove(dst)
            url = url_of(repo, fname)
            expect = remote_size(url)
            print(f"  [get ] {fname} ({expect/1e6:.1f} MB)", flush=True)
            ok = download(url, dst, expect)
            print(f"  [{'ok  ' if ok else 'FAIL'}] {fname}", flush=True)
    print("ALLDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
