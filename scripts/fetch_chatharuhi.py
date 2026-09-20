"""Download the ChatHaruhi role corpora that overlap with InCharacter.

Emotional RAG builds its InCharacter memory from ChatHaruhi / RoleLLM /
character.ai dialogue.  ``silk-road/ChatHaruhi-RolePlaying`` on HF hosts one
jsonl per character, and 16 of its characters coincide with the 32 InCharacter
characters that carry BFI + 16Personalities ground-truth labels.  Those 16 are
the memory backbone of this replication.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parallel_fetch import download, head  # noqa: E402

BASE = "https://hf-mirror.com/datasets/silk-road/ChatHaruhi-RolePlaying/resolve/main"
DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "raw", "chatharuhi")

# InCharacter character key -> ChatHaruhi file stem
ROLES = {
    "Hermione-en": "Hermione",
    "Sheldon-en": "Sheldon",
    "raidenShogun-zh": "raidenShogun",
    "zhongli-zh": "zhongli",
    "hutao-zh": "hutao",
    "haruhi-zh": "haruhi",
    "ayaka-zh": "ayaka",
    "wanderer-zh": "wanderer",
    "Luna-en": "Luna",
    "Snape-en": "Snape",
    "Malfoy-en": "Malfoy",
    "Ron-en": "Ron",
    "Dumbledore-en": "Dumbledore",
    "McGonagall-en": "McGonagall",
    "Harry-en": "Harry",
    "Raj-en": "Raj",
}


def main() -> int:
    os.makedirs(DST, exist_ok=True)
    for key, stem in ROLES.items():
        dst = os.path.join(DST, f"{stem}.jsonl")
        if os.path.exists(dst) and os.path.getsize(dst) > 1024:
            print(f"[skip] {stem}", flush=True)
            continue
        url = f"{BASE}/{stem}.jsonl"
        try:
            total, _ = head(url)
        except Exception as e:
            print(f"[warn] head {stem}: {e}", flush=True)
            total = 0
        if total == 0:
            # hf-mirror does not always answer HEAD on dataset files; pull
            # directly with urllib and skip the size check.
            try:
                import urllib.request

                req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
                with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
                    f.write(r.read())
                ok = os.path.getsize(dst) > 1024
            except Exception as e:
                print(f"[FAIL] {stem}: {type(e).__name__}: {e}", flush=True)
                ok = False
        else:
            ok = download(url, dst, nthreads=4, expect=total)
        print(f"[{'ok  ' if ok else 'FAIL'}] {stem} ({os.path.getsize(dst)/1e6 if os.path.exists(dst) else 0:.2f} MB)", flush=True)
    print("ALLDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
