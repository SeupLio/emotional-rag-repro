"""Build the unified character corpus used by every experiment.

Output: ``data/corpus.json``

    {
      "<character key>": {
        "name": ..., "lang": "zh" | "en",
        "memories": [str, ...],
        "labels": {"BFI": {...}, "16Personalities": {...}}
      }, ...
    }

Memory comes from ChatHaruhi (``silk-road/ChatHaruhi-RolePlaying``); ground
truth comes from InCharacter's ``characters_labels.json``.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
from config import DATA_DIR, RAW_DIR  # noqa: E402

INCHAR = RAW_DIR / "InCharacter-main" / "data"
CH_DIR = RAW_DIR / "chatharuhi"

ROLE_FILE = {
    "Hermione-en": "Hermione", "Sheldon-en": "Sheldon",
    "raidenShogun-zh": "raidenShogun", "zhongli-zh": "zhongli",
    "hutao-zh": "hutao", "haruhi-zh": "haruhi",
    "ayaka-zh": "ayaka", "wanderer-zh": "wanderer",
    "Luna-en": "Luna", "Snape-en": "Snape", "Malfoy-en": "Malfoy",
    "Ron-en": "Ron", "Dumbledore-en": "Dumbledore",
    "McGonagall-en": "McGonagall", "Harry-en": "Harry", "Raj-en": "Raj",
}

MIN_CHARS = 12
MAX_CHARS = 1200


def load_memories(stem: str) -> list[str]:
    """Read a ChatHaruhi jsonl into a list of dialogue fragments."""
    path = CH_DIR / f"{stem}.jsonl"
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            # skip the system prompt / config bookkeeping rows
            if obj.get("luotuo_openai") in ("system_prompt", "config"):
                continue
            if obj.get("bge_zh_s15") in ("system_prompt", "config"):
                continue
            text = (obj.get("text") or "").strip()
            if not text:
                continue
            text = re.sub(r"\s+\n", "\n", text)
            if len(text) < MIN_CHARS:
                continue
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]
            out.append(text)
    return out


def main() -> int:
    chars = json.load(open(INCHAR / "characters.json", encoding="utf-8"))
    labels = json.load(open(INCHAR / "characters_labels.json", encoding="utf-8"))
    labels = labels.get("annotation", labels)

    corpus = {}
    stats = []
    for key, stem in ROLE_FILE.items():
        if key not in chars or key not in labels:
            print(f"[skip] {key}: missing character entry or label")
            continue
        mem = load_memories(stem)
        if len(mem) < 20:
            print(f"[skip] {key}: only {len(mem)} memories")
            continue
        lang = "zh" if key.endswith("-zh") else "en"
        corpus[key] = {
            "name": stem,
            "lang": lang,
            "memories": mem,
            "labels": labels[key],
        }
        stats.append((key, lang, len(mem)))

    out = DATA_DIR / "corpus.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"\nwrote {out} with {len(corpus)} characters")
    tot = sum(s[2] for s in stats)
    print(f"total memories: {tot}, mean/char: {tot/max(len(stats),1):.0f}")
    for k, l, n in stats:
        print(f"  {k:26s} {l}  {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
