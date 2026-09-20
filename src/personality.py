"""Personality questionnaire protocol (BFI + 16Personalities / MBTI).

The paper evaluates a role-playing agent by making it answer an open
psychological questionnaire and comparing the induced personality profile with
the ground-truth profile collected from personality-database.com.  We use the
exact questionnaire files and ground-truth labels shipped with **InCharacter**
(the dataset the paper builds on), so the metric definitions line up:

* ``Acc(Dim)``  -- fraction of personality dimensions whose H/L/X band matches
* ``Acc(Full)`` -- fraction of characters where *all* dimensions match
* ``MSE``/``MAE`` -- error of the normalised per-dimension score (0..1)
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import RAW_DIR

INCHAR = RAW_DIR / "InCharacter-main" / "data"
Q_DIR = INCHAR / "questionnaires"
LABEL_PATH = INCHAR / "characters_labels.json"
CHAR_PATH = INCHAR / "characters.json"

BFI_DIMS = ["Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Openness"]
MBTI_DIMS = ["E/I", "S/N", "T/F", "P/J"]

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Questionnaire loading
# ---------------------------------------------------------------------------

@dataclass
class Questionnaire:
    name: str
    dims: List[str]
    qids: List[str]
    text: Dict[str, str]          # qid -> prompt text
    dim_of: Dict[str, str]        # qid -> dimension
    reverse: set                  # qids scored in reverse
    cat_questions: Dict[str, List[str]]
    crowd: Dict[str, tuple]       # dim -> (mean, std)
    lo: int
    hi: int
    lang: str = "zh"

    def questions_for(self, dim: str) -> List[str]:
        return self.cat_questions.get(dim, [])


def _load(name: str, lang: str = "zh", dim_order: Optional[List[str]] = None) -> Questionnaire:
    with open(Q_DIR / f"{name}.json", encoding="utf-8") as f:
        raw = json.load(f)
    questions = raw["questions"]
    key = "rewritten_zh" if lang == "zh" else "rewritten_en"
    text = {qid: q.get(key) or q.get("origin_zh") or q.get("origin_en")
            for qid, q in questions.items()}
    dim_of = {qid: q["dimension"] for qid, q in questions.items()}
    cats = raw["categories"]
    dims = dim_order or [c["cat_name"] for c in cats]
    cat_questions = {c["cat_name"]: [str(i) for i in c["cat_questions"]] for c in cats}
    crowd = {}
    for c in cats:
        cr = (c.get("crowd") or [{}])[0]
        if "mean" in cr:
            crowd[c["cat_name"]] = (float(cr["mean"]), float(cr.get("std", 1.0)))
    lo, hi = raw.get("range", [1, 5])
    return Questionnaire(
        name=name, dims=dims, qids=list(questions.keys()), text=text, dim_of=dim_of,
        reverse={str(i) for i in raw.get("reverse", [])},
        cat_questions=cat_questions, crowd=crowd, lo=int(lo), hi=int(hi), lang=lang,
    )


def load_bfi(lang: str = "zh") -> Questionnaire:
    return _load("BFI", lang, BFI_DIMS)


def load_mbti(lang: str = "zh") -> Questionnaire:
    return _load("16Personalities", lang, MBTI_DIMS)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def load_labels() -> Dict[str, Dict[str, Dict[str, dict]]]:
    with open(LABEL_PATH, encoding="utf-8") as f:
        blob = json.load(f)
    return blob.get("annotation", blob)


def load_pdb_labels() -> Dict[str, Dict[str, Dict[str, dict]]]:
    """personality-database labels -- only used to decide which dims are 'X'."""
    with open(LABEL_PATH, encoding="utf-8") as f:
        blob = json.load(f)
    return blob.get("pdb", blob)


def load_characters() -> Dict[str, dict]:
    with open(CHAR_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Answer parsing & scoring
# ---------------------------------------------------------------------------

def parse_likert(raw: str, lo: int, hi: int) -> float:
    """Extract the Likert value from a completion; fall back to the midpoint."""
    if not raw:
        return (lo + hi) / 2.0
    m = _NUM.search(raw)
    if not m:
        return (lo + hi) / 2.0
    v = float(m.group(0))
    if v < lo or v > hi:
        v = min(max(v, float(lo)), float(hi))
    return v


def score_dimension(q: Questionnaire, answers: Dict[str, float], dim: str) -> float:
    """Aggregate the item scores of one dimension.

    Follows InCharacter's official aggregation (``personality_tests.py``):

    * BFI            -- mean of the item scores on the native 1..5 scale.
    * 16Personalities-- each item is mapped ``(v - 1) / 6 * 100`` first and the
      mean is taken, so the dimension score is a percentage in ``[0, 100]``.

    Reverse-worded items are flipped using the questionnaire's own ``reverse``
    list (see deviation D7 in the reproduction report: InCharacter's reference
    implementation does not do this, the psychometric convention does).
    """
    qs = q.cat_questions.get(dim, [])
    vals = []
    for qid in qs:
        if qid not in answers:
            continue
        v = answers[qid]
        if qid in q.reverse:
            v = (q.lo + q.hi) - v
        if q.name == "16Personalities":
            v = (v - q.lo) / (q.hi - q.lo) * 100.0
        vals.append(v)
    if not vals:
        return (q.lo + q.hi) / 2.0
    return float(np.mean(vals))


def normalise(q: Questionnaire, dim: str, value: float) -> float:
    """Map a raw dimension score onto [0, 1] using InCharacter's own spans."""
    if q.name == "BFI":
        return (value - 1.0) / 4.0
    return value / 100.0


def band_of(q: Questionnaire, dim: str, value: float) -> str:
    """InCharacter's rule: ``H if score > range_middle else L``.

    ``range_middle`` is 3 for BFI (range 1..5) and 50 for 16Personalities
    (mapped to 0..100).  Predictions therefore never carry an ``X`` band --
    only the *gold* PDB labels do, and those dimensions are skipped.
    """
    middle = 50.0 if q.name == "16Personalities" else 3.0
    return "H" if value > middle else "L"


def profile_from_answers(q: Questionnaire, answers: Dict[str, float]):
    scores = {d: score_dimension(q, answers, d) for d in q.dims}
    norms = {d: float(np.clip(normalise(q, d, v), 0.0, 1.0)) for d, v in scores.items()}
    bands = {d: band_of(q, d, v) for d, v in scores.items()}
    return scores, norms, bands


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    acc_dim: float = 0.0
    acc_full: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    n_char: int = 0
    per_char: List[dict] = field(default_factory=list)


def evaluate(preds: Dict[str, dict], golds: Dict[str, dict], dims: Sequence[str]) -> MetricResult:
    """preds/golds: character -> {dim: {'norm': float, 'band': str}}"""
    acc_dims, full_hits, sq, ab, n = [], 0, [], [], 0
    per_char = []
    for ch, gold in golds.items():
        if ch not in preds:
            continue
        p = preds[ch]
        hit = True
        for d in dims:
            if d not in gold or d not in p:
                continue
            if gold[d]["band"] == "X":
                # defensive: normally filtered out by gold_profiles(pdb=...)
                continue
            acc_dims.append(1.0 if p[d]["band"] == gold[d]["band"] else 0.0)
            e = p[d]["norm"] - gold[d]["norm"]
            sq.append(e * e)
            ab.append(abs(e))
            if p[d]["band"] != gold[d]["band"]:
                hit = False
        n += 1
        if hit:
            full_hits += 1
        per_char.append({
            "character": ch,
            "bands_pred": {d: p.get(d, {}).get("band") for d in dims},
            "bands_gold": {d: gold.get(d, {}).get("band") for d in dims},
        })
    return MetricResult(
        acc_dim=float(np.mean(acc_dims)) if acc_dims else 0.0,
        acc_full=full_hits / n if n else 0.0,
        mse=float(np.mean(sq)) if sq else 0.0,
        mae=float(np.mean(ab)) if ab else 0.0,
        n_char=n,
        per_char=per_char,
    )


def gold_profiles(labels: Dict[str, dict], scale: str, dims: Sequence[str],
                  pdb: Optional[Dict[str, dict]] = None):
    """Convert InCharacter ground-truth labels into the metric's input form.

    ``labels`` is the human-annotation block (``character_labels['annotation']``)
    -- the gold score *and* the gold H/L/X band.  ``pdb`` is the
    personality-database block and is used **only** to decide which dimensions
    are dropped: InCharacter's ``calculate_measured_alignment`` does

        if labels_pdb[rpa][dim]['type'] == 'X': continue

    so a dimension whose PDB band is ambiguous is excluded from every metric
    (Acc(Dim), Acc(Full), MSE, MAE) instead of being counted as a miss.
    """
    out: Dict[str, dict] = {}
    for ch, blob in labels.items():
        if scale not in blob:
            continue
        dim_map = {}
        for d in dims:
            if d not in blob[scale]:
                continue
            if pdb is not None:
                pd = (pdb.get(ch, {}).get(scale, {}) or {}).get(d, {}) or {}
                if pd.get("type") == "X":
                    continue
            raw = blob[scale][d].get("score")
            if raw is None:
                continue
            norm = raw / 100.0 if scale == "16Personalities" else (raw - 1.0) / 4.0
            dim_map[d] = {"norm": float(np.clip(norm, 0.0, 1.0)),
                          "band": blob[scale][d].get("type", "X")}
        if dim_map:
            out[ch] = dim_map
    return out


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_qa_prompt(character: str, memories: Sequence[str], question: str,
                    lang: str = "zh", lo: int = 1, hi: int = 5) -> str:
    mem = "\n".join(f"- {m}" for m in memories)
    if lang == "zh":
        return (
            f"你现在扮演角色「{character}」。请始终保持这个角色的性格、语气与价值观。\n\n"
            f"下面是与你有关的若干记忆片段：\n{mem}\n\n"
            f"请以「{character}」的身份回答下面的问题。\n"
            f"只输出一个 {lo} 到 {hi} 之间的整数（{lo}=完全不同意，{hi}=完全同意），不要输出任何解释。\n\n"
            f"问题：{question}\n"
            f"回答："
        )
    return (
        f"You are now role-playing as \"{character}\". Stay in character: keep this "
        f"character's personality, tone and values at all times.\n\n"
        f"Here are some memory fragments related to you:\n{mem}\n\n"
        f"Answer the question below as \"{character}\".\n"
        f"Output a single integer between {lo} and {hi} ({lo}=strongly disagree, "
        f"{hi}=strongly agree). Do not output any explanation.\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )
