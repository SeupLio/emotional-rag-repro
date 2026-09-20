"""Plutchik emotion-vector estimation (function ``G`` in the paper).

The paper obtains the emotion vector with GPT-3.5 through a hand-written
emotional prompt that asks for an integer score in ``[1, 10]`` on each of
Plutchik's eight basic emotions.  We keep the *prompt contract* identical and
only swap the scoring model for the locally hosted backbone (see
``config.BACKBONE_MODEL``); every difference this causes is documented in the
reproduction report.

An additional multimodal estimator is provided in ``multimodal_emotion.py``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from config import (
    DATA_DIR,
    EMOTION_SCORE_MAX,
    EMOTION_SCORE_MIN,
    PLUTCHIK_EMOTIONS,
)

# ---------------------------------------------------------------------------
# Prompts -- same structure as the paper's "emotional prompt": task
# description, definition of the emotion dimensions, scoring standard, and a
# strict output format.
# ---------------------------------------------------------------------------

_EMO_DEF = {
    "zh": {
        "joy": "快乐：愉悦、满足、开心",
        "acceptance": "接纳：信任、认可、赞同",
        "fear": "恐惧：害怕、担忧、焦虑",
        "surprise": "惊讶：意外、震惊、诧异",
        "sadness": "悲伤：难过、失落、沮丧",
        "disgust": "厌恶：反感、嫌弃、排斥",
        "anger": "愤怒：生气、恼火、愤慨",
        "anticipation": "期待：盼望、憧憬、预见",
    },
    "en": {
        "joy": "joy: pleasure, satisfaction, happiness",
        "acceptance": "acceptance: trust, approval, agreement",
        "fear": "fear: being afraid, worried, anxious",
        "surprise": "surprise: unexpectedness, shock, astonishment",
        "sadness": "sadness: sorrow, loss, depression",
        "disgust": "disgust: dislike, aversion, repulsion",
        "anger": "anger: being angry, annoyed, indignant",
        "anticipation": "anticipation: expectancy, hope, looking forward",
    },
}

_TEMPLATE_ZH = """你是一个情感标注器。请阅读下面的文本，并依据 Plutchik 情感环的 8 个维度打分。

评分标准：每个维度给出 {lo} 到 {hi} 之间的整数，{lo} 表示该情感极弱，{hi} 表示该情感极强。
维度定义：
{defs}

只输出一行 JSON，键为下列英文维度名，值为整数，不要输出任何解释。
{{"joy": ?, "acceptance": ?, "fear": ?, "surprise": ?, "sadness": ?, "disgust": ?, "anger": ?, "anticipation": ?}}

文本：
{text}

JSON："""

_TEMPLATE_EN = """You are an emotion annotator. Read the text below and score it along the eight
dimensions of Plutchik's wheel of emotions.

Scoring standard: give an integer between {lo} and {hi} for each dimension,
where {lo} means the emotion is extremely weak and {hi} means extremely strong.
Dimension definitions:
{defs}

Output one line of JSON only, with the English dimension names as keys and
integers as values. Do not output any explanation.
{{"joy": ?, "acceptance": ?, "fear": ?, "surprise": ?, "sadness": ?, "disgust": ?, "anger": ?, "anticipation": ?}}

Text:
{text}

JSON:"""


def build_emotion_prompt(text: str, lang: str = "en") -> str:
    defs = "\n".join(f"- {_EMO_DEF[lang][k]}" for k in PLUTCHIK_EMOTIONS)
    tpl = _TEMPLATE_ZH if lang == "zh" else _TEMPLATE_EN
    return tpl.format(
        lo=EMOTION_SCORE_MIN,
        hi=EMOTION_SCORE_MAX,
        defs=defs,
        text=text[:1500],
    )


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{[^{}]*\}")
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_emotion_vector(raw: str) -> np.ndarray:
    """Parse a model completion into an 8-d Plutchik vector.

    Falls back progressively: strict JSON -> any JSON object -> first eight
    numbers in the text -> neutral vector (all 5s).
    """
    vec = np.full(len(PLUTCHIK_EMOTIONS), 5.0, dtype=np.float64)
    if not raw:
        return vec

    m = _JSON_RE.search(raw)
    if m:
        try:
            obj = json.loads(m.group(0))
            vals = []
            for k in PLUTCHIK_EMOTIONS:
                v = obj.get(k, None)
                if v is None:
                    vals.append(5.0)
                else:
                    vals.append(float(_NUM_RE.search(str(v)).group(0))
                                if _NUM_RE.search(str(v)) else 5.0)
            return np.clip(np.asarray(vals, dtype=np.float64),
                           EMOTION_SCORE_MIN, EMOTION_SCORE_MAX)
        except Exception:
            pass

    nums = _NUM_RE.findall(raw)
    if len(nums) >= len(PLUTCHIK_EMOTIONS):
        vals = [float(x) for x in nums[: len(PLUTCHIK_EMOTIONS)]]
        return np.clip(np.asarray(vals, dtype=np.float64),
                       EMOTION_SCORE_MIN, EMOTION_SCORE_MAX)
    return vec


def neutral_vector() -> np.ndarray:
    return np.full(len(PLUTCHIK_EMOTIONS), 5.0, dtype=np.float64)


# ---------------------------------------------------------------------------
# Estimator with an on-disk cache (emotion scoring is the most expensive part
# of the pipeline and is reused across all retrieval strategies).
# ---------------------------------------------------------------------------


class EmotionEstimator:
    """Wraps a text-generation backend and caches emotion vectors on disk."""

    def __init__(self, backend, cache_path: str | Path | None = None,
                 lang: str = "en", batch_size: int = 16):
        self.backend = backend
        self.lang = lang
        self.batch_size = batch_size
        self.cache_path = Path(cache_path) if cache_path else DATA_DIR / f"emotion_cache_{lang}.json"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, list[float]] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}

    # -- cache helpers -----------------------------------------------------
    @staticmethod
    def _key(text: str) -> str:
        return str(abs(hash(text)))

    def _flush(self) -> None:
        try:
            self.cache_path.write_text(
                json.dumps(self.cache, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:
            pass

    # -- main API ----------------------------------------------------------
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(N, 8)`` array of Plutchik emotion vectors."""
        out = np.zeros((len(texts), len(PLUTCHIK_EMOTIONS)), dtype=np.float64)
        todo: List[int] = []
        prompts: List[str] = []
        for i, t in enumerate(texts):
            k = self._key(t)
            if k in self.cache:
                out[i] = np.asarray(self.cache[k], dtype=np.float64)
            else:
                todo.append(i)
                prompts.append(build_emotion_prompt(t, self.lang))
        if todo:
            for start in range(0, len(todo), self.batch_size):
                chunk_idx = todo[start: start + self.batch_size]
                chunk_pr = prompts[start: start + self.batch_size]
                completions = self.backend.generate(
                    chunk_pr, max_new_tokens=64, temperature=0.0, stop=None
                )
                for i, c in zip(chunk_idx, completions):
                    v = parse_emotion_vector(c)
                    out[i] = v
                    self.cache[self._key(texts[i])] = [round(float(x), 4) for x in v]
            self._flush()
        return out

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def __len__(self) -> int:
        return len(self.cache)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between ``a`` (N, d) and ``b`` (M, d)."""
    na = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    nb = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return na @ nb.T


def l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)


def iter_batches(items: Iterable, n: int):
    items = list(items)
    for i in range(0, len(items), n):
        yield items[i: i + n]
