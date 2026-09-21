"""Zero-shot Plutchik emotion channel: bge embedding vs. emotion anchor sentences.

The paper's ``G`` is a black box that turns text into an 8-d Plutchik vector
scaled 1..10.  Two substitutes are implemented in this project:

* :mod:`emo_classifier` -- a supervised Go-Emotions classifier (needs a 712 MB
  download that this machine's uplink could not finish reliably);
* **this module** -- a *zero-shot* estimator that scores a text against eight
  anchor sentences, one per Plutchik dimension, using **the paper's own
  semantic encoder** ``F`` (bge-base).  No extra weights, works for both the
  English and the Chinese half of InCharacter.

``G(text) = 1 + 9 * minmax_d( cos(F(text), F(anchor_d)) )``

so each dimension lands in ``[1, 10]`` exactly as the paper requires, and the
profile across dimensions reflects which emotion the text is closest to.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from config import EMOTION_SCORE_MAX, EMOTION_SCORE_MIN, PLUTCHIK_EMOTIONS

ANCHORS: Dict[str, Dict[str, str]] = {
    "en": {
        "joy": "I am so happy, this is wonderful, I feel great joy and delight.",
        "acceptance": "I trust you completely, I agree and accept this wholeheartedly.",
        "fear": "I am terrified and anxious, I am afraid something terrible will happen.",
        "surprise": "What a shock! I never expected this, I am completely astonished.",
        "sadness": "I feel deep sorrow and grief, my heart is broken and I want to cry.",
        "disgust": "That is revolting and repulsive, I feel sickened and full of aversion.",
        "anger": "I am furious and outraged, this makes me extremely angry and indignant.",
        "anticipation": "I look forward to it eagerly, I can't wait to see what happens next.",
    },
    "zh": {
        "joy": "我太开心了，真是令人愉快的好消息，我感到非常高兴和满足。",
        "acceptance": "我完全信任你，我认同并接受这件事，心里很踏实。",
        "fear": "我非常害怕和焦虑，担心会有可怕的事情发生，心里发慌。",
        "surprise": "太震惊了！我完全没有预料到，真是出乎意料让我惊讶。",
        "sadness": "我心里充满悲伤和难过，心都碎了，很想哭。",
        "disgust": "这太令人反感恶心了，我对此充满厌恶和排斥。",
        "anger": "我非常愤怒和恼火，这件事让我气愤到无法忍受。",
        "anticipation": "我满怀期待地盼望着，迫不及待想看到接下来会发生什么。",
    },
}


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = x.min(axis=1, keepdims=True)
    hi = x.max(axis=1, keepdims=True)
    return (x - lo) / (hi - lo + 1e-9)


class AnchorEmotionEstimator:
    """Plutchik vector from bge similarity to eight emotion anchor sentences."""

    def __init__(self, encoder_for, batch_size: int = 64):
        """``encoder_for(lang)`` must return an object with ``.encode(texts)``."""
        self.encoder_for = encoder_for
        self.batch_size = batch_size
        self._cache: Dict[str, np.ndarray] = {}

    def _anchors(self, lang: str) -> np.ndarray:
        if lang not in self._cache:
            enc = self.encoder_for(lang)
            texts = [ANCHORS.get(lang, ANCHORS["en"])[d] for d in PLUTCHIK_EMOTIONS]
            A = enc.encode(texts)
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
            self._cache[lang] = A
        return self._cache[lang]

    def encode(self, texts: Sequence[str], lang: str = "en") -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, len(PLUTCHIK_EMOTIONS)))
        enc = self.encoder_for(lang)
        E = enc.encode(texts, batch_size=self.batch_size) if _accepts(enc) else enc.encode(texts)
        E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
        cos = E @ self._anchors(lang).T                       # (N, 8)
        v = EMOTION_SCORE_MIN + (EMOTION_SCORE_MAX - EMOTION_SCORE_MIN) * _minmax(cos)
        return np.clip(v, EMOTION_SCORE_MIN, EMOTION_SCORE_MAX)


def _accepts(enc) -> bool:
    try:
        import inspect
        return "batch_size" in inspect.signature(enc.encode).parameters
    except Exception:
        return False


def demo() -> None:
    from embeddings import get_encoder

    cache: Dict[str, object] = {}

    def enc_for(lang: str):
        if lang not in cache:
            cache[lang] = get_encoder(lang)
        return cache[lang]

    est = AnchorEmotionEstimator(enc_for)
    samples = [
        ("en", "I am absolutely furious, I will never forgive him."),
        ("en", "Congratulations! What a wonderful surprise, I'm so happy for you!"),
        ("en", "I feel so lonely and hopeless tonight."),
        ("zh", "这件事让我非常愤怒，我绝对不会原谅他。"),
        ("zh", "听到这个消息，我心里一阵失落和悲伤。"),
    ]
    for lang, s in samples:
        v = est.encode([s], lang=lang)[0]
        top = PLUTCHIK_EMOTIONS[int(np.argmax(v))]
        print(f"{lang} {s[:40]:44s} {np.round(v, 1)} -> {top}")


if __name__ == "__main__":
    demo()
