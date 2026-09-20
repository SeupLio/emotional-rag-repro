"""Text emotion channel: Go Emotions (multilingual) -> Plutchik 8-d vector.

Why this exists
---------------
The paper obtains its emotion vector with GPT-3.5 through an "emotional
prompt" that asks for an integer 1..10 on each Plutchik dimension.  The local
1.5B backbone cannot follow that contract: on held-out probes it either
repeats the prompt or emits an immediate EOS, so **every** vector collapses to
the neutral fallback (see ``report/reproduction_report.md``, deviation D3).

Rather than reporting an experiment with a constant emotion vector (which would
make all five retrieval strategies mathematically identical), the text channel
is produced by a pretrained emotion classifier whose 28-label taxonomy maps
onto Plutchik's wheel.  The **contract** is unchanged -- an 8-d vector scaled
1..10 -- only the estimator behind it is substituted, which is exactly the
substitution the reproduction report is written to quantify.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import numpy as np

from config import (
    EMOTION_SCORE_MAX, EMOTION_SCORE_MIN, MODEL_DIR, PLUTCHIK_EMOTIONS,
)

EMO_MODEL_DIR = str(MODEL_DIR / "multilingual_go_emotions")

# Go Emotions label -> Plutchik 8-d intensity row (1..10).
# Unspecified dimensions fall back to ``BASE``.
BASE = 2.0
_GO_ROWS: Dict[str, Dict[str, float]] = {
    "admiration":   {"acceptance": 8, "joy": 6},
    "amusement":    {"joy": 8},
    "anger":        {"anger": 9, "disgust": 4},
    "annoyance":    {"anger": 7, "disgust": 4},
    "approval":     {"acceptance": 8},
    "caring":       {"acceptance": 7, "joy": 5},
    "confusion":    {"surprise": 6, "fear": 3},
    "curiosity":    {"anticipation": 8, "surprise": 3},
    "desire":       {"anticipation": 8, "joy": 4},
    "disappointment": {"sadness": 7, "surprise": 3},
    "disapproval":  {"disgust": 6, "anger": 4},
    "disgust":      {"disgust": 9},
    "embarrassment": {"fear": 5, "sadness": 4, "surprise": 4},
    "excitement":   {"joy": 8, "anticipation": 7},
    "fear":         {"fear": 9},
    "gratitude":    {"acceptance": 8, "joy": 6},
    "grief":        {"sadness": 9},
    "joy":          {"joy": 9},
    "love":         {"joy": 8, "acceptance": 8},
    "nervousness":  {"fear": 8, "anticipation": 3},
    "optimism":     {"anticipation": 8, "joy": 6},
    "pride":        {"joy": 7, "acceptance": 5},
    "realization":  {"surprise": 7},
    "relief":       {"joy": 6, "acceptance": 4},
    "remorse":      {"sadness": 7, "fear": 3},
    "sadness":      {"sadness": 9},
    "surprise":     {"surprise": 9},
    "neutral":      {d: 3.0 for d in PLUTCHIK_EMOTIONS},
}


def build_matrix(labels: Sequence[str]) -> np.ndarray:
    """(C, 8) mapping matrix aligned with the classifier's label order."""
    rows = []
    for lab in labels:
        spec = _GO_ROWS.get(str(lab).lower(), {})
        row = [float(spec.get(d, BASE)) for d in PLUTCHIK_EMOTIONS]
        rows.append(row)
    return np.asarray(rows, dtype=np.float64)


class TextEmotionEstimator:
    """Multilingual Go-Emotions classifier -> Plutchik 8-d emotion vector."""

    def __init__(self, model_dir: str = EMO_MODEL_DIR, device: str = "cpu",
                 batch_size: int = 32, max_len: int = 256):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.model_dir = model_dir
        self.batch_size = batch_size
        self.max_len = max_len
        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval().to(self.device)
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.labels = [id2label.get(i, f"LAB_{i}") for i in range(len(id2label))]
        self.M = build_matrix(self.labels)

    def predict_probs(self, texts: Sequence[str]) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [t[: self.max_len * 4] for t in texts[i: i + self.batch_size]]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_len, return_tensors="pt")
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits.float()
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
                out.append(probs)
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(self.labels)))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        probs = self.predict_probs(list(texts))
        vec = probs @ self.M if len(probs) else np.full((len(texts), len(PLUTCHIK_EMOTIONS)), 5.0)
        return np.clip(vec, EMOTION_SCORE_MIN, EMOTION_SCORE_MAX)

    def encode_with_probs(self, texts: Sequence[str]):
        probs = self.predict_probs(list(texts))
        vec = np.clip(probs @ self.M, EMOTION_SCORE_MIN, EMOTION_SCORE_MAX)
        return vec, probs


def dominant_emotion(vec: np.ndarray) -> str:
    return PLUTCHIK_EMOTIONS[int(np.argmax(vec))]


def demo() -> None:
    est = TextEmotionEstimator()
    samples = [
        "I am absolutely furious, I will never forgive him.",
        "Congratulations! You did a great job!",
        "这件事让我非常愤怒，我绝对不会原谅他。",
        "听到这个消息，我心里一阵失落和悲伤。",
    ]
    vecs, probs = est.encode_with_probs(samples)
    for s, v, p in zip(samples, vecs, probs):
        top = est.labels[int(np.argmax(p))]
        print(f"{s[:38]:40s} top={top:14s} {np.round(v,1)} -> {dominant_emotion(v)}")


if __name__ == "__main__":
    demo()
