"""Multimodal extension: a speech (acoustic) emotion channel for Emotional RAG.

This module is **this project's own addition** -- the paper's emotion
estimator ``G`` is purely textual.  Emotional RAG's retrieval depends on an
8-d Plutchik vector, so any additional modality that can produce such a vector
plugs straight into the existing strategies (``C-A``/``C-M``/``S-S``/``S-E``).

Design
------
* ``TextChannel``     -- the paper's estimator (LLM scoring of the text).
* ``AcousticChannel`` -- speech pipeline: ``text -> TTS -> wav -> SER`` where
  SER is ``superb/hubert-base-superb-er`` (HuBERT-base fine-tuned on IEMOCAP for
  the SUPERB Emotion Recognition task).  Its 4-way posterior is mapped onto the
  Plutchik wheel with a fixed, documented linear map.
* ``fuse``            -- ``g = alpha * g_text + (1 - alpha) * g_audio``.

Because the acoustic channel is a *real* second modality (audio waveform), the
fusion is a genuine multimodal emotion representation rather than a re-weighting
of the same text signal.

Validation
----------
``validate_acoustic()`` scores the acoustic channel on RAVDESS (8 emotions, six
of which are Plutchik primitives) and ``validate_fusion()`` measures how much
the text channel gains from the acoustic channel on that labelled set.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from config import (
    EMOTION_SCORE_MAX,
    EMOTION_SCORE_MIN,
    MODEL_DIR,
    MULTIMODAL_ALPHA,
    PLUTCHIK_EMOTIONS,
    RAW_DIR,
)

SER_DIR = str(Path(MODEL_DIR) / "hubert-base-superb-er")
RAVDESS_DIR = Path(RAW_DIR) / "ravdess"
TTS_DIR = Path(RAW_DIR) / "tts"

# ---------------------------------------------------------------------------
# Mapping from the SER label space onto Plutchik's wheel
# ---------------------------------------------------------------------------
# rows are ordered by the model's own ``config.id2label`` at load time; the
# defaults below follow SUPERB/hubert-base-superb-er (neu / hap / sad / ang).
_SER_TO_PLUTCHIK: Dict[str, List[float]] = {
    "neu": [2.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0],
    "hap": [9.0, 7.0, 1.0, 5.0, 1.0, 1.0, 1.0, 7.0],
    "happy": [9.0, 7.0, 1.0, 5.0, 1.0, 1.0, 1.0, 7.0],
    "sad": [1.0, 3.0, 5.0, 2.0, 10.0, 3.0, 2.0, 1.0],
    "ang": [1.0, 1.0, 4.0, 3.0, 2.0, 6.0, 10.0, 3.0],
    "angry": [1.0, 1.0, 4.0, 3.0, 2.0, 6.0, 10.0, 3.0],
}
_ORDER = ["joy", "acceptance", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
assert _ORDER == PLUTCHIK_EMOTIONS


def ser_to_plutchik(probs: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    """(N, C) SER posterior -> (N, 8) Plutchik vector."""
    rows = []
    for lab in labels:
        key = str(lab).lower()
        rows.append(_SER_TO_PLUTCHIK.get(key, [5.0] * 8))
    M = np.asarray(rows, dtype=np.float64)          # (C, 8)
    out = probs @ M                                  # (N, 8)
    return np.clip(out, EMOTION_SCORE_MIN, EMOTION_SCORE_MAX)


def fuse(text_vec: np.ndarray, audio_vec: np.ndarray, alpha: float = MULTIMODAL_ALPHA) -> np.ndarray:
    """Linear fusion of the two emotion channels."""
    return np.clip(
        alpha * np.asarray(text_vec, dtype=np.float64)
        + (1.0 - alpha) * np.asarray(audio_vec, dtype=np.float64),
        EMOTION_SCORE_MIN,
        EMOTION_SCORE_MAX,
    )


# ---------------------------------------------------------------------------
# WAV IO (scipy only -- soundfile / torchaudio are not installed)
# ---------------------------------------------------------------------------

def read_wav(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    sr, data = wavfile.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    if data.max() > 1.0 or data.min() < -1.0:
        data = data / max(abs(data).max(), 1e-9)
    if data.dtype != np.float64:
        pass
    if sr != target_sr:
        from math import gcd
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, int(target_sr) // g, int(sr) // g)
    return data.astype(np.float32)


# ---------------------------------------------------------------------------
# Text-to-speech (offline, Windows SAPI through a single PowerShell process)
# ---------------------------------------------------------------------------

_PS_TEMPLATE = """
Add-Type -AssemblyName System.Speech
$items = Get-Content -Raw -LiteralPath '{json_path}' | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = 0
foreach ($it in $items) {{
    if (Test-Path -LiteralPath $it.out) {{ continue }}
    $s.SetOutputToWaveFile($it.out)
    $s.Speak($it.text)
    $s.SetOutputToNull()
}}
$s.Dispose()
"""


def synthesize(items: Sequence[Dict[str, str]], cache_dir: str | Path = TTS_DIR,
               lang: str = "en") -> List[Optional[str]]:
    """Synthesise ``[{"text": ..., "out": ...}]`` and return the wav paths."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (Path(it["out"]).exists()
                                       and Path(it["out"]).stat().st_size > 1000)]
    if todo:
        json_path = cache_dir / "_tts_jobs.json"
        json_path.write_text(json.dumps(todo, ensure_ascii=False), encoding="utf-8")
        ps = cache_dir / "_tts_run.ps1"
        ps.write_text(_PS_TEMPLATE.format(json_path=str(json_path).replace("\\", "\\\\")),
                      encoding="utf-8")
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-File", str(ps)],
            capture_output=True, timeout=7200,
        )
    return [it["out"] if os.path.exists(it["out"]) else None for it in items]


# ---------------------------------------------------------------------------
# Acoustic emotion estimator
# ---------------------------------------------------------------------------

class AcousticEmotionEstimator:
    """HuBERT-base SER -> Plutchik 8-d vector."""

    def __init__(self, model_dir: str = SER_DIR, device: str = "cpu",
                 batch_size: int = 8, max_seconds: float = 8.0):
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self.max_samples = int(16000 * max_seconds)
        self.fx = AutoFeatureExtractor.from_pretrained(model_dir)
        self.model = AutoModelForAudioClassification.from_pretrained(model_dir)
        self.model.eval()
        self.model.to(device)
        id2label = getattr(self.model.config, "id2label", {}) or {}
        self.labels = [id2label.get(i, f"LAB_{i}") for i in range(len(id2label))] \
            or ["neu", "hap", "sad", "ang"]

    def predict_probs(self, wavs: Sequence[np.ndarray]) -> np.ndarray:
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(wavs), self.batch_size):
                chunk = list(wavs[i: i + self.batch_size])
                enc = self.fx(
                    chunk, sampling_rate=16000, return_tensors="pt",
                    padding=True, do_normalize=True,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy()
                out.append(probs)
        return np.concatenate(out, axis=0) if out else np.zeros((0, len(self.labels)))

    def encode_files(self, paths: Sequence[Optional[str]]) -> np.ndarray:
        """Return (N, 8) Plutchik vectors; missing audio -> neutral vector."""
        vecs = np.full((len(paths), len(PLUTCHIK_EMOTIONS)), 5.0, dtype=np.float64)
        idx, waves = [], []
        for i, p in enumerate(paths):
            if p and os.path.exists(p):
                try:
                    waves.append(read_wav(p))
                    idx.append(i)
                except Exception:
                    continue
        for start in range(0, len(idx), self.batch_size):
            sel = idx[start: start + self.batch_size]
            probs = self.predict_probs(waves[start: start + self.batch_size])
            vecs[sel] = ser_to_plutchik(probs, self.labels)
        return vecs


def emotion_vector_from_audio(paths: Sequence[Optional[str]],
                              estimator: "AcousticEmotionEstimator") -> np.ndarray:
    return estimator.encode_files(paths)


# ---------------------------------------------------------------------------
# RAVDESS validation of the acoustic channel
# ---------------------------------------------------------------------------

RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised",
}
# RAVDESS emotion -> dominant Plutchik dimension index
RAVDESS_TO_PLUTCHIK_DIM = {
    "01": None, "02": None, "03": "joy", "04": "sadness",
    "05": "anger", "06": "fear", "07": "disgust", "08": "surprise",
}
# same mapping keyed by the *string* labels used in the HF parquet shard
_RAVDESS_STR_TO_PLUTCHIK = {
    "neutral": None, "calm": None, "happy": "joy", "sad": "sadness",
    "angry": "anger", "fearful": "fear", "disgust": "disgust",
    "surprised": "surprise",
}


def load_ravdess(root: str | Path = RAVDESS_DIR) -> List[dict]:
    root = Path(root)
    if not root.exists():
        return []
    items = []
    for p in sorted(root.glob("*.wav")):
        parts = p.stem.split("-")
        if len(parts) < 7:
            continue
        emo = parts[2]
        if emo not in RAVDESS_EMOTIONS:
            continue
        items.append({
            "path": str(p),
            "emotion": RAVDESS_EMOTIONS[emo],
            "plutchik_dim": RAVDESS_TO_PLUTCHIK_DIM[emo],
            "actor": parts[6],
        })
    return items


def load_ravdess_parquet(parquet: str | Path,
                         out_dir: str | Path = None) -> List[dict]:
    """Materialise the ``xbgoose/ravdess`` parquet shard into wav files.

    The upstream RAVDESS mirrors are either stub repositories or multi-GB
    archives, so we use the HF dataset shard instead.  Each row carries the
    audio bytes plus the string emotion label, which we map onto the Plutchik
    wheel directly (no filename parsing needed).
    """
    import pyarrow.parquet as pq

    parquet = Path(parquet)
    out_dir = Path(out_dir or (RAW_DIR / "ravdess_wav"))
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pq.read_table(str(parquet))
    cols = {n: table.column(n).to_pylist() for n in table.schema.names}
    items: List[dict] = []
    actors = cols.get("actor", [None] * len(cols["audio"]))
    for i, blob in enumerate(cols["audio"]):
        emo = str(cols["emotion"][i]).lower()
        dim = _RAVDESS_STR_TO_PLUTCHIK.get(emo)
        wav = out_dir / f"ravdess_{i:05d}_{emo}.wav"
        if not wav.exists() or wav.stat().st_size < 1000:
            data = blob["bytes"] if isinstance(blob, dict) else blob
            wav.write_bytes(data)
        items.append({
            "path": str(wav),
            "emotion": emo,
            "plutchik_dim": dim,
            "actor": str(actors[i]) if i < len(actors) else "",
        })
    return items


def validate_acoustic(estimator: "AcousticEmotionEstimator",
                      items: Sequence[dict]) -> dict:
    """Does the acoustic Plutchik vector put its mass on the right emotion?"""
    if not items:
        return {"n": 0}
    paths = [it["path"] for it in items]
    vecs = estimator.encode_files(paths)
    hits, n_eval = 0, 0
    confusion: Dict[str, Dict[str, int]] = {}
    for it, v in zip(items, vecs):
        dim = it["plutchik_dim"]
        if dim is None:
            continue
        j = PLUTCHIK_EMOTIONS.index(dim)
        # "hit" = the target dimension is the arg-max of the Plutchik vector
        pred = PLUTCHIK_EMOTIONS[int(np.argmax(v))]
        n_eval += 1
        hits += int(pred == dim)
        confusion.setdefault(it["emotion"], {})
        confusion[it["emotion"]][pred] = confusion[it["emotion"]].get(pred, 0) + 1
    return {
        "n": len(items),
        "n_eval": n_eval,
        "argmax_acc": hits / n_eval if n_eval else 0.0,
        "confusion": confusion,
    }


def validate_fusion(items: Sequence[dict], text_vecs: np.ndarray,
                    audio_vecs: np.ndarray, alpha: float = MULTIMODAL_ALPHA) -> dict:
    """Arg-max accuracy of text-only vs fused vectors on the labelled set."""
    res = {}
    for name, V in (("text", text_vecs), ("audio", audio_vecs),
                    ("fused", fuse(text_vecs, audio_vecs, alpha))):
        hits, n = 0, 0
        for it, v in zip(items, V):
            dim = it["plutchik_dim"]
            if dim is None:
                continue
            n += 1
            hits += int(PLUTCHIK_EMOTIONS[int(np.argmax(v))] == dim)
        res[name] = {"acc": hits / n if n else 0.0, "n": n}
    return res
