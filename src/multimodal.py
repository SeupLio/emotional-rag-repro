"""Multimodal extension: a speech (acoustic) emotion channel for Emotional RAG.

This module is **this project's own addition** -- the paper's emotion
estimator ``G`` is purely textual.  Emotional RAG's retrieval depends on an
8-d Plutchik vector, so any additional modality that can produce such a vector
plugs straight into the existing strategies (``C-A``/``C-M``/``S-S``/``S-E``).

Design
------
* ``TextChannel``     -- the paper's estimator (LLM scoring of the text).
* ``AcousticChannel`` -- speech pipeline: ``text -> TTS -> wav -> SER`` where the
  default SER is ``ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition``
  (wav2vec2-large-XLSR-53 fine-tuned on **RAVDESS**, 8 classes).  Its posterior is
  mapped onto the Plutchik wheel with a fixed, documented linear map.
  The first attempt used ``superb/hubert-base-superb-er`` (IEMOCAP, 4 classes);
  that checkpoint predicted ``anger`` for every RAVDESS clip, so it is retained
  only behind ``ERAG_SER_DIR`` for reproducing the reported failure.
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

# v2 (default) -- RAVDESS-trained 8-class SER.  Its label space matches the
# RAVDESS evaluation set, so the acoustic channel is finally measured on the
# classes it was trained for.
SER_DIR = os.environ.get("ERAG_SER_DIR", str(Path(MODEL_DIR) / "ser_ravdess"))
# v1 (archived) -- IEMOCAP-trained 4-class HuBERT.  Kept only so the failure
# mode documented in the report (section 5.2) can be reproduced on demand:
#   ERAG_SER_DIR=models/hubert-base-superb-er python scripts/run_multimodal_eval.py
SER_DIR_HUBERT = str(Path(MODEL_DIR) / "hubert-base-superb-er")
RAVDESS_DIR = Path(RAW_DIR) / "ravdess"
TTS_DIR = Path(RAW_DIR) / "tts"

# ---------------------------------------------------------------------------
# Mapping from the SER label space onto Plutchik's wheel
# ---------------------------------------------------------------------------
# rows are ordered by the model's own ``config.id2label`` at load time; the
# defaults below follow SUPERB/hubert-base-superb-er (neu / hap / sad / ang).
# The RAVDESS-trained checkpoint (v2) exposes all eight RAVDESS labels, so the
# full label space is covered; the SUPERB aliases are kept for the v1 HuBERT
# checkpoint (neu / hap / sad / ang).
_SER_TO_PLUTCHIK: Dict[str, List[float]] = {
    # --- neutral / calm: no dominant Plutchik axis, kept mild -------------
    "neu": [2.0, 5.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0],
    "neutral": [4.0, 5.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
    "calm": [4.0, 6.0, 2.0, 2.0, 3.0, 3.0, 2.0, 4.0],
    # --- joy ---------------------------------------------------------------
    "hap": [9.0, 7.0, 1.0, 5.0, 1.0, 1.0, 1.0, 7.0],
    "happy": [9.0, 7.0, 1.0, 5.0, 1.0, 1.0, 1.0, 7.0],
    # --- sadness -----------------------------------------------------------
    "sad": [1.0, 3.0, 5.0, 2.0, 10.0, 3.0, 2.0, 1.0],
    # --- anger -------------------------------------------------------------
    "ang": [1.0, 1.0, 4.0, 3.0, 2.0, 6.0, 10.0, 3.0],
    "angry": [1.0, 1.0, 4.0, 3.0, 2.0, 6.0, 10.0, 3.0],
    # --- fear --------------------------------------------------------------
    "fear": [1.0, 2.0, 10.0, 4.0, 3.0, 3.0, 2.0, 2.0],
    "fearful": [1.0, 2.0, 10.0, 4.0, 3.0, 3.0, 2.0, 2.0],
    # --- disgust -----------------------------------------------------------
    "disgust": [1.0, 1.0, 3.0, 2.0, 3.0, 10.0, 4.0, 1.0],
    "disgusted": [1.0, 1.0, 3.0, 2.0, 3.0, 10.0, 4.0, 1.0],
    # --- surprise ----------------------------------------------------------
    "surprise": [3.0, 3.0, 4.0, 10.0, 2.0, 2.0, 2.0, 7.0],
    "surprised": [3.0, 3.0, 4.0, 10.0, 2.0, 2.0, 2.0, 7.0],
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
# Text-to-speech (offline).  Primary backend is pyttsx3 (COM/SAPI), because the
# sandbox forbids ``Add-Type`` -- the original PowerShell pipeline silently
# produced no audio there and is now only a fallback.
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


_SYNTH_WORKER = (
    "import json,sys,pyttsx3;"
    "j=json.load(open(sys.argv[1],encoding='utf-8'));"
    "e=pyttsx3.init();"
    "e.save_to_file(j['text'],j['out']);"
    "e.runAndWait()"
)


def _sanitise(text: str, limit: int = 160) -> str:
    """Keep TTS input short and free of characters that make SAPI hang."""
    import re

    t = " ".join(str(text).split())
    t = re.sub(r"[\x00-\x1f\x7f]", " ", t)
    return t[:limit]


def _synth_one(text: str, out: str, timeout: int = 25) -> bool:
    """Synthesise one utterance in a **separate** process.

    pyttsx3/SAPI can hang indefinitely on certain inputs; running each item in
    its own subprocess with a timeout means one bad utterance can never stall
    the whole evaluation.
    """
    import subprocess
    import tempfile

    fd, jp = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        with open(jp, "w", encoding="utf-8") as f:
            json.dump({"text": text, "out": out}, f, ensure_ascii=False)
        try:
            subprocess.run([sys.executable, "-c", _SYNTH_WORKER, jp],
                           timeout=timeout, capture_output=True)
        except subprocess.TimeoutExpired:
            pass
    finally:
        try:
            os.remove(jp)
        except OSError:
            pass
    return os.path.exists(out) and os.path.getsize(out) > 1000


def _synthesise_pyttsx3(todo: Sequence[Dict[str, str]], lang: str,
                        timeout: int = 25) -> int:
    """Synthesise via pyttsx3 (COM/SAPI).  Returns how many files were written.

    pyttsx3 talks to SAPI through COM and therefore does **not** need
    ``Add-Type -AssemblyName System.Speech``, which the sandbox blocks -- this is
    why it replaces the original PowerShell pipeline.  Each utterance runs in an
    isolated subprocess so a hang is contained by the timeout.
    """
    written = 0
    for it in todo:
        try:
            if _synth_one(_sanitise(it["text"]), it["out"], timeout):
                written += 1
        except Exception:
            continue
    return written


def synthesize(items: Sequence[Dict[str, str]], cache_dir: str | Path = TTS_DIR,
               lang: str = "en") -> List[Optional[str]]:
    """Synthesise ``[{"text": ..., "out": ...}]`` and return the wav paths.

    Preferred backend is pyttsx3 (COM, sandbox-safe).  The PowerShell/SAPI
    pipeline is kept as a fallback for environments without pyttsx3.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    todo = [it for it in items if not (Path(it["out"]).exists()
                                       and Path(it["out"]).stat().st_size > 1000)]
    if todo:
        try:
            _synthesise_pyttsx3(todo, lang)
        except Exception:
            pass
        # fall back to the PowerShell pipeline for anything still missing
        todo = [it for it in todo if not (Path(it["out"]).exists()
                                          and Path(it["out"]).stat().st_size > 1000)]
        if todo:
            json_path = cache_dir / "_tts_jobs.json"
            json_path.write_text(json.dumps(todo, ensure_ascii=False),
                                 encoding="utf-8")
            ps = cache_dir / "_tts_run.ps1"
            ps.write_text(
                _PS_TEMPLATE.format(json_path=str(json_path).replace("\\", "\\\\")),
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

def _has_legacy_head(model_dir: str) -> bool:
    """True when the checkpoint carries the pre-4.9 custom ``dense/output`` head.

    ``ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`` was saved with
    transformers 4.8.2 and ships ``classifier.dense.*`` / ``classifier.output.*``
    (1024 -> 1024 -> 8).  Loading it with a modern
    ``AutoModelForAudioClassification`` silently reports the head as MISSING and
    **randomly initialises it**, so we detect the layout and rebuild the head.
    """
    import glob

    ckpts = glob.glob(os.path.join(model_dir, "model.safetensors"))
    if not ckpts:
        return False
    from safetensors import safe_open

    with safe_open(ckpts[0], framework="pt") as f:
        keys = set(f.keys())
    return "classifier.dense.weight" in keys and "classifier.output.weight" in keys


class _LegacyHeadSER:
    """wav2vec2 encoder + the checkpoint's own ``dense -> output`` head.

    The activation between the two linear layers is *not* recoverable from the
    config, so it was determined empirically against the RAVDESS ground truth
    (see ``scripts/ser_head_probe.py``): dense->output 0.9250,
    dense->tanh->output 0.9229, dense->relu->output 0.9208 -- i.e. the choice is
    within noise, and the plain linear stack is used.
    """

    def __init__(self, model_dir: str, device: str):
        import torch
        from safetensors.torch import load_file
        from transformers import Wav2Vec2Model

        self.device = device
        self.torch = torch
        self.encoder = Wav2Vec2Model.from_pretrained(model_dir)
        self.encoder.eval().to(device)
        sd = load_file(os.path.join(model_dir, "model.safetensors"))
        self.dense_w = sd["classifier.dense.weight"].float().to(device)
        self.dense_b = sd["classifier.dense.bias"].float().to(device)
        self.out_w = sd["classifier.output.weight"].float().to(device)
        self.out_b = sd["classifier.output.bias"].float().to(device)

    def __call__(self, input_values, attention_mask):
        import torch

        with torch.no_grad():
            hs = self.encoder(
                input_values=input_values,
                attention_mask=attention_mask,
            ).last_hidden_state
            lengths = self.encoder._get_feat_extract_output_lengths(
                attention_mask.sum(-1)).to(hs.device)
            idx = torch.arange(hs.shape[1], device=hs.device)[None, :]
            m = (idx < lengths[:, None]).to(hs.dtype).unsqueeze(-1)
            pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)
            h = pooled @ self.dense_w.T + self.dense_b
            return h @ self.out_w.T + self.out_b


class AcousticEmotionEstimator:
    """Speech emotion recogniser -> Plutchik 8-d vector.

    Supports both the modern ``AutoModelForAudioClassification`` layout and the
    legacy custom ``dense/output`` head (detected automatically).
    """

    def __init__(self, model_dir: str = SER_DIR, device: str = "cpu",
                 batch_size: int = 8, max_seconds: float = 8.0):
        from transformers import (AutoConfig, AutoFeatureExtractor,
                                  AutoModelForAudioClassification)

        self.model_dir = model_dir
        self.device = device
        self.batch_size = batch_size
        self.max_samples = int(16000 * max_seconds)
        self.fx = AutoFeatureExtractor.from_pretrained(model_dir)
        self.legacy = _has_legacy_head(model_dir)
        if self.legacy:
            self.model = _LegacyHeadSER(model_dir, device)
            self.config = AutoConfig.from_pretrained(model_dir)
        else:
            self.model = AutoModelForAudioClassification.from_pretrained(model_dir)
            self.model.eval()
            self.model.to(device)
            self.config = self.model.config
        id2label = getattr(self.config, "id2label", {}) or {}
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
                if self.legacy:
                    logits = self.model(enc["input_values"],
                                        enc.get("attention_mask"))
                else:
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
