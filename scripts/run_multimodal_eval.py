"""Multimodal extension evaluation.

MM-1  acoustic channel validity on RAVDESS (real emotional speech, 8 classes)
MM-2  cross-channel agreement on the role-playing memory corpus
MM-3 is produced by ``run_experiment.py --channel fused --tts``.

Outputs ``results/multimodal_eval.json``.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from config import DATA_DIR, PLUTCHIK_EMOTIONS, RESULTS_DIR  # noqa: E402
from multimodal import (  # noqa: E402
    AcousticEmotionEstimator, RAVDESS_EMOTIONS, RAVDESS_TO_PLUTCHIK_DIM, load_ravdess,
)
from emo_classifier import TextEmotionEstimator  # noqa: E402

PARQUET = Path(ROOT) / "raw" / "ravdess_parquet" / "train-00000.parquet"
WAV_DIR = Path(ROOT) / "raw" / "ravdess_wav"


def log(m: str) -> None:
    print(m, flush=True)


def extract_ravdess(limit: int = 0) -> list[dict]:
    """Materialise wav files from the parquet shard."""
    import pyarrow.parquet as pq

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    t = pq.ParquetFile(str(PARQUET)).read()
    audio = t.column("audio").to_pylist()
    emo = t.column("emotion").to_pylist()
    actor = t.column("actor").to_pylist()
    vocal = t.column("vocal_channel").to_pylist()
    items = []
    for i, a in enumerate(audio):
        if str(vocal[i]).lower() != "speech":
            continue
        lab = str(emo[i]).strip().lower()
        if lab not in ("neutral", "calm", "happy", "sad", "angry", "fearful",
                       "disgust", "surprised"):
            continue
        out = WAV_DIR / f"{lab}_{actor[i]}_{i:04d}.wav"
        if not out.exists():
            try:
                out.write_bytes(a["bytes"])
            except Exception:
                continue
        items.append({
            "path": str(out),
            "emotion": lab,
            "actor": actor[i],
            "plutchik_dim": {
                "neutral": None, "calm": None, "happy": "joy", "sad": "sadness",
                "angry": "anger", "fearful": "fear", "disgust": "disgust",
                "surprised": "surprise",
            }[lab],
        })
        if limit and len(items) >= limit:
            break
    return items


def mm1(est: AcousticEmotionEstimator, items: list[dict], batch: int = 16) -> dict:
    """Acoustic-channel emotion recognition on RAVDESS."""
    labeled = [it for it in items if it["plutchik_dim"]]
    if not labeled:
        return {"error": "no labelled items"}
    paths = [it["path"] for it in labeled]
    vecs = np.zeros((len(paths), 8))
    for i in range(0, len(paths), batch):
        vecs[i: i + batch] = est.encode_files(paths[i: i + batch])
    gold = np.array([PLUTCHIK_EMOTIONS.index(it["plutchik_dim"]) for it in labeled])
    pred = vecs.argmax(axis=1)
    acc = float((pred == gold).mean())
    # macro-F1 over the six Plutchik-mappable classes actually present
    f1s, per_class = {}, {}
    for j, dim in enumerate(PLUTCHIK_EMOTIONS):
        tp = int(((pred == j) & (gold == j)).sum())
        fp = int(((pred == j) & (gold != j)).sum())
        fn = int(((pred != j) & (gold == j)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s[dim] = 2 * p * r / (p + r) if p + r else 0.0
        per_class[dim] = {"tp": tp, "fp": fp, "fn": fn, "support": tp + fn}
    # 4-way SER-native accuracy (map Plutchik argmax back to SER space)
    ser_gold = {"joy": "hap", "sadness": "sad", "anger": "ang",
                "fear": "fear", "disgust": "disgust", "surprise": "surprise"}
    # confusion over the 6 mapped classes
    conf = defaultdict(Counter)
    for it, p in zip(labeled, pred):
        conf[it["emotion"]][PLUTCHIK_EMOTIONS[p]] += 1
    return {
        "n": len(labeled),
        "argmax_acc_8d": acc,
        "macro_f1": float(np.mean(list(f1s.values()))) if f1s else 0.0,
        "per_class_f1": f1s,
        "per_class": per_class,
        "confusion": {k: dict(v) for k, v in conf.items()},
        "emotion_counts": dict(Counter(it["emotion"] for it in labeled)),
    }


def make_text_estimator(source: str, lang: str):
    """Text emotion channel used in MM-2.

    ``anchor`` (default) reuses the paper's own encoder ``F`` and needs no
    extra weights -- the Go-Emotions checkpoint is 712 MB and is optional.
    """
    if source == "cls":
        try:
            return TextEmotionEstimator(device="cpu")
        except Exception as e:  # pragma: no cover
            print(f"   Go-Emotions unavailable ({e}); falling back to anchors")
    from emo_anchor import AnchorEmotionEstimator
    from embeddings import get_encoder

    cache: dict = {}

    def enc_for(l):
        if l not in cache:
            cache[l] = get_encoder(l)
        return cache[l]

    base = AnchorEmotionEstimator(enc_for)

    class _Wrap:
        def encode(self, texts):
            return base.encode(texts, lang=lang)

    return _Wrap()


def mm2(ac_est: AcousticEmotionEstimator, chars: list[str], n_mem: int = 64,
        text_source: str = "anchor") -> dict:
    """Cross-channel agreement on TTS-synthesised role memories."""
    from multimodal import synthesize

    corpus = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    out = {}
    for key in chars:
        if key not in corpus:
            continue
        lang = corpus[key]["lang"]
        mem = corpus[key]["memories"][:n_mem]
        wav_dir = Path(ROOT) / "raw" / "tts" / key
        wav_dir.mkdir(parents=True, exist_ok=True)
        jobs = [{"text": m[:400], "out": str(wav_dir / f"mm{i:04d}.wav")}
                for i, m in enumerate(mem)]
        paths = synthesize(jobs, cache_dir=wav_dir, lang=lang)
        have = sum(1 for p in paths if p)
        if have == 0:
            out[key] = {"error": "TTS produced no audio"}
            continue
        text_est = make_text_estimator(text_source, lang)
        v_txt = text_est.encode(mem)
        v_aud = ac_est.encode_files(paths)
        # cosine agreement + per-dimension Pearson correlation
        na = v_txt / (np.linalg.norm(v_txt, axis=1, keepdims=True) + 1e-12)
        nb = v_aud / (np.linalg.norm(v_aud, axis=1, keepdims=True) + 1e-12)
        cos = float((na * nb).sum(axis=1).mean())
        cors = []
        for j in range(8):
            a, b = v_txt[:, j], v_aud[:, j]
            if a.std() < 1e-9 or b.std() < 1e-9:
                continue
            cors.append(float(np.corrcoef(a, b)[0, 1]))
        out[key] = {
            "lang": lang, "n": len(mem), "n_audio": have,
            "cosine": cos,
            "mean_pearson_per_dim": float(np.mean(cors)) if cors else None,
            "audio_vec_std": [round(float(x), 3) for x in v_aud.std(axis=0)],
            "text_vec_std": [round(float(x), 3) for x in v_txt.std(axis=0)],
        }
        log(f"  {key}: cosine={cos:.3f} pearson={out[key]['mean_pearson_per_dim']}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=480)
    ap.add_argument("--mm2-chars", default="Hermione-en,Sheldon-en")
    ap.add_argument("--mm2-mem", type=int, default=64)
    ap.add_argument("--skip-mm1", action="store_true")
    ap.add_argument("--skip-mm2", action="store_true")
    ap.add_argument("--text-source", default="anchor", choices=["anchor", "cls"])
    ap.add_argument("--ser", default=None,
                    help="SER checkpoint dir (default: models/ser_ravdess, the "
                         "RAVDESS-trained 8-class model). Pass "
                         "models/hubert-base-superb-er to reproduce the v1 "
                         "IEMOCAP checkpoint failure reported in section 5.2.")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="device for the SER forward pass")
    args = ap.parse_args()

    ser_kw = {"device": args.device}
    if args.ser:
        ser_kw["model_dir"] = args.ser

    res: dict = {"meta": {"limit": args.limit,
                          "ser": args.ser or "default:models/ser_ravdess"}}
    t0 = time.time()

    if not args.skip_mm1:
        items = extract_ravdess(args.limit)
        log(f"[MM-1] RAVDESS speech clips: {len(items)} "
            f"({Counter(i['emotion'] for i in items)})")
        ac = AcousticEmotionEstimator(**ser_kw)
        log(f"       SER checkpoint: {ac.model_dir}")
        log(f"       SER labels: {ac.labels}")
        res["mm1_acoustic_ravdess"] = mm1(ac, items)
        log(f"       argmax_acc_8d={res['mm1_acoustic_ravdess'].get('argmax_acc_8d')} "
            f"macro_f1={res['mm1_acoustic_ravdess'].get('macro_f1')}")

    if not args.skip_mm2:
        log("[MM-2] cross-channel agreement")
        ac2 = AcousticEmotionEstimator(**ser_kw)
        res["mm2_cross_channel"] = mm2(
            ac2, [c for c in args.mm2_chars.split(",") if c], args.mm2_mem,
            text_source=args.text_source)
        res["meta"]["text_source"] = args.text_source

    res["meta"]["seconds"] = round(time.time() - t0, 1)
    out = RESULTS_DIR / "multimodal_eval.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
