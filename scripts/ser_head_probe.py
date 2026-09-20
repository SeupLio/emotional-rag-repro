"""Determine the correct classification head of the RAVDESS SER checkpoint.

Problem
-------
``ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`` was saved with
transformers **4.8.2** and carries a *custom* two-layer head:

    classifier.dense.weight   (1024, 1024)
    classifier.dense.bias     (1024,)
    classifier.output.weight  (8, 1024)
    classifier.output.bias    (8,)

Modern transformers' ``Wav2Vec2ForSequenceClassification`` instead expects
``projector`` (classifier_proj_size=256 x 1024) and ``classifier`` (8 x 256).
Loading the checkpoint as-is therefore reports the head as MISSING and
**randomly initialises it** -- the model would emit garbage while looking like
it loaded fine.

There is no reliable way to know from the config whether the original head
applied an activation between the two linear layers, so instead of guessing we
*measure*: run the encoder once to get mean-pooled features, then score every
candidate head against the RAVDESS ground truth.

Usage
-----
    python scripts/ser_head_probe.py [--limit 480] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from multimodal import read_wav  # noqa: E402
from run_multimodal_eval import extract_ravdess  # noqa: E402

SER_DIR = os.path.join(ROOT, "models", "ser_ravdess")
OUT = os.path.join(ROOT, "results", "ser_head_probe.json")


def load_head_tensors(ckpt: str):
    from safetensors.torch import load_file

    sd = load_file(ckpt)
    return {
        "dense_w": sd["classifier.dense.weight"],
        "dense_b": sd["classifier.dense.bias"],
        "out_w": sd["classifier.output.weight"],
        "out_b": sd["classifier.output.bias"],
    }


CANDIDATES = {
    # mean-pool -> dense -> TANH -> output  (Wav2Vec2ClassificationHead style)
    "dense_tanh_out": lambda x, h: _lin(
        np.tanh(_lin(x, h["dense_w"], h["dense_b"])), h["out_w"], h["out_b"]),
    # mean-pool -> dense -> output  (no activation)
    "dense_out": lambda x, h: _lin(
        _lin(x, h["dense_w"], h["dense_b"]), h["out_w"], h["out_b"]),
    # mean-pool -> dense -> ReLU -> output
    "dense_relu_out": lambda x, h: _lin(
        np.maximum(_lin(x, h["dense_w"], h["dense_b"]), 0), h["out_w"], h["out_b"]),
}


def _lin(x, w, b):
    return x @ w.numpy().T + b.numpy()


def pooled_features(paths, device, batch=16):
    """Run the wav2vec2 encoder once; return mean-pooled (N, 1024) features."""
    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2Model

    fx = AutoFeatureExtractor.from_pretrained(SER_DIR)
    model = Wav2Vec2Model.from_pretrained(SER_DIR)
    model.eval().to(device)

    feats = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            chunk = [read_wav(p) for p in paths[i: i + batch]]
            enc = fx(chunk, sampling_rate=16000, return_tensors="pt",
                     padding=True, do_normalize=True)
            enc = {k: v.to(device) for k, v in enc.items()}
            hs = model(**enc).last_hidden_state            # (B, T, 1024)
            mask = enc.get("attention_mask")
            if mask is not None:
                # attention_mask is in *raw sample* space while hs is in
                # down-sampled frame space -> convert the lengths first.
                lengths = model._get_feat_extract_output_lengths(
                    mask.sum(-1))
                lengths = lengths.to(hs.device)
                idx = torch.arange(hs.shape[1], device=hs.device)[None, :]
                m = (idx < lengths[:, None]).to(hs.dtype).unsqueeze(-1)
                pooled = (hs * m).sum(1) / m.sum(1).clamp(min=1e-6)
            else:
                pooled = hs.mean(1)
            feats.append(pooled.float().cpu().numpy())
            print(f"  encoded {min(i+batch, len(paths))}/{len(paths)}", flush=True)
    return np.concatenate(feats, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=480)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    items = extract_ravdess(args.limit)
    # full 8-class evaluation (not just the 6 Plutchik-mappable ones)
    paths = [it["path"] for it in items]
    gold = [it["emotion"] for it in items]
    print(f"[probe] {len(items)} clips, {len(set(gold))} emotions", flush=True)

    head = load_head_tensors(os.path.join(SER_DIR, "model.safetensors"))
    X = pooled_features(paths, args.device, args.batch)
    print(f"[probe] pooled features {X.shape}", flush=True)

    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(SER_DIR)
    labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]

    results = {}
    for name, fn in CANDIDATES.items():
        logits = fn(X, head)
        pred = [labels[int(j)] for j in logits.argmax(axis=1)]
        acc = float(np.mean([p == g for p, g in zip(pred, gold)]))
        results[name] = {"acc_8class": acc}
        print(f"  {name:16} acc_8class = {acc:.4f}", flush=True)

    best = max(results, key=lambda k: results[k]["acc_8class"])
    print(f"\n[probe] BEST HEAD = {best} "
          f"(acc={results[best]['acc_8class']:.4f})", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    Path(OUT).write_text(json.dumps(
        {"best": best, "results": results, "n": len(items),
         "labels": labels}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[probe] wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
