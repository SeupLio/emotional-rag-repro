"""Main experiment: Emotional RAG on InCharacter (BFI + 16Personalities).

Usage
-----
    python scripts/run_experiment.py --out results/main.json
    python scripts/run_experiment.py --channel fused --chars Hermione-en,Sheldon-en \
        --out results/mm.json

Outputs a JSON blob with per-character predictions and the aggregated
Acc(Dim) / Acc(Full) / MSE / MAE for every retrieval strategy.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from config import (  # noqa: E402
    DATA_DIR, LOG_DIR, MULTIMODAL_ALPHA, RESULTS_DIR, SEED, TOP_K,
    RETRIEVAL_STRATEGIES, MAX_NEW_TOKENS_ANSWER,
)
from embeddings import get_encoder  # noqa: E402
from emotion import EmotionEstimator, cosine_sim  # noqa: E402
from llm import LocalLLM  # noqa: E402
from personality import (  # noqa: E402
    BFI_DIMS, MBTI_DIMS, build_qa_prompt, evaluate, gold_profiles,
    load_bfi, load_mbti, load_pdb_labels, parse_likert, profile_from_answers,
)


def likert_scores(llm, prompts, lo, hi, batch):
    """Expected Likert value under the model's first-token distribution."""
    choices = [str(i) for i in range(lo, hi + 1)]
    probs, argmax = llm.score_choices(prompts, choices, batch_size=batch)
    vals = np.arange(lo, hi + 1, dtype=np.float64)
    exp = [float(np.dot(np.asarray(p, dtype=np.float64), vals)) for p in probs]
    arg = [lo + a for a in argmax]
    return exp, arg, probs
from retrieval import retrieve  # noqa: E402

MEM_SNIPPET = 220          # characters per memory inside the prompt


def log(msg: str) -> None:
    print(msg, flush=True)


def sample_memories(mem: List[str], n: int, seed: int) -> List[str]:
    if len(mem) <= n:
        return list(mem)
    rnd = random.Random(seed)
    return rnd.sample(mem, n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chars", default="", help="comma separated character keys")
    ap.add_argument("--mem", type=int, default=128)
    ap.add_argument("--strategies", default=",".join(RETRIEVAL_STRATEGIES))
    ap.add_argument("--channel", default="text", choices=["text", "fused", "audio"])
    ap.add_argument("--alpha", type=float, default=MULTIMODAL_ALPHA)
    ap.add_argument("--out", default="results/main.json")
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--emotion-batch", type=int, default=16)
    ap.add_argument("--limit-questions", type=int, default=0)
    ap.add_argument("--tts", action="store_true", help="synthesise audio for the acoustic channel")
    ap.add_argument("--emo-source", default="anchor", choices=["anchor", "cls", "llm"],
                    help="'anchor' = zero-shot bge anchors (default), "
                         "'cls' = Go-Emotions classifier, 'llm' = backbone scoring")
    args = ap.parse_args()

    t_start = time.time()
    corpus = json.loads((DATA_DIR / "corpus.json").read_text(encoding="utf-8"))
    keys = [k.strip() for k in args.chars.split(",") if k.strip()] or sorted(corpus)
    keys = [k for k in keys if k in corpus]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    log(f"characters: {len(keys)} -> {keys}")
    log(f"strategies: {strategies}  channel={args.channel}  alpha={args.alpha}")

    llm = LocalLLM()
    if args.emo_source == "llm":
        estimator_zh = EmotionEstimator(llm, lang="zh", batch_size=args.emotion_batch)
        estimator_en = EmotionEstimator(llm, lang="en", batch_size=args.emotion_batch)
        log("text emotion channel: backbone LLM scoring (paper-faithful)")
    elif args.emo_source == "cls":
        from emo_classifier import TextEmotionEstimator
        est = TextEmotionEstimator(device="cpu", batch_size=32)
        estimator_zh = estimator_en = est
        log(f"text emotion channel: Go-Emotions ({len(est.labels)} labels)")
    else:
        from emo_anchor import AnchorEmotionEstimator

        base = AnchorEmotionEstimator(
            lambda lang: enc_cache.setdefault(lang, get_encoder(lang)))

        class _Wrap:
            def __init__(self, b, lang):
                self.b, self.lang = b, lang

            def encode(self, texts):
                return self.b.encode(texts, lang=self.lang)

        estimator_zh, estimator_en = _Wrap(base, "zh"), _Wrap(base, "en")
        log("text emotion channel: zero-shot bge anchors (no extra weights)")
    enc_cache: Dict[str, object] = {}

    acoustic = None
    if args.channel in ("fused", "audio"):
        from multimodal import AcousticEmotionEstimator, synthesize
        acoustic = AcousticEmotionEstimator(device="cpu")
        log(f"SER labels: {acoustic.labels}")

    results = {
        "meta": {
            "channel": args.channel, "alpha": args.alpha, "top_k": TOP_K,
            "mem_per_char": args.mem, "strategies": strategies, "chars": keys,
            "seed": SEED, "backbone": "Qwen2.5-1.5B-Instruct",
            "emo_source": args.emo_source,
        },
        "per_char": {},
        "metrics": {},
        "diagnostics": {},
    }

    for ci, key in enumerate(keys):
        entry = corpus[key]
        lang = entry["lang"]
        name = entry.get("name", key)
        memories = sample_memories(entry["memories"], args.mem, SEED + ci)
        log(f"\n=== [{ci+1}/{len(keys)}] {key} lang={lang} memories={len(memories)}")

        if lang not in enc_cache:
            enc_cache[lang] = get_encoder(lang)
        enc = enc_cache[lang]
        est = estimator_zh if lang == "zh" else estimator_en

        M_sem = enc.encode(memories)
        t0 = time.time()
        M_emo_text = est.encode(memories)
        log(f"    semantic {M_sem.shape}  emotion(text) {M_emo_text.shape} "
            f"in {time.time()-t0:.0f}s")

        if acoustic is not None:
            from multimodal import synthesize, fuse as fuse_vec
            wav_dir = Path(DATA_DIR).parent / "raw" / "tts" / key
            wav_dir.mkdir(parents=True, exist_ok=True)
            jobs = [{"text": m[:400], "out": str(wav_dir / f"m{i:04d}.wav")}
                    for i, m in enumerate(memories)]
            if args.tts:
                t0 = time.time()
                paths = synthesize(jobs, cache_dir=wav_dir, lang=lang)
                ok = sum(1 for p in paths if p)
                log(f"    TTS: {ok}/{len(jobs)} wavs in {time.time()-t0:.0f}s")
            else:
                paths = [j["out"] if os.path.exists(j["out"]) else None for j in jobs]
            t0 = time.time()
            M_emo_audio = acoustic.encode_files(paths)
            log(f"    emotion(audio) in {time.time()-t0:.0f}s")
        else:
            M_emo_audio = None

        results["per_char"][key] = {"lang": lang, "n_mem": len(memories),
                                    "strategies": {}}

        for scale, dims, loader in (("BFI", BFI_DIMS, load_bfi),
                                    ("16Personalities", MBTI_DIMS, load_mbti)):
            q = loader(lang)
            qids = list(q.text.keys())
            if args.limit_questions:
                # keep the questionnaire balanced across dimensions
                picks: List[str] = []
                per = max(1, args.limit_questions // len(dims))
                for d in dims:
                    picks += q.cat_questions.get(d, [])[:per]
                qids = [x for x in qids if x in picks]
            questions = [q.text[x] for x in qids]

            Q_sem = enc.encode(questions)
            Q_emo_text = est.encode(questions)
            if acoustic is not None and args.tts:
                wav_dir_q = Path(DATA_DIR).parent / "raw" / "tts" / key / scale
                wav_dir_q.mkdir(parents=True, exist_ok=True)
                jobs_q = [{"text": t[:400], "out": str(wav_dir_q / f"q{i:04d}.wav")}
                          for i, t in enumerate(questions)]
                paths_q = synthesize(jobs_q, cache_dir=wav_dir_q, lang=lang)
                Q_emo_audio = acoustic.encode_files(paths_q)
            elif acoustic is not None:
                wav_dir_q = Path(DATA_DIR).parent / "raw" / "tts" / key / scale
                paths_q = [str(wav_dir_q / f"q{i:04d}.wav") for i in range(len(questions))]
                paths_q = [p if os.path.exists(p) else None for p in paths_q]
                Q_emo_audio = acoustic.encode_files(paths_q)
            else:
                Q_emo_audio = None

            if args.channel == "text":
                Q_emo, M_emo = Q_emo_text, M_emo_text
            elif args.channel == "audio":
                Q_emo, M_emo = Q_emo_audio, M_emo_audio
            else:
                from multimodal import fuse as fuse_vec
                Q_emo = fuse_vec(Q_emo_text, Q_emo_audio, args.alpha)
                M_emo = fuse_vec(M_emo_text, M_emo_audio, args.alpha)

            for strat in strategies:
                t0 = time.time()
                prompts, emo_gaps = [], []
                for i, qid in enumerate(qids):
                    idx = retrieve(strat, Q_sem[i], Q_emo[i], M_sem, M_emo, TOP_K)
                    snips = [memories[j][:MEM_SNIPPET] for j in idx]
                    prompts.append(build_qa_prompt(name, snips, questions[i],
                                                   lang=lang, lo=q.lo, hi=q.hi))
                    if len(idx):
                        emo_gaps.append(float(np.mean(
                            1.0 - cosine_sim(Q_emo[i][None, :], M_emo[idx])[0])))
                exp_vals, arg_vals, prob_rows = likert_scores(
                    llm, prompts, q.lo, q.hi, args.batch)
                answers = {qid: v for qid, v in zip(qids, exp_vals)}
                scores, norms, bands = profile_from_answers(q, answers)
                scores_arg, norms_arg, bands_arg = profile_from_answers(
                    q, {qid: float(v) for qid, v in zip(qids, arg_vals)})
                results["per_char"][key]["strategies"].setdefault(strat, {})[scale] = {
                    "scores": scores, "norms": norms, "bands": bands,
                    "scores_argmax": scores_arg, "norms_argmax": norms_arg,
                    "bands_argmax": bands_arg,
                }
                results["diagnostics"].setdefault(f"{key}|{scale}|{strat}", {
                    "mean_emotion_dist": float(np.mean(emo_gaps)) if emo_gaps else None,
                    "n_items": len(qids),
                    "seconds": round(time.time() - t0, 1),
                    "answer_mean": float(np.mean(exp_vals)),
                    "answer_std": float(np.std(exp_vals)),
                    "argmax_hist": {str(c): int(sum(1 for a in arg_vals if a == c))
                                    for c in range(q.lo, q.hi + 1)},
                })
                log(f"    {scale:16s} {strat:9s} done in {time.time()-t0:.0f}s "
                    f"emo_dist={np.mean(emo_gaps) if emo_gaps else float('nan'):.3f}")

    # ---------------- aggregate -------------------------------------------
    # Gold = InCharacter's human annotation; the PDB block only decides which
    # dimensions are 'X' and therefore excluded (official protocol).
    labels_all = __import__("personality").load_labels()
    pdb_all = __import__("personality").load_pdb_labels()
    metrics = {}
    for scale, dims in (("BFI", BFI_DIMS), ("16Personalities", MBTI_DIMS)):
        golds = gold_profiles(labels_all, scale, dims, pdb=pdb_all)
        for strat in strategies:
            for tag, nk, bk in (("exp", "norms", "bands"),
                                ("argmax", "norms_argmax", "bands_argmax")):
                preds = {}
                for key in keys:
                    blob = results["per_char"][key]["strategies"].get(strat, {}).get(scale)
                    if not blob or nk not in blob:
                        continue
                    preds[key] = {d: {"norm": blob[nk][d], "band": blob[bk][d]}
                                  for d in dims if d in blob[nk]}
                m = evaluate(preds, golds, dims)
                metrics[f"{scale}|{strat}|{tag}"] = {
                    "acc_dim": m.acc_dim, "acc_full": m.acc_full,
                    "mse": m.mse, "mae": m.mae, "n_char": m.n_char,
                }
    results["metrics"] = metrics

    out_path = Path(RESULTS_DIR) / Path(args.out).name if not os.path.isabs(args.out) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"\nwrote {out_path}")
    for k, v in metrics.items():
        log(f"  {k:28s} Acc(Dim)={v['acc_dim']:.3f} Acc(Full)={v['acc_full']:.3f} "
            f"MSE={v['mse']:.4f} MAE={v['mae']:.4f} n={v['n_char']}")
    log(f"total {time.time()-t_start:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
