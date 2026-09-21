# Emotional RAG — Reproduction + Multimodal Speech-Emotion Extension

> **中文摘要**：本项目在单张 RTX 3060 Laptop（6 GB）上**本地复现**了论文
> *Emotional RAG*（arXiv:2410.23041），并在其基础上新增了一条**语音声学情感通道**
> 的多模态扩展。完整的检索框架与评测协议均可复现；但因骨干从论文的 6B/72B/GPT-3.5
> 降为本地 **Qwen2.5-1.5B-Instruct**，绝对数值无法与论文对齐——这是唯一量级上的差异源。
> 多模态声学通道在真实情感语音（RAVDESS）上有效（8 类准确率 0.93，同域口径）。
> 完整差异归因见 [`report/reproduction_report.md`](report/reproduction_report.md)。

**Paper:** Le Huang, Hengzhi Lan, Zijun Sun, Chuan Shi, Ting Bai.
*Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval.* arXiv:2410.23041 (2024).
**Repo:** https://github.com/SeupLio/emotional-rag-repro
**Hardware:** single RTX 3060 Laptop (6 GB), no external LLM API — all inference runs locally.

---

## 1. Background — why this project

The core challenge for a role-playing agent (e.g. a game-AI NPC that must stay
*in-character* across a long dialogue) is **memory**: retrieving the right
memories so the agent keeps a consistent personality.

*Emotional RAG* proposes that retrieval should not be driven by **semantic**
similarity alone. Inspired by the psychology theory of *Mood-Dependent Memory*,
it adds a second, **emotional** retrieval channel: each query and each memory is
encoded as an 8-dimensional **Plutchik** emotion vector, and the two channels are
fused to rank memories before they are injected into the prompt.

This project reproduces that idea **fully offline on consumer hardware**, and adds
the piece the paper is missing: an emotion vector that does **not** have to come
from text. We add a **speech-acoustic emotion channel** — turning the work toward
**multimodal emotion understanding**, which is exactly where game-AI NPC
personality and affective computing meet.

---

## 2. What this project does (the approach)

The pipeline mirrors the paper's notation:

| Symbol | Paper | This repo |
|---|---|---|
| `F` semantic encoder | bge-base, 768-d | identical (bge-base-zh / bge-base-en) |
| `G` emotion vector | GPT-3.5 + prompt → 8-d, 1–10 | **replaced** by a Go-Emotions classifier → Plutchik 8-d (or an anchor encoder) |
| `M` retrieval | ordinary / C-A / C-M / S-S / S-E, Top-10 | identical |
| Personality eval | BFI + 16Personalities, Acc(Dim)/MSE/MAE | identical (InCharacter official protocol) |
| Backbone LLM | ChatGLM3-6B / Qwen1.5-72B / GPT-3.5 | **replaced** by Qwen2.5-1.5B-Instruct (local, 6 GB) |
| Acoustic channel | *(absent in paper)* | **added**: speech → SER → Plutchik 8-d, fused at α=0.5 |

**Multimodal extension (added here).** The text emotion channel and a new
acoustic channel both produce Plutchik 8-d vectors and are fused with
`g = α·g_text + (1−α)·g_audio`. Because the fused vector feeds the paper's *existing*
five retrieval strategies unchanged, the acoustic path is a genuinely independent
second modality, not a re-weighting of the same text signal:

```
memory / questionnaire text ──TTS──▶ wav ──▶ SER ──▶ Plutchik 8-d  g_audio
                                                                   │
query text ──(text emotion channel / anchor)──▶ g_text ─────────────┤
                                                                   ▼
                                    g = α·g_text + (1−α)·g_audio   (fuse)
                                                                   ▼
                       into the paper's 5 strategies: C-A / C-M / S-S / S-E
```

---

## 3. Reproduction results

### 3.1 Main results — Chinese characters (6, interpretable)

| Strategy | BFI Acc(Dim) | BFI MSE | MBTI Acc(Dim) | MBTI MSE |
|---|---|---|---|---|
| ordinary (semantic baseline) | 0.630 | 0.1228 | 0.455 | 0.1356 |
| C-A (combine-add) | 0.741 | 0.1278 | **0.682** | 0.1348 |
| C-M (combine-multiply) | 0.593 | 0.1284 | 0.500 | 0.1434 |
| S-S (semantic-first) | 0.630 | 0.1305 | 0.500 | 0.1302 |
| **S-E (emotion-first)** | **0.815** | **0.1180** | 0.455 | 0.1358 |

The emotional strategies beat the semantic baseline (S-E BFI 0.815 > 0.630;
C-A MBTI 0.682 > 0.455), and the average **query–memory emotion distance drops
29 %–53 %** vs ordinary — i.e. emotionally closer memories get retrieved, and the
agent's personality answers track the gold labels better. **This reproduces the
paper's core mechanism locally.**

### 3.2 Gap vs the paper (Table II)

| Setting | BFI Acc(Dim) | MBTI Acc(Dim) |
|---|---|---|
| **This repro**, ordinary | 0.630 | 0.455 |
| **This repro**, best strategy (S-E / C-A) | **0.815** / 0.682 | 0.455 / **0.682** |
| ChatGLM3-6B — ordinary (paper) | 0.624 | 0.669 |
| Qwen1.5-72B — Emotional RAG (paper) | 0.726 | 0.793 |
| GPT-3.5 — Emotional RAG (paper) | 0.701 | 0.785 |

Our Chinese *ordinary* BFI (0.630) already sits near the paper's ChatGLM3-6B
value (0.624); the best emotional strategy even pushes BFI to 0.815. See §4 for
why absolute numbers still cannot be aligned.

---

## 4. Why the numbers diverge from the paper

This is the crux of the reproduction, so it is spelled out explicitly.

**Reproduced faithfully.** The whole framework — Plutchik 8-d emotion modelling
`G`, bge semantic encoding `F`, the five fusion strategies, Top-10 injection, and
the BFI / 16Personalities evaluation with H/L/X banding — runs end-to-end with no
external API. The evaluation protocol is **strictly aligned to the InCharacter
official implementation**, so there is *no* "evaluation-mismatch" artifact
polluting the comparison.

**The mechanism holds.** On Chinese characters, emotional retrieval cuts the
average query–memory emotion distance by 29 %–53 % and raises personality
accuracy. The paper's claim — *emotion as an independent retrieval channel works*
— is confirmed on local hardware.

**But absolute accuracy does not match. The dominant cause is the backbone (D1).**
The paper uses ChatGLM3-6B / Qwen1.5-72B / GPT-3.5; we use **Qwen2.5-1.5B-Instruct**
on a 6 GB laptop. A 1.5B model has far weaker *self-consistency* on personality
questionnaires, which is the single source of difference at the scale that
matters. Everything else is a deliberate, documented substitution:

| # | Deviation | Type | Effect on metrics |
|---|---|---|---|
| **D1** | Backbone 1.5B → replaces 6B/72B/3.5 | backbone swap | **dominant**: lower absolute accuracy, English collapse |
| D2 | Answer reading: first-token Likert expectation (replaces greedy) | necessary fix | changes metric口径; not directly comparable to paper |
| D3 | Go-Emotions / anchor replaces GPT-3.5 emotion scoring | estimator swap | contract unchanged (8-d, 1–10) |
| D4/D5 | 15 of 32 InCharacter characters | data scale | conclusions use Acc(Dim)/MSE |
| D6 | ChatHaruhi memory corpus (vs Character-LLM-style) | data source | limited effect on gain direction |
| D7 | Evaluation protocol aligned to official | positive | removes spurious differences |
| D8 | Multimodal acoustic channel (added, not in paper) | extension | acoustic channel valid on real speech |

**A real finding — language asymmetry.** The *same* 1.5B backbone answers the
**Chinese** questionnaire discriminatively per item, but collapses to always
answering **"1"** on the **English** questionnaire (the first-token distribution
is ~[0.8, 0.15, 0.02, 0.01, 0.03], independent of the question). We confirmed
this is the small model's weak English instruction-following, **not a pipeline
bug**. Therefore Chinese results are the interpretable primary; English results
are kept in the repo but explicitly flagged as degraded and excluded from strategy
conclusions.

---

## 5. Multimodal acoustic channel (the extension)

| Version | SER checkpoint | Label space | 8-d argmax acc | macro-F1 (6 mapped classes) |
|---|---|---|---|---|
| v1 | `superb/hubert-base-superb-er` | IEMOCAP **4-class** | 0.174 (≈ random) | 0.053 |
| **v2** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | RAVDESS **8-class** | **0.930** | **0.936** |

**v1 failed** because a 4-class IEMOCAP checkpoint was applied to 8-class RAVDESS
(constant `anger` prediction). **v2** swaps in a RAVDESS-trained 8-class model and
fixes a silent pitfall: the checkpoint was saved by transformers 4.8.2 with a
custom 2-layer MLP head (1024→1024→8) that modern transformers loads as
**MISSING / randomly initialised without error**. The head is auto-detected and
rebuilt, with the activation choice decided empirically against RAVDESS ground
truth (0.9250 / 0.9229 / 0.9208 — within noise).

> **Honest caveat.** 0.930 is an **in-domain** number (the checkpoint was
> fine-tuned on RAVDESS, and we evaluate on RAVDESS). It proves the acoustic
> channel is *genuinely usable* (vs v1's 0.174), but is **not** a cross-corpus
> generalisation score. A speaker-disjoint re-test is future work.

**MM-2 — cross-channel agreement (now runs).** Role-memory text is synthesised to
speech and the text vs acoustic Plutchik vectors are compared. Result: cosine
≈ 0.80 but **per-dimension Pearson ≈ 0** — because neutral TTS produces nearly
*constant* acoustic emotion (its vector std is 2–4× smaller than the text channel).
So under neutral TTS the acoustic channel adds little; **the acoustic channel only
pays off when the speech actually carries emotion** (real emotional speech, as in
MM-1).

---

## 6. Limitations (honest)

- **Backbone (open):** 7B+/72B or an API backbone would bring absolute accuracy
  and English answers much closer to the paper; the local 6 GB GPU cannot hold 7B
  stably.
- **Acoustic in-domain (open):** 0.930 needs a speaker-disjoint re-test to become
  a generalisation claim.
- **MM-2 (open):** neutral TTS carries no emotion, so the acoustic channel shows
  no incremental signal — emotional TTS or real role speech is needed.
- **Data (open):** only 15 of the paper's 32 characters (those with both ChatHaruhi
  memory and InCharacter labels).

---

## 7. Reproduce

```bash
pip install -r requirements.txt

# 1) data + models (use hf-mirror.com when huggingface.co is unreachable)
python scripts/fetch_chatharuhi.py      # character memory corpus
python scripts/prepare_data.py          # -> data/corpus.json
python scripts/fetch_models.py          # bge-base-zh/en + Qwen2.5-1.5B-Instruct
python scripts/fetch_emo_model.py       # Go-Emotions (optional; anchor encoder is default)
python scripts/fetch_ser_ravdess.py     # RAVDESS 8-class SER (multimodal extension)
python scripts/fetch_ravdess.py         # RAVDESS emotional-speech eval set

# 2) main reproduction (paper)
python scripts/run_experiment.py --lang zh --out results/main_zh.json

# 3) multimodal extension
python scripts/run_multimodal_eval.py --limit 480 --device cuda
```

> **Two environment gotchas (already handled):** transformers ≥ 5 + torch 2.5
> refuse to load `.bin` (CVE-2025-32434) → `convert_to_safetensors.py` converts
> them; and 1.5B on 6 GB needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true`
> plus last-token-only logits (`logits_to_keep=1`) to avoid OOM.

---

## 8. Repository layout

```
emotional-rag-repro/
├── src/            # F (embeddings), G (emo_anchor / emo_classifier), M (retrieval),
│                   #   llm, personality, multimodal (speech channel)
├── scripts/        # prepare_data, run_experiment, run_multimodal_eval,
│                   #   convert_to_safetensors, fetch_* , ser_head_probe
├── report/         # reproduction_report.md  (full D1–D8 divergence analysis)
└── results/        # main_zh.json, main_en.json, multimodal_eval.json
```

## 9. Citation

```bibtex
@misc{huang2024emotional,
  title   = {Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval},
  author  = {Huang, Le and Lan, Hengzhi and Sun, Zijun and Shi, Chuan and Bai, Ting},
  journal = {arXiv preprint arXiv:2410.23041},
  year    = {2024}
}
```

## 10. License

[MIT](LICENSE) — free to reuse, modify, and redistribute.
