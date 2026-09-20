"""Emotional RAG replication -- shared configuration.

Reference paper
---------------
Le Huang, Hengzhi Lan, Zijun Sun, Chuan Shi, Ting Bai.
"Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval"
arXiv:2410.23041 (2024-10-30).

All hyper-parameters that the paper states explicitly are kept identical to the
original setup (bge embedding, 768-d, Plutchik 8-d emotion vector scored 1..10,
Top-10 retrieved memories).  Everything that had to be substituted because of
the local hardware is marked with ``# SUBSTITUTION`` so the reproduction
report can point at it.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = PROJECT_ROOT / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_DIR = PROJECT_ROOT / "report"
LOG_DIR = PROJECT_ROOT / "logs"

for _d in (DATA_DIR, RESULTS_DIR, REPORT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Embedding -- identical to the paper
# --------------------------------------------------------------------------
EMBED_MODEL_ZH = str(MODEL_DIR / "bge-base-zh-v1.5")   # BAAI/bge-base-zh-v1.5
EMBED_MODEL_EN = str(MODEL_DIR / "bge-base-en-v1.5")   # BAAI/bge-base-en-v1.5
EMBED_DIM = 768

# --------------------------------------------------------------------------
# Backbone LLM -- SUBSTITUTION
# The paper uses chatglm3-6b / Qwen1.5-72B-Chat-GPTQ-Int4 / gpt-3.5-turbo-0125.
# Local hardware is a single RTX 3060 Laptop (6 GB) with no LLM API access, so a
# 1.5B instruct model is used instead.  This is the single largest source of
# divergence and is analysed in the reproduction report.
# --------------------------------------------------------------------------
BACKBONE_MODEL = str(MODEL_DIR / "Qwen2.5-1.5B-Instruct")
BACKBONE_NAME = "Qwen2.5-1.5B-Instruct"
BACKBONE_DTYPE = "fp16"
MAX_NEW_TOKENS_ANSWER = 24
MAX_NEW_TOKENS_REPLY = 96
MAX_NEW_TOKENS_EMOTION = 60

# --------------------------------------------------------------------------
# Emotion vector -- identical to the paper
# Plutchik's wheel: 8 basic emotions, each scored 1..10.
# --------------------------------------------------------------------------
PLUTCHIK_EMOTIONS = [
    "joy",
    "acceptance",
    "fear",
    "surprise",
    "sadness",
    "disgust",
    "anger",
    "anticipation",
]
EMOTION_DIM = len(PLUTCHIK_EMOTIONS)
EMOTION_SCORE_MIN = 1
EMOTION_SCORE_MAX = 10

# --------------------------------------------------------------------------
# Retrieval -- identical to the paper
# --------------------------------------------------------------------------
TOP_K = 10            # Top-10 memories injected into the prompt
SEQ_POOL = 30         # intermediate pool size used by the sequential strategies

RETRIEVAL_STRATEGIES = ["ordinary", "C-A", "C-M", "S-S", "S-E"]

# --------------------------------------------------------------------------
# Multimodal extension (this project's addition, not part of the paper)
# --------------------------------------------------------------------------
# Weight of the acoustic channel when fusing text- and speech-based emotion
# vectors into a multimodal emotion vector.
MULTIMODAL_ALPHA = 0.5

# --------------------------------------------------------------------------
# Misc
# --------------------------------------------------------------------------
SEED = 20241030       # paper submission date, keeps runs reproducible
DEVICE = os.environ.get("ERAG_DEVICE", "cuda")


def ensure_dirs() -> None:
    for _d in (DATA_DIR, RESULTS_DIR, REPORT_DIR, LOG_DIR, MODEL_DIR):
        _d.mkdir(parents=True, exist_ok=True)
