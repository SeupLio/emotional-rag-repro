"""Emotional retrieval -- function ``M`` in the paper (Section II-B3).

Four strategies are implemented exactly as described in the paper:

* ``ordinary`` : semantic similarity only (the baseline).
* ``C-A``      : combination strategy with the *add* function.
* ``C-M``      : combination strategy with the *multiple* function.
* ``S-S``      : sequential strategy, semantic first, then re-rank emotionally.
* ``S-E``      : sequential strategy, emotional first, then re-rank semantically.

Every strategy returns the indices of the ``top_k`` memories with the smallest
final distance.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from config import SEQ_POOL, TOP_K


def _semantic_dist(q_sem: np.ndarray, m_sem: np.ndarray) -> np.ndarray:
    """Euclidean distance -- the paper lists Euclidean/cosine as options and
    uses the distance form (smaller = more similar)."""
    return np.linalg.norm(m_sem - q_sem, axis=1)


def _emotion_dist(q_emo: np.ndarray, m_emo: np.ndarray) -> np.ndarray:
    """``1 - cosine`` as in Eq. (7) of the paper.

    Accepts a single query vector ``(d,)`` -- returning ``(N,)`` -- or a batch
    of queries ``(Q, d)`` -- returning ``(Q, N)``.
    """
    q = np.atleast_2d(np.asarray(q_emo, dtype=np.float64))
    m = np.atleast_2d(np.asarray(m_emo, dtype=np.float64))
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    mn = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)
    out = 1.0 - (qn @ mn.T)
    return out[0] if np.ndim(q_emo) == 1 else out


def retrieve(
    strategy: str,
    q_sem: np.ndarray,
    q_emo: np.ndarray,
    m_sem: np.ndarray,
    m_emo: np.ndarray,
    top_k: int = TOP_K,
    seq_pool: int = SEQ_POOL,
) -> np.ndarray:
    """Return the indices of the retrieved memories (best first)."""
    n = m_sem.shape[0]
    k = min(top_k, n)
    if k <= 0:
        return np.asarray([], dtype=int)

    if strategy == "ordinary":
        final = _semantic_dist(q_sem, m_sem)

    elif strategy in ("C-A", "C-M"):
        d_sem = _semantic_dist(q_sem, m_sem)
        d_emo = _emotion_dist(q_emo, m_emo)
        final = d_sem + d_emo if strategy == "C-A" else d_sem * d_emo

    elif strategy in ("S-S", "S-E"):
        d_sem = _semantic_dist(q_sem, m_sem)
        d_emo = _emotion_dist(q_emo, m_emo)
        pool = min(seq_pool, n)
        first = d_sem if strategy == "S-S" else d_emo
        second = d_emo if strategy == "S-S" else d_sem
        cand = np.argsort(first)[:pool]
        order = cand[np.argsort(second[cand])]
        return order[:k]

    else:
        raise ValueError(f"unknown retrieval strategy: {strategy}")

    return np.argsort(final)[:k]


def retrieve_batch(
    strategy: str,
    Q_sem: np.ndarray,
    Q_emo: np.ndarray,
    m_sem: np.ndarray,
    m_emo: np.ndarray,
    top_k: int = TOP_K,
    seq_pool: int = SEQ_POOL,
) -> List[np.ndarray]:
    """Vectorised-ish helper: run :func:`retrieve` for a batch of queries."""
    return [
        retrieve(strategy, Q_sem[i], Q_emo[i], m_sem, m_emo, top_k, seq_pool)
        for i in range(Q_sem.shape[0])
    ]


def retrieval_scores(
    strategy: str,
    q_sem: np.ndarray,
    q_emo: np.ndarray,
    m_sem: np.ndarray,
    m_emo: np.ndarray,
) -> Dict[str, float]:
    """Expose the two channel scores for analysis / ablation logging."""
    return {
        "semantic": float(_semantic_dist(q_sem, m_sem).mean()),
        "emotional": float(_emotion_dist(q_emo, m_emo).mean()),
    }
