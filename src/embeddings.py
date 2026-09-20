"""Semantic encoding -- function ``F`` in the paper (bge-base, 768-d).

The paper uses ``BAAI/bge-base-zh-v1.5`` for every dataset.  We keep that
encoder for the Chinese corpora and add the English twin for the English
corpus; both are 768-d BERT models with identical pooling, so the geometry the
retrieval strategies operate on is unchanged.

Loaded with plain ``transformers`` (CLS pooling + L2 normalisation) instead of
``sentence-transformers`` to keep the dependency surface small.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from config import DATA_DIR, DEVICE, EMBED_MODEL_EN, EMBED_MODEL_ZH


class BGEEncoder:
    def __init__(self, model_path: str, device: str = DEVICE, max_len: int = 256,
                 batch_size: int = 64, cache_dir: str | Path | None = None):
        self.model_path = model_path
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()
        if device.startswith("cuda") and torch.cuda.is_available():
            self.model = self.model.to(device).half()
        else:
            self.device = "cpu"
            self.model = self.model.to("cpu")
        self.cache_dir = Path(cache_dir) if cache_dir else DATA_DIR / "emb_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _cache_name(texts: Sequence[str]) -> str:
        h = hashlib.md5(("\x1f".join(texts)).encode("utf-8")).hexdigest()[:16]
        return f"emb_{len(texts)}_{h}.npy"

    def encode(self, texts: Sequence[str], use_cache: bool = True) -> np.ndarray:
        texts = list(texts)
        cpath = self.cache_dir / self._cache_name(texts)
        if use_cache and cpath.exists():
            try:
                arr = np.load(cpath)
                if arr.shape[0] == len(texts):
                    return arr
            except Exception:
                pass

        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i: i + self.batch_size]
                enc = self.tok(
                    batch, padding=True, truncation=True,
                    max_length=self.max_len, return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                out = self.model(**enc)
                # bge uses the [CLS] token representation
                cls = out.last_hidden_state[:, 0]
                cls = torch.nn.functional.normalize(cls, p=2, dim=1)
                outs.append(cls.float().cpu().numpy())
        arr = np.concatenate(outs, axis=0) if outs else np.zeros((0, 768), dtype=np.float32)
        if use_cache:
            try:
                np.save(cpath, arr)
            except Exception:
                pass
        return arr


def get_encoder(lang: str = "zh", **kw) -> BGEEncoder:
    path = EMBED_MODEL_ZH if lang == "zh" else EMBED_MODEL_EN
    if not os.path.exists(os.path.join(path, "config.json")):
        # fall back to whichever encoder is actually present
        path = EMBED_MODEL_ZH if os.path.exists(
            os.path.join(EMBED_MODEL_ZH, "config.json")) else EMBED_MODEL_EN
    return BGEEncoder(path, **kw)
