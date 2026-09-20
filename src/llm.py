"""Local HF backbone used for every LLM call in the pipeline.

The paper calls ``gpt-3.5-turbo-0125`` (and additionally ChatGLM3-6B /
Qwen1.5-72B) for three distinct jobs:

1. scoring the emotion vector,
2. acting as the role-playing agent that answers the questionnaires,
3. analysing the answers into personality predictions.

A single locally hosted model takes over all three roles here.  The swap is
the dominant source of numerical divergence and is quantified in the
reproduction report.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import BACKBONE_MODEL, DEVICE, MAX_NEW_TOKENS_EMOTION


class LocalLLM:
    def __init__(self, model_path: str = BACKBONE_MODEL, device: str = DEVICE,
                 dtype: str = "fp16", max_input: int = 2048):
        self.model_path = model_path
        self.max_input = max_input
        self.device = device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"
        dtype_map = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype_map.get(dtype, torch.float16),
            trust_remote_code=True,
        )
        self.model.eval()
        self.model = self.model.to(self.device)

    # ------------------------------------------------------------------
    def _chat(self, prompt: str) -> str:
        """Wrap a raw instruction in the model's chat template when possible."""
        try:
            msgs = [{"role": "user", "content": prompt}]
            return self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return prompt

    @torch.no_grad()
    def generate(
        self,
        prompts: Sequence[str],
        max_new_tokens: int = MAX_NEW_TOKENS_EMOTION,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
        batch_size: int = 8,
    ) -> List[str]:
        texts = [self._chat(p) for p in prompts]
        results: List[str] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i: i + batch_size]
            enc = self.tok(
                chunk, padding=True, truncation=True,
                max_length=self.max_input, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            do_sample = temperature is not None and temperature > 0
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
            gen = out[:, enc["input_ids"].shape[1]:]
            for row in gen:
                s = self.tok.decode(row, skip_special_tokens=True)
                if stop:
                    for sp in stop:
                        if sp in s:
                            s = s.split(sp)[0]
                results.append(s.strip())
        return results

    def generate_one(self, prompt: str, **kw) -> str:
        return self.generate([prompt], **kw)[0]

    # ------------------------------------------------------------------
    @torch.no_grad()
    def score_choices(
        self,
        prompts: Sequence[str],
        choices: Sequence[str],
        batch_size: int = 8,
    ) -> tuple[list[list[float]], list[int]]:
        """Distribution over ``choices`` for the first generated token.

        A 1.5B backbone *greedy*-decoding a Likert item almost always emits the
        same digit, which zeroes out any difference between retrieval
        strategies.  Reading the model's distribution over the answer tokens
        instead gives a continuous, prompt-sensitive score at the cost of a
        single forward pass (no decoding at all, so it is also much faster).
        """
        import torch

        token_ids: List[int] = []
        for c in choices:
            ids = self.tok.encode(str(c), add_special_tokens=False)
            token_ids.append(ids[0] if ids else self.tok.unk_token_id)

        probs_all: list[list[float]] = []
        argmax_all: list[int] = []
        for i in range(0, len(prompts), batch_size):
            chunk = [self._chat(p) for p in prompts[i: i + batch_size]]
            enc = self.tok(chunk, padding=True, truncation=True,
                           max_length=self.max_input, return_tensors="pt")
            enc = {k: v.to(self.device) for k, v in enc.items()}
            # Only the *last* position is needed; computing the lm_head over the
            # full sequence is what blows up a 6 GB card (151936-vocab logits).
            try:
                out = self.model(**enc, logits_to_keep=1)
            except TypeError:
                try:
                    out = self.model(**enc, num_logits_to_keep=1)
                except TypeError:
                    out = self.model(**enc)
            logits = out.logits[:, -1, :].float()
            sel = logits[:, token_ids]
            p = torch.softmax(sel, dim=-1).cpu().numpy()
            for row in p:
                probs_all.append([float(x) for x in row])
                argmax_all.append(int(np.argmax(row)))
            del enc, out, logits, sel
            torch.cuda.empty_cache()
        return probs_all, argmax_all
