# Emotional RAG 复现 + 多模态情感扩展

本地可复现的 **Emotional RAG**（arXiv:2410.23041）复现工程，并在此基础上增加了
一个**语音声学情感通道**的多模态扩展（MM-ERAG）。

> 论文：Le Huang, Hengzhi Lan, Zijun Sun, Chuan Shi, Ting Bai.
> *Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval*.
> arXiv:2410.23041 (2024-10-30).

---

## 1. 这条线为什么和「游戏 AI NPC / 多模态情感理解」相关

角色扮演智能体的核心是**记忆**：让 NPC 在长对话里保持人格一致。Emotional RAG 的做法是
在检索记忆时除语义相似度外，再引入一条**情感相似度**通道（基于心境依存记忆理论
Mood-Dependent Memory），用 Plutchik 情感环的 8 维向量度量 query 与记忆的情绪距离。
本工程把它落到游戏 NPC 场景，并补上论文缺失的那一环——**情感向量不只来自文本，
也可以来自语音**，从而把「人格记忆」与「多模态情感理解」接在一起。

## 2. 方法（对应论文符号）

| 符号 | 论文实现 | 本工程实现 |
|---|---|---|
| `F` 语义编码 | bge-base-zh 768-d | 同（英文角色用 bge-base-en） |
| `G` 情感向量 | GPT-3.5 + 情感提示词，8 维 1–10 | **替换**为 Go-Emotions 多语言分类器（见 D3） |
| `M` 检索 | ordinary / C-A / C-M / S-S / S-E | 完全一致 |
| 人格评测 | BFI + 16Personalities，Acc(Dim)/Acc(Full)/MSE/MAE | 完全一致（InCharacter 原始问卷与标签） |
| 骨干 LLM | chatglm3-6b / Qwen1.5-72B / gpt-3.5-turbo | **替换**为 Qwen2.5-1.5B-Instruct（见 D1） |

## 3. 目录结构

```
emotional-rag-repro/
├── src/
│   ├── config.py            # 全部超参，替换处标 # SUBSTITUTION
│   ├── embeddings.py        # F: bge 语义编码
│   ├── emotion.py           # G: 论文原版 LLM 情感打分（保留，用于对照）
│   ├── emo_classifier.py    # G': Go-Emotions -> Plutchik 8 维（实际使用）
│   ├── retrieval.py         # M: 五种检索策略
│   ├── llm.py               # 本地骨干 + Likert 分布打分
│   ├── personality.py       # BFI / 16Personalities 协议与指标
│   └── multimodal.py        # 语音通道：TTS -> HuBERT SER -> Plutchik + 融合
├── scripts/
│   ├── prepare_data.py      # ChatHaruhi 记忆 + InCharacter 标签 -> corpus.json
│   ├── run_experiment.py    # 主实验（5 策略 × 2 问卷 × N 角色）
│   ├── run_multimodal_eval.py  # MM-1 声学通道验证 / MM-2 双通道一致性
│   └── fetch_*.py           # 模型与数据下载（hf-mirror 多线程分段）
├── report/reproduction_report.md   # 复现差异分析报告
└── results/                 # 实验输出 JSON
```

## 4. 快速开始

```bash
pip install -r requirements.txt

# 1) 数据与模型
python scripts/fetch_chatharuhi.py        # 角色记忆语料
python scripts/prepare_data.py            # -> data/corpus.json
python scripts/fetch_models_mt.py         # bge-base-zh + Qwen2.5-1.5B
python scripts/fetch_bge_en.py            # bge-base-en
python scripts/fetch_emo_model.py         # Go-Emotions 文本情感通道
python scripts/fetch_hubert.py            # HuBERT SER 语音情感通道

# 2) 主实验（论文复现）
python scripts/run_experiment.py --chars Hermione-en,Sheldon-en,Ron-en,Dumbledore-en \
    --mem 128 --out results/main.json

# 3) 多模态扩展
python scripts/run_multimodal_eval.py --limit 480
python scripts/run_experiment.py --channel fused --tts \
    --chars Hermione-en,Sheldon-en,Ron-en,Dumbledore-en --out results/mm.json
```

> 国内网络：HuggingFace 主站不可达，所有下载脚本走 `hf-mirror.com`，并实现了
> **多线程分段下载 + 分段完整性重试**（单连接在这里只有约 40 KB/s）。

## 5. 主要结论

见 [`report/reproduction_report.md`](report/reproduction_report.md)。三句话版本：

1. **检索机制本身可以复现**——五种策略的相对关系、情感通道带来的检索分布变化都能在本地跑出来。
2. **绝对数值无法对齐**——骨干从 GPT-3.5/72B 换成 1.5B 后，人格评测的绝对精度显著下降，
   这是本次复现最大的差异来源。
3. **多模态扩展有效但受限于语音情感表达力**——声学通道在真实情感语音（RAVDESS）上能独立识别情绪，
   但在中性 TTS 合成的角色语料上信息量不足，融合增益有限（详见报告 §5）。

## 6. 引用

```bibtex
@article{huang2024emotional,
  title   = {Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval},
  author  = {Huang, Le and Lan, Hengzhi and Sun, Zijun and Shi, Chuan and Bai, Ting},
  journal = {arXiv preprint arXiv:2410.23041},
  year    = {2024}
}
```
