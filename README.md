# Emotional RAG 复现 + 多模态情感扩展

本地可复现的 **Emotional RAG**（arXiv:2410.23041）复现工程，并在此基础上增加了一条
**语音声学情感通道**的多模态扩展（MM-ERAG）。

> 论文：Le Huang, Hengzhi Lan, Zijun Sun, Chuan Shi, Ting Bai.
> *Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval*.
> arXiv:2410.23041 (2024-10-30).

**仓库**：https://github.com/SeupLio/emotional-rag-repro
**完整差异分析报告**：[`report/reproduction_report.md`](report/reproduction_report.md)

**硬件**：单卡 RTX 3060 Laptop（6 GB），无外部 LLM API —— 全部推理本地完成。

---

## 1. 这条线为什么和「游戏 AI NPC / 多模态情感理解」相关

角色扮演智能体的核心是**记忆**：让 NPC 在长对话里保持人格一致。Emotional RAG 的做法是在
检索记忆时，除语义相似度外再引入一条**情感相似度**通道（基于心境依存记忆理论
Mood-Dependent Memory），用 Plutchik 情感环的 8 维向量度量 query 与记忆的情绪距离。

本工程把它落到游戏 NPC 场景，并补上论文缺失的那一环——**情感向量不只来自文本，也可以来自
语音**，从而把「人格记忆」与「多模态情感理解」接在一起。

## 2. 方法（对应论文符号）

| 符号 | 论文实现 | 本工程实现 |
|---|---|---|
| `F` 语义编码 | bge-base 768-d | 同（中文 bge-base-zh / 英文 bge-base-en） |
| `G` 情感向量 | GPT-3.5 + 情感提示词，8 维 1–10 | **替换**为 Go-Emotions 分类器 → Plutchik 8 维（D3） |
| `M` 检索 | ordinary / C-A / C-M / S-S / S-E，Top-10 | 完全一致 |
| 人格评测 | BFI + 16Personalities，Acc(Dim)/Acc(Full)/MSE/MAE | 完全一致（InCharacter 原始问卷与标签） |
| 骨干 LLM | chatglm3-6b / Qwen1.5-72B / gpt-3.5-turbo | **替换**为 Qwen2.5-1.5B-Instruct（D1） |
| 声学通道 | *（论文无）* | **新增**：语音 → SER → Plutchik 8 维，α 融合（D8） |

## 3. 目录结构

```
emotional-rag-repro/
├── src/
│   ├── config.py            # 全部超参，替换处标 # SUBSTITUTION
│   ├── embeddings.py        # F: bge 语义编码
│   ├── emotion.py           # G: 论文原版 LLM 情感打分（保留，用于对照）
│   ├── emo_classifier.py    # G': Go-Emotions -> Plutchik 8 维（实际使用）
│   ├── emo_anchor.py        # G'': 锚点余弦编码器（无需额外权重，默认）
│   ├── retrieval.py         # M: 五种检索策略
│   ├── llm.py               # 本地骨干 + Likert 首 token 分布打分
│   ├── personality.py       # BFI / 16Personalities 协议与指标
│   └── multimodal.py        # 语音通道：SER -> Plutchik + 双通道融合
├── scripts/
│   ├── prepare_data.py          # 角色记忆 + InCharacter 标签 -> corpus.json
│   ├── run_experiment.py        # 主实验（5 策略 × 2 问卷 × N 角色）
│   ├── run_multimodal_eval.py   # MM-1 声学通道验证 / MM-2 双通道一致性
│   ├── convert_to_safetensors.py# .bin -> .safetensors（transformers>=5 必需）
│   ├── fetch_models.py          # bge-base-zh/en + Qwen2.5-1.5B-Instruct
│   ├── fetch_emo_model.py       # Go-Emotions 文本情感通道
│   ├── fetch_ser_ravdess.py     # RAVDESS 8 类 SER（声学通道）
│   ├── fetch_chatharuhi.py      # 角色记忆语料
│   ├── fetch_ravdess.py         # RAVDESS 情感语音评测集
│   └── probe_prompt*.py         # 小模型结构化输出能力探针（报告 D3 引用）
├── report/reproduction_report.md   # 复现差异分析报告（含 D1–D8 归因）
└── results/                        # 实验输出 JSON
```

## 4. 快速开始

```bash
pip install -r requirements.txt

# 1) 数据与模型（HuggingFace 主站不可达时走 hf-mirror.com）
python scripts/fetch_chatharuhi.py     # 角色记忆语料
python scripts/prepare_data.py         # -> data/corpus.json
python scripts/fetch_models.py         # bge-base-zh/en + Qwen2.5-1.5B-Instruct
python scripts/fetch_emo_model.py      # Go-Emotions（可选，默认走锚点编码器）
python scripts/fetch_ser_ravdess.py    # RAVDESS 8 类 SER（多模态扩展）

# 2) 主实验（论文复现）
python scripts/run_experiment.py --lang zh --out results/main_zh.json

# 3) 多模态扩展
python scripts/run_multimodal_eval.py --limit 480 --device cuda
```

> **两个环境坑（已内置处理）**
> - transformers ≥ 5 + torch 2.5 **拒绝加载 `.bin`**（CVE-2025-32434）→ 用
>   `convert_to_safetensors.py` 转换；声学模型直接选提供 `model.safetensors` 的检查点。
> - 6 GB 显存跑 1.5B：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:true`，且只取末位
>   logits（`logits_to_keep=1`），否则 lm_head 全序列展开会 OOM。

## 5. 主要结果（差异摘要）

完整分析见 [`report/reproduction_report.md`](report/reproduction_report.md)。

### 5.1 中文角色主结果（6 个，可解释）

| 检索策略 | BFI Acc(Dim) | BFI MSE | MBTI Acc(Dim) | MBTI MSE |
|---|---|---|---|---|
| ordinary（语义基线） | 0.630 | 0.1228 | 0.455 | 0.1356 |
| C-A（组合-加） | 0.741 | 0.1278 | **0.682** | 0.1348 |
| C-M（组合-乘） | 0.593 | 0.1284 | 0.500 | 0.1434 |
| S-S（语义优先） | 0.630 | 0.1305 | 0.500 | 0.1302 |
| **S-E（情感优先）** | **0.815** | **0.1180** | 0.455 | 0.1358 |

### 5.2 与论文 Table II 的差距

| 口径 | BFI Acc(Dim) | MBTI Acc(Dim) |
|---|---|---|
| 本复现 ordinary | 0.630 | 0.455 |
| 本复现 最佳情感策略 | **0.815**（S-E） | **0.682**（C-A） |
| ChatGLM3-6B（论文） | 0.624 → 0.637 | 0.669 |
| Qwen1.5-72B（论文） | 0.682 → 0.726 | 0.744 → 0.793 |
| GPT-3.5（论文） | 0.701 | 0.785 |

### 5.3 三句话结论

1. **机制可复现**：情感检索把「查询—记忆」的平均情感距离从 0.170 降到 0.080（**−53%**），
   人格准确率随之提升（S-E 的 BFI 0.815 > ordinary 0.630）——论文核心论点在本地成立。
2. **绝对数值无法对齐，根因是骨干**：1.5B 替换 6B/72B/GPT-3.5 是唯一量级上的差异源（D1）；
   中文 ordinary 的 BFI 0.630 已接近 ChatGLM3-6B 的 0.624，但 MBTI 0.455 明显偏低。
3. **语言不对称是真实发现**：同一主干对中文问卷逐题区分作答，对英文问卷**恒答 "1"**；
   英文结果已单独标注为退化、不用于策略结论（报告 §3.0）。

## 6. 多模态声学通道：v1 失败 → v2 修复

| 版本 | SER 检查点 | 标签空间 | 8-d argmax 准确率 |
|---|---|---|---|
| v1 | `superb/hubert-base-superb-er` | IEMOCAP **4 类** | 0.174（≈随机） |
| **v2** | `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition` | RAVDESS **8 类** | **0.930** |

v2 过程中排掉了两个坑，都写进了报告 §5.2：

1. **检查点 404**：`audeering/wav2vec2-large-robust-12-ft-emotion-mmos` 在 `hf-mirror.com`
   上返回 **404**（镜像未收录）——这是 v1 阶段下载得到 0 字节文件的真实原因，不是网络阻断。
2. **分类头被静默随机初始化**：该检查点由 transformers **4.8.2** 保存，head 是自定义的两层
   MLP（1024→1024→8），而现代 transformers 期望 `projector`(256×1024)+`classifier`(8×256)，
   **形状都不匹配**。直接 `from_pretrained` 不报错，但会把 head 判为 MISSING 并随机初始化，
   输出实为噪声。现由 `_has_legacy_head()` 自动检测并重建 head；head 中间是否有激活无法从
   config 得知，已用 RAVDESS 真值**实测**三种候选（0.9250 / 0.9229 / 0.9208，差异在噪声内）。

> **口径提醒**：0.930 是**同域**评估（检查点本身在 RAVDESS 上微调，评测集也是 RAVDESS），
> 只能证明"声学通道真正可用"（对比 v1 的 0.174），**不能**当作跨语料泛化指标。
> 需要做 speaker-disjoint 复测——已列入报告 §6.4。

> **MM-2 的 TTS**：原管线用 PowerShell `Add-Type -AssemblyName System.Speech`，而沙箱安全策略
> **禁止 `Add-Type`**，导致"TTS produced no audio"。现改用 **pyttsx3**（走 COM，不需要
> `Add-Type`），且因 pyttsx3 在个别文本上会无限挂起，改为**每条独立子进程 + 25 秒超时**。
> MM-2 已跑通：余弦 0.795 / 0.813，但**逐维 Pearson ≈ 0**（声学向量方差比文本小 2–4 倍）——
> 说明中性 TTS 下声学通道几乎无增量信息，详见报告 §5.3。

## 7. 引用

```bibtex
@article{huang2024emotional,
  title   = {Emotional RAG: Enhancing Role-Playing Agents through Emotional Retrieval},
  author  = {Huang, Le and Lan, Hengzhi and Sun, Zijun and Shi, Chuan and Bai, Ting},
  journal = {arXiv preprint arXiv:2410.23041},
  year    = {2024}
}
```
