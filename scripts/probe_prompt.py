"""Probe which emotion-scoring prompt makes the local backbone emit valid JSON."""
import sys, os
sys.path.insert(0, 'src')
from llm import LocalLLM
from emotion import parse_emotion_vector

llm = LocalLLM()

DIMS = ["joy", "acceptance", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
DEF_ZH = {"joy": "快乐", "acceptance": "接纳/信任", "fear": "恐惧", "surprise": "惊讶",
          "sadness": "悲伤", "disgust": "厌恶", "anger": "愤怒", "anticipation": "期待"}

V1 = """你是一个情感标注器。阅读下面的文本，按 Plutchik 情感环的 8 个维度打分，每个维度给 1 到 10 的整数（1=极弱，10=极强）。
维度：{defs}
输出格式（只输出这一行 JSON，不要任何解释）：{{"joy": <1-10>, "acceptance": <1-10>, "fear": <1-10>, "surprise": <1-10>, "sadness": <1-10>, "disgust": <1-10>, "anger": <1-10>, "anticipation": <1-10>}}
文本：{text}
JSON："""

V2 = """任务：判断下面这句话的情绪强度。
这句话的情绪在 8 个维度上分别有多强？每维给 1-10 的整数。
维度：{defs}
句子：{text}
请输出 JSON："""

V3 = """Score the emotion of the text on Plutchik's 8 dimensions (integer 1-10).
Dimensions: {defs}
Text: {text}
Output JSON only:"""

DEFS_ZH = ", ".join(f"{k}({v})" for k, v in DEF_ZH.items())
DEFS_EN = ", ".join(DIMS)

texts = ["我今天非常开心，终于见到了久违的朋友！",
         "这件事让我非常愤怒，我绝对不会原谅他。",
         "听到这个消息，我心里一阵失落和悲伤。"]
cases = []
for t in texts:
    cases.append(("zh-V1", V1.format(defs=DEFS_ZH, text=t)))
    cases.append(("zh-V2", V2.format(defs=DEFS_ZH, text=t)))
    cases.append(("en-V3", V3.format(defs=DEFS_EN, text=t)))

for tag, p in cases:
    out = llm.generate([p], max_new_tokens=48, temperature=0.0, batch_size=1)
    print(f"--- {tag}: {repr(out[0])[:220]}")
    print("    parsed:", parse_emotion_vector(out[0]))
