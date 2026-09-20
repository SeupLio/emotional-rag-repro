"""Probe few-shot emotion scoring prompts on the local 1.5B backbone."""
import sys, os, json, re
sys.path.insert(0, 'src')
from llm import LocalLLM
from emotion import parse_emotion_vector

llm = LocalLLM()
DIMS = ["joy", "acceptance", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]

SHOT_ZH = """下面是情感标注的示例。

文本：今天真是美好的一天，我太高兴了！
JSON：{{"joy": 9, "acceptance": 6, "fear": 1, "surprise": 3, "sadness": 1, "disgust": 1, "anger": 1, "anticipation": 5}}

文本：这一切都毁了，我感到深深的悲伤和绝望。
JSON：{{"joy": 1, "acceptance": 2, "fear": 4, "surprise": 2, "sadness": 9, "disgust": 3, "anger": 3, "anticipation": 1}}

文本：{text}
JSON："""

SHOT_EN = """Examples of emotion annotation.

Text: What a wonderful day, I am so happy!
JSON: {{"joy": 9, "acceptance": 6, "fear": 1, "surprise": 3, "sadness": 1, "disgust": 1, "anger": 1, "anticipation": 5}}

Text: Everything is ruined, I feel deep sorrow and despair.
JSON: {{"joy": 1, "acceptance": 2, "fear": 4, "surprise": 2, "sadness": 9, "disgust": 3, "anger": 3, "anticipation": 1}}

Text: {text}
JSON:"""

texts_zh = ["我今天非常开心，终于见到了久违的朋友！",
            "这件事让我非常愤怒，我绝对不会原谅他。",
            "听到这个消息，我心里一阵失落和悲伤。",
            "别担心，明天一切都会好起来的，我们慢慢来。"]
texts_en = ["I am absolutely furious, I will never forgive him.",
            "I feel so lonely and hopeless tonight.",
            "Congratulations! You did a great job!",
            "I can't wait to see what happens next."]

for lang, tpl, texts in (("zh", SHOT_ZH, texts_zh), ("en", SHOT_EN, texts_en)):
    prompts = [tpl.format(text=t) for t in texts]
    outs = llm.generate(prompts, max_new_tokens=40, temperature=0.0, batch_size=4)
    print(f"======== {lang}")
    for t, o in zip(texts, outs):
        print(f"  {t[:24]:26s} -> {repr(o)[:110]}")
        print(f"      parsed: {parse_emotion_vector(o)}")
