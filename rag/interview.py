"""Phase 3：模拟面试模式 —— 抽题 → 出题 → 点评 → 追问。"""
import random
from openai import OpenAI
import config

INTERVIEWER_SYSTEM = """你是 MockMate 的模拟面试官，一位资深大厂面试官。
面试流程：你出题 → 求职者回答 → 你点评 + 追问 → 求职者再答 → 继续追问或总结。
点评必须结合【标准答案要点】：指出答对的地方、遗漏的关键点、可以怎么说更好。
风格：专业、直接、有面试官的压迫感，但不刻薄。
每次回复：先点评求职者刚才的回答，再给出追问；觉得问够了就给出总结。
 # 排版规则（重要）
    1. 答案要有清晰的标题分层，但标题层级整体比原文档降两级：不要用 #（一级，聊天里太大），顶层用 #####，子层用 ######。
    2. 标题大小跟着内容走：内容丰富的部分用标题，简单的一句话用 **加粗** 即可，别为了标题而标题。
    3. 别照搬原文档的标题字号和表格，按上面规则自己重新组织。
"""


class Interviewer:
    def __init__(self, assistant):
        self.assistant = assistant   # HybridAssistant（用来抽知识点）
        self.llm = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    def ask(self, direction: str):
        """按方向抽一个知识点，出成面试题。返回 (题目, 知识点chunk)。"""
        hits = self.assistant.retriever.search(f"{direction} 核心概念 面试题", top_n=8)
        cid = random.choice([c for c, _ in hits])
        k = self.assistant.retriever.chunks[cid]
        resp = self.llm.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system",
                 "content": "你是资深大厂面试官。基于给定的知识点，用面试官口吻出一道自然、有区分度的面试题。只输出题目本身。"},
                {"role": "user", "content": f"方向：{direction}\n【知识点】\n{k['text']}"},
            ],
            temperature=0.9,
        )
        return resp.choices[0].message.content, k

    def chat(self, knowledge, messages):
        """面试官点评 + 追问。messages 是完整对话历史。"""
        sys_prompt = INTERVIEWER_SYSTEM + f"\n\n【标准答案要点】\n{knowledge['text']}"
        resp = self.llm.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[{"role": "system", "content": sys_prompt}] + messages,
            temperature=0.7,
        )
        return resp.choices[0].message.content

    def chat_stream(self, knowledge, messages):
        """流式点评 + 追问，逐个 yield (kind, text)，kind ∈ {'thinking', 'answer'}。"""
        sys_prompt = INTERVIEWER_SYSTEM + f"\n\n【标准答案要点】\n{knowledge['text']}"
        stream = self.llm.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[{"role": "system", "content": sys_prompt}] + messages,
            temperature=0.7,
            stream=True,
            extra_body={"thinking": {"type": "enabled"}},  # 开启思考
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                yield "thinking", rc
            c = getattr(delta, "content", None)
            if c:
                yield "answer", c