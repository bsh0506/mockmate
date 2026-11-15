"""Phase 2 对比评估：全局题覆盖率 —— 传统 RAG vs LightRAG。"""
import json
import logging
import re
import asyncio
from functools import partial
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from retrieve import Retriever
import config

WORKING_DIR = "./data/lightrag"

for name in ("lightrag", "httpx2", "openai"):
    logging.getLogger(name).setLevel(logging.ERROR)
def load_eval_set(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def judge_coverage(judge, answer, gold_points):
    """LLM-as-Judge：判断答案覆盖了几个 gold 要点，返回 0~1 覆盖率。"""
    prompt = f"""下面是助手对一道题的回答，以及这道题"标准答案应覆盖的要点"。
请判断回答覆盖了其中几个要点（意思点到就算覆盖）。
只输出一个 0 到 {len(gold_points)} 的整数，不要输出其他内容。

【回答】
{answer}

【应覆盖的要点】
{json.dumps(gold_points, ensure_ascii=False)}
"""
    resp = judge.chat.completions.create(
        model=config.DEEPSEEK_CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    nums = re.findall(r"\d+", resp.choices[0].message.content)
    if not nums:
        return 0.0
    covered = int(nums[0])
    return min(covered, len(gold_points)) / len(gold_points)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    return await openai_complete_if_cache(
        config.DEEPSEEK_CHAT_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        **kwargs,
    )


async def main():
    # 传统 RAG + Judge
    retriever = Retriever()
    judge = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)

    # LightRAG（复用已经建好的图，不会重新建图）
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=config.VECTOR_DIM,
            max_token_size=8192,
            func=partial(
                openai_embed.func,
                model=config.EMBEDDING_MODEL,
                base_url=config.SILICONFLOW_BASE_URL,
                api_key=config.SILICONFLOW_API_KEY,
            ),
        ),
    )
    await rag.initialize_storages()

    items = [it for it in load_eval_set("./data/eval/eval_set.jsonl") if it["category"] == "global"]

    print(f"{'id':<6}{'传统RAG':<12}{'LightRAG':<12}")
    print("-" * 30)
    trad_sum = lr_sum = 0.0
    for item in items:
        q = item["query"]
        points = item["gold_points"]

        trad_ans = retriever(q)                                        # 传统 RAG
        trad_cov = judge_coverage(judge, trad_ans, points)

        lr_resp = await rag.aquery(q, param=QueryParam(mode="hybrid"))  # LightRAG
        lr_ans = lr_resp if isinstance(lr_resp, str) else str(lr_resp)
        lr_cov = judge_coverage(judge, lr_ans, points)

        trad_sum += trad_cov
        lr_sum += lr_cov
        print(f"{item['id']:<6}{trad_cov:<12.0%}{lr_cov:<12.0%}")

    n = len(items)
    print("-" * 30)
    print(f"{'平均':<6}{trad_sum/n:<12.0%}{lr_sum/n:<12.0%}")

    await rag.finalize_storages()
    retriever.qdrant.close()


if __name__ == "__main__":
    asyncio.run(main())