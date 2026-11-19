"""两层评估：检索层(gold_chunks → hit@k) + 答案层(gold_points → 覆盖率)。"""
import json
import re
from collections import defaultdict
from openai import OpenAI
from rag.retrieve import Retriever
import config


def load_eval_set(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def multi_hop_hit(retriever, gold_chunks, hits):
    """gold 涉及的所有源文件，是否都被 top-k 覆盖（每文件至少一块）。"""
    gold_files = {retriever.chunks[cid]["source"] for cid in gold_chunks}
    hit_files = {retriever.chunks[cid]["source"] for cid, _ in hits}
    return 1.0 if gold_files <= hit_files else 0.0


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
    text = resp.choices[0].message.content
    nums = re.findall(r"\d+", text)
    if not nums:
        return 0.0
    covered = int(nums[0])
    return min(covered, len(gold_points)) / len(gold_points)


def main(n_list=(5, 10)):
    r = Retriever()
    judge = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    items = load_eval_set(str(config.DATA_DIR / "eval" / "eval_set.jsonl"))

    # ---------- 检索层：specific + relation ----------
    for n in n_list:
        hit = defaultdict(float)
        cnt = defaultdict(int)
        for item in items:
            cat = item["category"]
            if cat == "global":
                continue
            hits = r.search(item["query"], top_n=n)
            hit[cat] += multi_hop_hit(r, item["gold_chunks"], hits)
            cnt[cat] += 1
        print(f"===== 检索层 top_n={n} =====")
        print(f"{'类别':<10}{'题数':<6}{'hit@'+str(n)}")
        for cat in sorted(cnt):
            print(f"{cat:<10}{cnt[cat]:<6}{hit[cat]/cnt[cat]:.0%}")
        print()
        # ---------- 答案层：global（要点覆盖率） ----------
        print("===== 答案层：global 要点覆盖率（LLM-as-Judge）=====")
        cov_sum, cov_cnt = 0.0, 0
        for item in items:
            if item["category"] != "global":
                continue
            answer = r(item["query"],top_n = n)            # 走完整 RAG：检索 + 生成
            cov = judge_coverage(judge, answer, item["gold_points"])
            cov_sum += cov
            cov_cnt += 1
            print(f"  {item['id']}: {cov:.0%}   ({item['query'][:20]}...)")
        if cov_cnt:
            print(f"  平均覆盖率: {cov_sum/cov_cnt:.0%}")


if __name__ == "__main__":
    main()