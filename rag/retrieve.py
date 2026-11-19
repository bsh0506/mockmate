"""Step 3 检索层：混合检索(dense+BM25) → RRF → Rerank。"""
import json
import pickle
from re import search


import requests
import jieba
from openai import OpenAI
from qdrant_client import QdrantClient


import config

jieba.setLogLevel(20)

SYSTEM_PROMPT = """你是 MockMate，一位经验丰富的大厂面试官兼面试陪练。
        你的目标不是复述资料，而是帮求职者真正搞懂技术点、并知道面试时怎么表达。

        # 你的说话风格
        1. 结论先行：先一句话给答案，再展开。
        2. 像面对面辅导，口语化；不要用"根据资料显示""资料中提到"这种书面腔。
        3. 面向面试：经常点出"这道题面试官在考什么""你可以这样答""记住这个关键词"。
        4. 结构化但自然：可以用分点、对比，但每一点都要像人在讲。
        
         # 排版规则（重要）
        1. 答案要有清晰的标题分层，但标题层级整体比原文档降两级：不要用 #（一级，聊天里太大），顶层用 #####，子层用 ######。
        2. 标题大小跟着内容走：内容丰富的部分用标题，简单的一句话用 **加粗** 即可，别为了标题而标题。
        3. 别照搬原文档的标题字号和表格，按上面规则自己重新组织。

        # 硬性约束（任何时候不能违反）
        1. 只能基于【参考资料】回答，资料里没有的不要编造。
        2. 资料确实覆盖不到时，诚实说"这块资料里没讲到"，不要硬编。
        3. 引用来源放在答案最后，用「参考：xxx」一句话列出，别塞进正文。
        """
def rrf_merge(dense_hits, sparse_hits, top_k, k=config.RRF_K):
    scores = {}
    for rank, (cid, _) in enumerate(dense_hits):        # 遍历两路所有文档
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (cid, _) in enumerate(sparse_hits):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    merged = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
    return [(cid, s) for cid, s in merged]


class Retriever:
    def __init__(self):
        # 1. 读 chunks.jsonl（要 text 和 source/heading）
        with open(config.CHUNKS_FILE,"r") as f:
            self.chunks = [json.loads(l) for l in f.readlines()]
        self.texts = [chunk["text"] for chunk in self.chunks]
        # 2. 读 bm25.pkl
        with open(config.BM25_FILE,"rb") as f:
            self.bm25 = pickle.load(f)
        # 3. 建 Qdrant 客户端
        self.qdrant = QdrantClient(path=config.QDRANT_PATH)
        # 4. 建 embed_client（SiliconFlow）
        self.embed_client = OpenAI(api_key=config.SILICONFLOW_API_KEY, base_url=config.SILICONFLOW_BASE_URL)
        # 5. 建 deepseek客户端
        self.ds_client = OpenAI(api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL)
    def _dense(self, query, k):
        """稠密召回，返回 [(chunk_id, score)]"""
        qv = self.embed_client.embeddings.create(model=config.EMBEDDING_MODEL, input=[query])
        vec = qv.data[0].embedding  # query 的 1024 维向量
        resp = self.qdrant.query_points(collection_name=config.COLLECTION, query=vec, limit=k)
        dense_hits = [(p.id, p.score) for p in resp.points]  # [(chunk_id, 分数), ...]
        return dense_hits
    def _sparse(self, query, k):
        """BM25 召回，返回 [(chunk_id, score)]"""
        qtok = [w for w in jieba.cut(query) if w.strip()]  # query 分词
        scores = self.bm25.get_scores(qtok)  # 每个 chunk 的 BM25 分数
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        sparse_hits = [(i, scores[i]) for i in ranked]
        return sparse_hits
    def _rerank(self, query, rrf_hits, top_n):
        """调 SiliconFlow rerank，返回重排后的 [(chunk_id, score)]"""
        documents = [self.texts[i] for i,_ in rrf_hits]
        response = requests.post(
            "https://api.siliconflow.cn/v1/rerank",
            headers={"Authorization": f"Bearer {config.SILICONFLOW_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": config.RERANK_MODEL,
                "query": query,
                "documents": documents,
                "return_documents": True,
                "top_n": top_n,
            }
        )
        results = response.json()["results"]
        return [(rrf_hits[item["index"]][0],item["relevance_score"]) for item in results]

    def search(self, query, top_k=config.VECTOR_TOP_K, top_n=config.RERANK_TOP_N):
        """完整检索：召回 → RRF → 精排，返回 top_n 个 [(chunk_id, score)]"""
        dense_hits = self._dense(query, top_k)
        sparse_hits = self._sparse(query, top_k)
        rrf_hits = rrf_merge(dense_hits, sparse_hits, top_k)
        # print([self.texts[index] for index, score in rrf_hits])
        rerank = self._rerank(query, rrf_hits, top_n)
        return rerank

    def generate(self, query, context):
        user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{query}"
        response = self.ds_client.chat.completions.create( # type: ignore
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            extra_body={"thinking": {"type": "disabled"}},
        )
        return response.choices[0].message.content

    def __call__(self, query, top_n=config.RERANK_TOP_N):
        rerank = self.search(query,top_n)
        context = "\n\n".join([self.texts[i] for i, _ in rerank])

        return self.generate(query, context)

    def answer_with_sources(self, query, top_n=config.RERANK_TOP_N):
        """跟 __call__ 一样，但额外返回命中的来源。"""
        rerank = self.search(query, top_n)
        context = "\n\n".join([self.texts[i] for i, _ in rerank])
        sources = [
            {
                "chunk_id": i,
                "heading": self.chunks[i].get("heading", ""),
                "source": self.chunks[i].get("source", ""),
            }
            for i, _ in rerank
        ]
        return self.generate(query, context), sources

    def generate_stream(self, query, context):
        """流式生成：逐个 yield (kind, text)，kind ∈ {'thinking', 'answer'}。"""
        user_prompt = f"【参考资料】\n{context}\n\n【用户问题】\n{query}"
        stream = self.ds_client.chat.completions.create(
            model=config.DEEPSEEK_CHAT_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            extra_body={"thinking": {"type": "enabled"}},
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
if __name__ == "__main__":
    # 测试用：实例化 Retriever，搜一个问题，打印命中的 chunk 标题
    retriever = Retriever()
    try :
        print(retriever("给我一个大模型应用开发的学习路线"))
    finally:
        retriever.qdrant.close()