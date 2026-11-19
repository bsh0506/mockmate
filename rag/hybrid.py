"""Phase 2 双路路由：具体题走传统 RAG，关系/全局题走 LightRAG。"""
import logging
from functools import partial
from openai import OpenAI
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
from rag.retrieve import Retriever
import config

for name in ("lightrag", "httpx2", "openai"):
    logging.getLogger(name).setLevel(logging.ERROR)   # 压掉 LightRAG 的 INFO 日志

WORKING_DIR = str(config.DATA_DIR / "lightrag")


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs):
    return await openai_complete_if_cache(
        config.DEEPSEEK_CHAT_MODEL, prompt,
        system_prompt=system_prompt, history_messages=history_messages,
        api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_BASE_URL,
        **kwargs,
    )


class HybridAssistant:
    def __init__(self, retriever, rag):
        self.retriever = retriever                      # 传统 RAG
        self.rag = rag                                  # LightRAG
        self.router = OpenAI(api_key=config.DEEPSEEK_API_KEY,
                             base_url=config.DEEPSEEK_BASE_URL)

    def route(self, query: str) -> str:
        """三层路由：规则 → 模型 → 兜底。返回 'lightrag' 或 'traditional'。"""
        # 第一层：规则（零成本，拦明显的关系/全局题）
        obvious = ["关系", "区别", "对比", "演进", "脉络", "主线", "交织", "全景", "历史"]
        if any(k in query for k in obvious):
            return "lightrag"
        # 第二层：模型路由（拿不准的交给 LLM 判断）
        try:
            prompt = (
                "判断下面这道题的类型，只输出「具体」或「关系/全局」两个字。\n"
                "- 具体：问某个概念的定义、原理、实现、步骤清单\n"
                "- 关系/全局：问概念之间的关系、对比、演进脉络、整体框架、技术主线\n"
                f"题目：{query}"
            )
            resp = self.router.chat.completions.create(
                model=config.DEEPSEEK_CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            label = resp.choices[0].message.content.strip()
            return "lightrag" if "关系" in label or "全局" in label else "traditional"
        except Exception as e:
            # 第三层：兜底（判断失败就默认传统 RAG，更稳）
            return "traditional"

    async def answer(self, query: str) -> str:
        path = self.route(query)
        print(f"[路由到 {path}]")
        if path == "lightrag":
            context = await self.rag.aquery(query, param=QueryParam(mode="hybrid")) # lightRag只负责检索
            return self.retriever.generate(query,context)
        return self.retriever(query)
    async def answer_with_meta(self, query):
        """跟 answer 一样，但返回 (answer, sources, path)。"""

        path = self.route(query)
        print(f"[路由到 {path}]")
        if path == "lightrag":
            context = await self.rag.aquery(query, param=QueryParam(mode="hybrid"))
            return self.retriever.generate(query, context), [], path
        answer, sources = self.retriever.answer_with_sources(query)
        return answer, sources, path
    async def answer_stream(self, query):
        """流式回答：路由 → 检索 → yield (kind, text)。第一条是 ('meta', {...})。"""
        path = self.route(query)
        print(f"[路由到 {path}]")
        if path == "lightrag":
            context = await self.rag.aquery(query, param=QueryParam(mode="hybrid"))
            sources = []
        else:
            rerank = self.retriever.search(query)
            context = "\n\n".join([self.retriever.texts[i] for i, _ in rerank])
            sources = [
                {"chunk_id": i,
                 "heading": self.retriever.chunks[i].get("heading", ""),
                 "source": self.retriever.chunks[i].get("source", "")}
                for i, _ in rerank
            ]
        yield "meta", {"path": path, "sources": sources}
        for kind, text in self.retriever.generate_stream(query, context):
            yield kind, text
    def close(self):
        self.retriever.qdrant.close()


async def create_assistant() -> HybridAssistant:
    retriever = Retriever()
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=config.VECTOR_DIM, max_token_size=8192,
            func=partial(openai_embed.func, model=config.EMBEDDING_MODEL,
                         base_url=config.SILICONFLOW_BASE_URL,
                         api_key=config.SILICONFLOW_API_KEY),
        ),
    )
    await rag.initialize_storages()
    return HybridAssistant(retriever, rag)