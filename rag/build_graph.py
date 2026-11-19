"""Phase 2：用 LightRAG 建知识图谱（DeepSeek 提取实体关系 + 硅基流动 embedding）。"""
import asyncio
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc
import config

WORKING_DIR = str(config.DATA_DIR / "lightrag")  # 图谱存储目录


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    """LightRAG 会调这个函数做实体关系提取和回答。这里接 DeepSeek。"""
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
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=config.VECTOR_DIM,      # 1024，bge-m3 输出维度
            max_token_size=8192,
            func=partial(
                openai_embed.func,                # 取未包装的原始函数
                model=config.EMBEDDING_MODEL,     # BAAI/bge-m3
                base_url=config.SILICONFLOW_BASE_URL,
                api_key=config.SILICONFLOW_API_KEY,
            ),
        ),
    )
    await rag.initialize_storages()

    # 插入 17 份面经（LightRAG 内部会再分块 + 抽取实体关系）
    files = sorted(config.RAW_DIR.glob("*.md"))
    for i, f in enumerate(files, 1):
        print(f"[{i}/{len(files)}] 插入 {f.name}")
        await rag.ainsert(f.read_text(encoding="utf-8"))

    print("\n图谱构建完成，开始测试 global 查询...")
    result = await rag.aquery(
        "整个知识库讲了哪几条技术主线？",
        param=QueryParam(mode="global"),
    )
    print("\n===== global 查询结果 =====\n")
    print(result)

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())