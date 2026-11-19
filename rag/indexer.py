import json
import pickle

import jieba
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

import config

jieba.setLogLevel(20)
def tokenize(text: str):
    return [w for w in jieba.cut(text) if w.strip()]
def embed_texts(client, texts, batch_size=32):
    """批量调 embedding，返回 1024 维向量列表"""
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=config.EMBEDDING_MODEL, input=batch)
        vectors.extend([item.embedding for item in resp.data])
    return vectors
def main():
    try:
        # 1. 读 chunks
        with open(config.CHUNKS_FILE, encoding="utf-8") as f:
            chunks = [json.loads(line) for line in f]
        texts = [c["text"] for c in chunks]
        print(f"读取 {len(chunks)} 个 chunk")

        # 2. 调 embedding API 生成向量
        client = OpenAI(api_key=config.SILICONFLOW_API_KEY, base_url=config.SILICONFLOW_BASE_URL)
        print("调用 embedding API...")
        vectors = embed_texts(client, texts)
        print(f"得到 {len(vectors)} 个向量")

        # 3. 写 Qdrant 向量库
        qdrant = QdrantClient(path=config.QDRANT_PATH)  # 本地嵌入式模式，数据存 data/qdrant
        if qdrant.collection_exists(config.COLLECTION):  # 已存在就先删掉，避免重复写
            qdrant.delete_collection(config.COLLECTION)
        qdrant.create_collection(
            collection_name=config.COLLECTION,
            vectors_config=VectorParams(size=config.VECTOR_DIM, distance=Distance.COSINE),
        )
        points = [
            PointStruct(
                id=c["id"],
                payload={"source": c["source"], "heading": c["heading"]},
                vector=v,  # 元数据，展示/溯源用
            )
            for c, v in zip(chunks, vectors)
        ]
        qdrant.upsert(collection_name=config.COLLECTION, points=points)
        print("向量库写入完成")

        # 4. 构建 BM25 关键词索引
        print("构建 BM25 索引...")
        tokenized = [tokenize(t) for t in texts]
        bm25 = BM25Okapi(tokenized)
        with open(config.BM25_FILE, "wb") as f:
            pickle.dump(bm25, f)
        print("BM25 索引完成，全部索引构建完毕")
    except FileNotFoundError:
        print('文件不存在！')
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()
