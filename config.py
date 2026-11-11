"""全局配置：所有可调参数集中在这里，方便面试时讲清「每个参数为什么这么设」。"""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(dotenv_path=ROOT)  # 读取项目根目录的 .env

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # 面经原文目录
CHUNKS_FILE = DATA_DIR / "chunks.jsonl"    # 分块结果
BM25_FILE = DATA_DIR / "bm25.pkl"          # BM25 索引
QDRANT_PATH = str(DATA_DIR / "qdrant")     # 向量库本地存储

# ---- 硅基流动（embedding + rerank）----
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
EMBEDDING_MODEL = "BAAI/bge-m3"               # 输出 1024 维
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-flash")

# 分块参数
CHUNK_MAX_CHARS = 800   # chunk 目标最大字符数
CHUNK_OVERLAP = 80      # 相邻 chunk 重叠字符数

# 检索参数
VECTOR_TOP_K = 20       # 粗召回每路取多少
RERANK_TOP_N = 5        # 精排后保留几条
RRF_K = 60              # RRF 融合常数

COLLECTION = "mianjing"
VECTOR_DIM = 1024

# 生成参数
TEMPERATURE = 0.1       # 调低抑制「编造」