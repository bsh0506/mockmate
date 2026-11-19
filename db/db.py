"""SQLAlchemy 连接：engine + session + 声明式基类。"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import config

engine = create_engine(
    config.SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True,      # 取连接前先 ping，防 MySQL 断连
    pool_recycle=3600,       # 连接 1 小时回收
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI 依赖：每个请求一个 session，用完关闭。"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
