"""SQLAlchemy 엔진/세션/Base. 프로토타입은 Alembic 대신 create_all로 시작."""
from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from card_rag.config import settings

engine = create_engine(settings.database_url, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def init_db() -> None:
    """pgvector 확장 보장 + 테이블 생성. (docker-compose가 확장을 이미 켜두지만 방어적으로)"""
    from sqlalchemy import text

    from card_rag.db import models  # noqa: F401  (모델 등록 목적 import)

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
