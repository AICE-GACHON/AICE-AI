"""환경 설정. `.env`에서 로드하며 모델/DB 기본값은 문서 확정 스택과 일치."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # API keys
    anthropic_api_key: str = ""
    cohere_api_key: str = ""

    # Models (서비스 기획 확정 스택)
    anthropic_model: str = "claude-haiku-4-5"
    cohere_embed_model: str = "embed-multilingual-v3.0"
    embed_dim: int = 1024

    # Database
    database_url: str = "postgresql+psycopg://cardrag:cardrag@localhost:5432/cardrag"


settings = Settings()
