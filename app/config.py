from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://postgres@127.0.0.1:5432/ps06"
    test_database_url: str = "postgresql+psycopg://postgres@127.0.0.1:5432/ps06_test"
    embedder: str = "local"           # local | hashing
    embedding_dim: int = 384
    llm_provider: str = "mock"        # mock | bedrock | openai
    llm_model: str = "amazon.nova-lite-v1:0"
    aws_region: str = "ap-south-1"
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    llm_timeout: float = 90.0
    session_secret_key: str = "ps06-jklu-careeros-dev-secret-key-change-in-prod-2026"
    session_ttl_seconds: int = 86400 * 7
    auth_mode: str = "demo"
    demo_role_switch: bool = True
    rate_limit_enabled: bool = False
    redis_url: str | None = None
    task_backend: str = "memory"      # memory | redis


settings = Settings()
