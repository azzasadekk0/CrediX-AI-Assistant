"""Environment configuration for the standalone Railway service."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "CrediX AI Assistant"
    app_env: str = "production"
    database_url: str = "sqlite:///./ai_assistant.db"
    cors_origins: str = "http://localhost:3000"
    supabase_url: str
    supabase_jwt_audience: str = "authenticated"
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.5-flash-lite"
    gemini_api_key: str | None = None
    groq_api_key: str | None = None
    credix_reranker_device: str = "auto"
    llm_timeout_seconds: float = 45.0
    llm_max_retries: int = 2

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_prefix="",
    )

    @property
    def normalized_origins(self) -> list[str]:
        return [item.strip().rstrip("/") for item in self.cors_origins.split(",") if item.strip()]

    @property
    def supabase_issuer(self) -> str:
        return f"{self.supabase_url.rstrip('/')}/auth/v1"

    @property
    def supabase_jwks_url(self) -> str:
        return f"{self.supabase_issuer}/.well-known/jwks.json"


settings = Settings()
