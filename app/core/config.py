from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+asyncpg://groundcontrol:groundcontrol@localhost:5432/groundcontrol"
    redis_url: str = "redis://localhost:6379/0"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"

    cloud_provider: str = ""
    aws_region: str = ""
    bedrock_model_id: str = ""
    vertex_project_id: str = ""
    vertex_location: str = ""
    vertex_model_id: str = ""

    local_embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384

    max_iterations: int = 3
    faithfulness_threshold: float = 0.75
    groundedness_threshold: float = 0.75

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "http://localhost:3000"
    prometheus_enabled: bool = True

    api_key: str = "changeme"
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
