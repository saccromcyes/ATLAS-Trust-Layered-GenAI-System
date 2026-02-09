from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    OPENAI_BASE_URL: str
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBED_MODEL: str = "text-embedding-3-small"

    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "atlas"
    POSTGRES_USER: str = "atlas"
    POSTGRES_PASSWORD: str = "atlas"

    QDRANT_HOST: str = "qdrant"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION: str = "atlas_chunks"

    APP_ENV: str = "prod"
    LOG_LEVEL: str = "INFO"
    MAX_UPLOAD_BYTES: int = 20_000_000

settings = Settings()
