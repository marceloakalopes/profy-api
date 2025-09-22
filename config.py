from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List

class Settings(BaseSettings):
    # API Settings
    API_VERSION: str = Field(default="v1")
    DEBUG: bool = Field(default=False)

    # Security
    ALLOWED_ORIGINS: List[str] = Field(default=[]) # No CORS by default

    # OpenAI
    OPENAI_API_KEY: str = Field(default="")
    OPENAI_MODEL: str = Field(default="gpt-4.1-nano")
    OPENAI_MAX_TOKENS: int = Field(default=10000)

    # File Processing
    MAX_FILE_SIZE: int = Field(default=500 * 1024) # 500 KB
    MAX_FILE_TOKENS: int = Field(default=7000)

    # API Key
    API_KEY: str = Field(default="")
    API_KEYS: List[str] = Field(default=[])

    # Mode
    MODE: str = Field(default="showcase")

    class Config:
        env_file = ".env"


settings = Settings()