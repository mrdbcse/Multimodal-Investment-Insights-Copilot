from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    EMBEDDING_DIM: int = 1536
    OPENAI_API_KEY: str

    class Config:
        env_file = ".env"


settings = Settings()
