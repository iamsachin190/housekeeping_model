import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    HF_TOKEN: str = ""
    CHROMA_DB_PATH: str = "./chroma_db"
    DATASET_DIR: str = "./dataset"
    IMAGES_DIR: str = "./dataset/images"

    class Config:
        env_file = ".env"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.IMAGES_DIR, exist_ok=True)
