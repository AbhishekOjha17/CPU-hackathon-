import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

MONGO_URI:str = os.getenv("MONGO_URI", "mongodb://localhost:27017")


class Settings(BaseSettings):
    APP_NAME: str = "RAG API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024
    UPLOAD_DIR: str = "uploads"
    
    # LLM Settings
    LLM_PROVIDER: str = "mock"
    OPENAI_API_KEY: str = ""
    
    # RAG Settings
    TOP_K: int = 5
    CHUNK_SIZE: int = 400
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    

    
    MODEL_DIR: str ="./model"

settings = Settings()

# Create upload directory
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)