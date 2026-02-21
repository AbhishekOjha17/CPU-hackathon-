import uvicorn
from core.config import settings

if __name__ == "__main__":
    print("=" * 50)
    print("RAG API SERVER")
    print("=" * 50)
    print(f"Version: {settings.APP_VERSION}")
    print(f"Debug: {settings.DEBUG}")
    print(f"Upload dir: {settings.UPLOAD_DIR}")
    print("=" * 50)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )