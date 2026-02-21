from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from core.config import settings
from api.routes import router



from db.mongo import connect_mongo, close_mongo

import os, uuid, requests, logging, bcrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Create app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    docs_url="/docs" if settings.DEBUG else None
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes



@app.on_event("startup")
async def startup_db_client():
    logger.info("Starting up - connecting to MongoDB...")
    connect_mongo()

# Close MongoDB connection on shutdown
@app.on_event("shutdown")
async def shutdown_db_client():
    logger.info("Shutting down - closing MongoDB connection...")
    close_mongo()

app.include_router(router, prefix="/api")
@app.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )