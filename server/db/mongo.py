# db/mongo.py

import logging
from pymongo import MongoClient
from core.config import MONGO_URI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mongo_client = None
db = None

def connect_mongo():
    global mongo_client, db
    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        mongo_client.server_info()
        db = mongo_client.cpuHackathon
        logger.info("✅ MongoDB connected successfully")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise

def close_mongo():
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed")