from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class UploadResponse(BaseModel):
    success: bool
    message: str
    files_uploaded: int
    chunks_added: int
    files: List[str]
    prediction: Optional[Dict[str, Any]] = None
    features_extracted: Optional[int] = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    success: bool
    answer: str
    chunks_retrieved: int

class StatsResponse(BaseModel):
    chunks: int
    has_embeddings: bool
    has_bm25: bool
    features_extracted: Optional[int] = None
    has_prediction: Optional[bool] = None

class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    risk_level: str
    confidence: Optional[float] = None