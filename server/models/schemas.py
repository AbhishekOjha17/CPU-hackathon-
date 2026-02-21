from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class WhatIfScenario(BaseModel):
    id: str
    title: str
    description: Optional[str] = None
    impact: str
    improvement_percentage: float
    current_value: str
    suggested_value: str
    reasoning: str
    probability_before: Optional[float] = None
    probability_after: Optional[float] = None
    risk_before: Optional[str] = None
    risk_after: Optional[str] = None
    parameter: Optional[str] = None
    action_plan: Optional[str] = None
    priority: Optional[str] = None
    source: Optional[str] = None

class WhatIfScenarios(BaseModel):
    logical_scenarios: List[WhatIfScenario]
    llm_scenarios: List[WhatIfScenario]
    combined_recommendations: List[WhatIfScenario]

class UploadResponse(BaseModel):
    success: bool
    message: str
    files_uploaded: int
    chunks_added: int
    files: List[str]
    prediction: Optional[Dict[str, Any]] = None
    features_extracted: int
    what_if_scenarios: Optional[WhatIfScenarios] = None  # Add this field

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