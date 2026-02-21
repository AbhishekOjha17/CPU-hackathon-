import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
import aiofiles
import json

# Add these imports at the top
import numpy as np
from typing import Dict, Any

from models.schemas import UploadResponse, QueryRequest, QueryResponse, StatsResponse, PredictionResponse
from services.rag import rag_engine
from core.config import settings

router = APIRouter()

@router.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "app": settings.APP_NAME}




@router.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload files for RAG processing and ML prediction
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    saved_files = []
    file_names = []
    
    print(f"\n{'='*60}")
    print(f"📤 Processing {len(files)} uploaded files...")
    print('='*60)
    
    for file in files:
        # Check file size
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        
        if size > settings.MAX_UPLOAD_SIZE:
            print(f"⚠️ File {file.filename} exceeds size limit, skipping")
            continue
        
        # Save file
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(settings.UPLOAD_DIR, filename)
        
        async with aiofiles.open(filepath, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        saved_files.append(filepath)
        file_names.append(file.filename)
        print(f"✅ Saved: {file.filename} -> {filename}")
    
    if not saved_files:
        raise HTTPException(status_code=400, detail="No valid files to process")
    
    # Process with RAG (this will also trigger feature extraction and prediction)
    chunks_added = rag_engine.ingest(saved_files)
    
    # Get prediction results
    prediction = rag_engine.get_prediction()
    features = rag_engine.get_features()
    
    print(f"\n{'='*60}")
    print("📊 PREDICTION RESULTS")
    print('='*60)
    if prediction:
        print(f"Risk Level: {prediction.get('risk_level', 'Unknown')}")
        print(f"Confidence: {prediction.get('confidence', 0):.2%}")
        print(f"Prediction: {prediction.get('prediction', 'N/A')}")
    print(f"Features Extracted: {len(features)}")
    print('='*60)
    
    return UploadResponse(
        success=True,
        message=f"Uploaded {len(saved_files)} files and generated prediction",
        files_uploaded=len(saved_files),
        chunks_added=chunks_added,
        files=file_names,
        prediction=prediction,
        features_extracted=len(features)
    )











@router.post("/upload-with-prediction")
async def upload_and_predict(files: List[UploadFile] = File(...)):
    """
    Upload files and get immediate prediction results
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    saved_files = []
    
    for file in files:
        # Save file
        ext = os.path.splitext(file.filename)[1]
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(settings.UPLOAD_DIR, filename)
        
        async with aiofiles.open(filepath, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        saved_files.append(filepath)
    
    # Process with RAG
    rag_engine.ingest(saved_files)
    
    # Get features and prediction
    features = rag_engine.get_features()
    prediction = rag_engine.get_prediction()
    
    return {
        "success": True,
        "files_processed": len(saved_files),
        "features_extracted": len(features),
        "prediction": prediction,
        "sample_features": dict(list(features.items())[:20])  # Return first 20 features as sample
    }

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Ask a question about the uploaded documents
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    answer = rag_engine.ask(request.query)
    
    return QueryResponse(
        success=True,
        answer=answer,
        chunks_retrieved=len(rag_engine.chunks) if rag_engine.chunks else 0
    )

@router.get("/prediction")
async def get_prediction():
    """
    Get the latest prediction results
    """
    prediction = rag_engine.get_prediction()
    features = rag_engine.get_features()
    
    if not prediction:
        return {
            "success": False,
            "message": "No prediction available. Please upload documents first."
        }
    
    return {
        "success": True,
        "prediction": prediction,
        "features_extracted": len(features),
        "features": features
    }

@router.get("/features")
async def get_features():
    """
    Get all extracted features
    """
    features = rag_engine.get_features()
    
    return {
        "success": True,
        "count": len(features),
        "features": features
    }

@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics"""
    stats = rag_engine.stats()
    return StatsResponse(**stats)

@router.post("/clear")
async def clear():
    """Clear all data"""
    rag_engine.clear()
    # Also clean up uploaded files
    for file in os.listdir(settings.UPLOAD_DIR):
        file_path = os.path.join(settings.UPLOAD_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    return {"success": True, "message": "All data cleared"}

@router.get("/files")
async def list_files():
    """List uploaded files"""
    files = []
    for f in os.listdir(settings.UPLOAD_DIR):
        file_path = os.path.join(settings.UPLOAD_DIR, f)
        if os.path.isfile(file_path):
            files.append({
                "name": f,
                "size": os.path.getsize(file_path),
                "created": os.path.getctime(file_path)
            })
    return {"files": files}