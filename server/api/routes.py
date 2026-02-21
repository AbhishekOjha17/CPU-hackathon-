import os
import uuid
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List
import aiofiles
import json



 
from datetime import datetime, timezone
from bson import ObjectId


# Add these imports at the top
import numpy as np
from typing import Dict, Any

from models.schemas import UploadResponse, QueryRequest, QueryResponse, StatsResponse, PredictionResponse
from services.rag import rag_engine
from services.llm import llm_service
from core.config import settings

router = APIRouter()

from pymongo import MongoClient
from core.config import MONGO_URI

mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
mongo_client.server_info()
db = mongo_client.cpuHackathon


@router.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "app": settings.APP_NAME}

@router.post("/upload-test" )
async def upppp():
    """New file"""
    
    
    mongo_save_success = False
    mongo_error = None
    
    try:
        # Check if db is connected'
         
        if db is not None:
            print(f"   ✅ db object exists: {db}")
            
            # Test connection with ping
            try:
                db.command('ping')
                print("   ✅ MongoDB ping successful")
            except Exception as ping_error:
                print(f"   ⚠️ MongoDB ping failed: {ping_error}")
                # Continue anyway, might still work
            
            # Prepare document for MongoDB with timezone-aware datetime
            loan_application = { 
                "mongo_save_status": "attempted"
            }
            
            print(f"   📄 Document prepared with {len(loan_application)} fields")
            
            # Insert into MongoDB
            result = db.loan_applications.insert_one(loan_application)
            print(f"   ✅ SAVED to MongoDB with ID: {result.inserted_id}")
            mongo_save_success = True
            
            # Verify the save (optional)
            try:
                saved_doc = db.loan_applications.find_one({"_id": result.inserted_id})
                if saved_doc:
                    print(f"   ✅ Verification: Document found in database")
                else:
                    print(f"   ⚠️ Verification: Document NOT found after save")
            except Exception as verify_error:
                print(f"   ⚠️ Verification failed: {verify_error}")
                
        else:
            mongo_error = "db object is None - MongoDB not connected"
            print(f"   ❌ {mongo_error}")
            print("   📝 Check: Is MongoDB running? Is MONGO_URI correct?")
            
            # Try to get more info about the db object
            try:
                from db.mongo import mongo_client
                print(f"   📝 mongo_client: {mongo_client}")
                if mongo_client:
                    print(f"   📝 mongo_client exists but db is None - check database name")
            except Exception as debug_error:
                print(f"   📝 Debug info unavailable: {debug_error}")
                
    except Exception as e: 
        print(f"   📝 MongoDB connection state - db: {db}, mongo_client: {mongo_client}")
 
                
                
                
                
                
                
                
                
@router.post("/upload", response_model=UploadResponse)
async def upload_files(files: List[UploadFile] = File(...)):
    """
    Upload files for RAG processing and ML prediction
    """
    import traceback
    from datetime import timezone
    
    request_id = str(uuid.uuid4())[:8]
    
    print(f"\n{'='*60}")
    print(f"📤 [{request_id}] Processing {len(files)} uploaded files...")
    print('='*60)
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    saved_files = []
    file_names = []
    file_errors = []
    
    # File processing section with better error tracking
    for idx, file in enumerate(files, 1):
        try:
            print(f"\n📄 [{request_id}] Processing file {idx}/{len(files)}: {file.filename}")
            
            # Check file size
            file.file.seek(0, 2)
            size = file.file.tell()
            file.file.seek(0)
            
            print(f"   📏 Size: {size / 1024:.2f} KB")
            
            if size > settings.MAX_UPLOAD_SIZE:
                error_msg = f"File exceeds size limit ({settings.MAX_UPLOAD_SIZE/1024/1024:.1f} MB)"
                print(f"   ⚠️ {error_msg}")
                file_errors.append({"file": file.filename, "error": error_msg})
                continue
            
            # Save file
            ext = os.path.splitext(file.filename)[1]
            filename = f"{uuid.uuid4().hex}{ext}"
            filepath = os.path.join(settings.UPLOAD_DIR, filename)
            
            # Ensure upload directory exists
            os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
            
            async with aiofiles.open(filepath, 'wb') as f:
                content = await file.read()
                await f.write(content)
            
            saved_files.append(filepath)
            file_names.append(file.filename)
            print(f"   ✅ Saved: {file.filename} -> {filename}")
            
        except Exception as e:
            error_msg = f"Failed to process file: {str(e)}"
            print(f"   ❌ {error_msg}")
            print(f"   🔍 Traceback: {traceback.format_exc()}")
            file_errors.append({"file": file.filename, "error": error_msg})
    
    if not saved_files:
        error_detail = "No valid files to process"
        if file_errors:
            error_detail += f". Errors: {file_errors}"
        raise HTTPException(status_code=400, detail=error_detail)
    
    # RAG Processing section
    print(f"\n{'='*60}")
    print(f"🤖 [{request_id}] Starting RAG processing...")
    print('='*60)
    
    try:
        chunks_added = rag_engine.ingest(saved_files)
        print(f"   ✅ RAG ingestion complete: {chunks_added} chunks created")
    except Exception as e:
        print(f"   ❌ RAG ingestion failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"RAG processing failed: {str(e)}")
    
    # Get prediction results
    try:
        prediction = rag_engine.get_prediction()
        features = rag_engine.get_features()
        print(f"   ✅ Prediction retrieved: {prediction.get('risk_level', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Failed to get prediction: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        prediction = {}
        features = {}
    
    # Generate what-if scenarios
    try:
        print(f"\n🧠 [{request_id}] Generating what-if scenarios...")
        what_if_scenarios = await generate_what_if_scenarios(features, prediction)
        print(f"   ✅ Generated {len(what_if_scenarios.get('combined_recommendations', []))} scenarios")
    except Exception as e:
        print(f"   ❌ What-if scenario generation failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        what_if_scenarios = {"logical_scenarios": [], "llm_scenarios": [], "combined_recommendations": []}
    
    # Generate detailed risk analysis
    try:
        print(f"\n🔬 [{request_id}] Generating detailed risk analysis...")
        risk_analysis = await generate_risk_analysis(features, prediction, request_id)
        print(f"   ✅ Risk analysis generated successfully")
    except Exception as e:
        print(f"   ❌ Risk analysis generation failed: {e}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        risk_analysis = get_fallback_risk_analysis(features, prediction)
    
    # Generate application ID
    application_id = str(uuid.uuid4())
    
    # --- SAVE TO MONGODB WITH IMPROVED ERROR HANDLING ---
    print(f"\n{'='*60}")
    print(f"💾 [{request_id}] Attempting to save to MongoDB...")
    print('='*60)
    
    mongo_save_success = False
    mongo_error = None
    
    try:
        # Check if db is connected
        if db is not None:
            print(f"   ✅ db object exists: {db}")
            
            # Test connection with ping
            try:
                db.command('ping')
                print("   ✅ MongoDB ping successful")
            except Exception as ping_error:
                print(f"   ⚠️ MongoDB ping failed: {ping_error}")
                # Continue anyway, might still work
            
            # Prepare document for MongoDB with timezone-aware datetime
            loan_application = {
                "application_id": application_id,
                "request_id": request_id,
                "created_at": datetime.now(timezone.utc),
                "files_processed": file_names,
                "file_paths": saved_files,
                "file_errors": file_errors if file_errors else None,
                "chunks_added": chunks_added,
                "extracted_features": features,
                "prediction_result": prediction,
                "what_if_scenarios": what_if_scenarios,
                "risk_analysis": risk_analysis,
                "features_count": len(features),
                "mongo_save_status": "attempted"
            }
            
            print(f"   📄 Document prepared with {len(loan_application)} fields")
            
            # Insert into MongoDB
            result = db.loan_applications.insert_one(loan_application)
            print(f"   ✅ SAVED to MongoDB with ID: {result.inserted_id}")
            mongo_save_success = True
            
            # Verify the save (optional)
            try:
                saved_doc = db.loan_applications.find_one({"_id": result.inserted_id})
                if saved_doc:
                    print(f"   ✅ Verification: Document found in database")
                else:
                    print(f"   ⚠️ Verification: Document NOT found after save")
            except Exception as verify_error:
                print(f"   ⚠️ Verification failed: {verify_error}")
                
        else:
            mongo_error = "db object is None - MongoDB not connected"
            print(f"   ❌ {mongo_error}")
            print("   📝 Check: Is MongoDB running? Is MONGO_URI correct?")
            
            # Try to get more info about the db object
            try:
                from db.mongo import mongo_client
                print(f"   📝 mongo_client: {mongo_client}")
                if mongo_client:
                    print(f"   📝 mongo_client exists but db is None - check database name")
            except Exception as debug_error:
                print(f"   📝 Debug info unavailable: {debug_error}")
            
    except Exception as e:
        mongo_error = str(e)
        print(f"   ❌ Failed to save to MongoDB: {mongo_error}")
        print(f"   🔍 Traceback: {traceback.format_exc()}")
        
        # Try to get more diagnostic info
        try:
            print(f"   📝 MongoDB connection state - db: {db}, mongo_client: {mongo_client}")
        except:
            pass
    
    # Final status
    if mongo_save_success:
        print(f"\n✅ [{request_id}] MongoDB save completed successfully")
    else:
        print(f"\n⚠️ [{request_id}] MongoDB save skipped/failed: {mongo_error or 'Unknown reason'}")
        print("   💡 Application will continue without database storage")
    
    # Print prediction results
    print(f"\n{'='*60}")
    print(f"📊 [{request_id}] PREDICTION RESULTS")
    print('='*60)
    if prediction:
        print(f"   • Risk Level: {prediction.get('risk_level', 'Unknown')}")
        print(f"   • Confidence: {prediction.get('confidence', 0):.2%}")
        print(f"   • Prediction: {prediction.get('prediction', 'N/A')}")
    else:
        print(f"   • No prediction available")
    print(f"   • Features Extracted: {len(features)}")
    print(f"   • Files Processed: {len(saved_files)}/{len(files)}")
    if file_errors:
        print(f"   • Files with Errors: {len(file_errors)}")
    print('='*60)
    
    return UploadResponse(
        success=True,
        message=f"Uploaded {len(saved_files)} files and generated prediction",
        files_uploaded=len(saved_files),
        chunks_added=chunks_added,
        files=file_names,
        prediction=prediction,
        features_extracted=len(features),
        what_if_scenarios=what_if_scenarios,
        risk_analysis=risk_analysis,
        application_id=application_id
    )

async def generate_what_if_scenarios(features: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate what-if scenarios using a combination of LLM and logical rules
    """
    
    # 1. LOGICAL SCENARIOS (Rule-based improvements)
    logical_scenarios = generate_logical_scenarios(features, prediction)
    
    # 2. LLM SCENARIOS (AI-generated suggestions)
    llm_scenarios = await generate_llm_scenarios(features, prediction)
    
    return {
        "logical_scenarios": logical_scenarios,
        "llm_scenarios": llm_scenarios,
        "combined_recommendations": merge_scenarios(logical_scenarios, llm_scenarios)
    }

def generate_logical_scenarios(features: Dict[str, Any], prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate what-if scenarios using pure logic for key parameters
    """
    scenarios = []
    
    # Helper function to calculate probability improvement
    def calculate_improvement(current_prob, improvement_factor):
        if current_prob:
            new_prob = min(current_prob + (current_prob * improvement_factor), 0.95)
            return new_prob
        return 0
    
    current_prob = prediction.get('probability', 0.7) if prediction else 0.7
    current_risk = prediction.get('risk_level', 'Medium Risk') if prediction else 'Medium Risk'
    
    # Scenario 1: Loan Amount Reduction
    if 'loan_amount_requested' in features:
        current_amount = features.get('loan_amount_requested', 0)
        if current_amount:
            reduced_amount = current_amount * 0.8  # 20% reduction
            improvement = 0.15  # 15% improvement
            new_prob = calculate_improvement(current_prob, improvement)
            
            scenarios.append({
                "id": "scenario_loan_amount",
                "title": "💰 Reduce Loan Amount",
                "description": f"Reducing your loan amount from ₹{current_amount:,.0f} to ₹{reduced_amount:,.0f} (20% reduction)",
                "impact": "Positive",
                "improvement_percentage": 15,
                "current_value": f"₹{current_amount:,.0f}",
                "suggested_value": f"₹{reduced_amount:,.0f}",
                "reasoning": "Lower loan amounts reduce your debt burden and improve loan-to-income ratio, making you less risky to lenders.",
                "probability_before": current_prob,
                "probability_after": new_prob,
                "risk_before": current_risk,
                "risk_after": "Low Risk" if new_prob > 0.8 else "Medium Risk" if new_prob > 0.5 else "High Risk"
            })
    
    # Scenario 2: Increase Down Payment (Loan-to-Value Ratio)
    if 'loan_to_value_ratio' in features:
        current_ltv = features.get('loan_to_value_ratio', 0.8)
        if current_ltv:
            improved_ltv = max(current_ltv - 0.15, 0.5)  # 15% reduction, min 50%
            improvement = 0.12
            new_prob = calculate_improvement(current_prob, improvement)
            
            scenarios.append({
                "id": "scenario_ltv",
                "title": "🏠 Increase Down Payment",
                "description": f"Increasing your down payment to improve Loan-to-Value ratio from {current_ltv:.0%} to {improved_ltv:.0%}",
                "impact": "Positive",
                "improvement_percentage": 12,
                "current_value": f"{current_ltv:.0%}",
                "suggested_value": f"{improved_ltv:.0%}",
                "reasoning": "A lower Loan-to-Value ratio means you have more equity in the property, reducing lender's risk.",
                "probability_before": current_prob,
                "probability_after": new_prob,
                "risk_before": current_risk,
                "risk_after": "Low Risk" if new_prob > 0.8 else "Medium Risk" if new_prob > 0.5 else "High Risk"
            })
    
    # Scenario 3: Improve CIBIL/Credit Score
    if 'credit_score' in features:
        current_score = features.get('credit_score', 650)
        if current_score:
            improved_score = min(current_score + 75, 900)  # Increase by 75 points
            improvement = 0.20  # 20% improvement
            new_prob = calculate_improvement(current_prob, improvement)
            
            scenarios.append({
                "id": "scenario_credit_score",
                "title": "📊 Improve Credit Score",
                "description": f"Improving your CIBIL/Credit score from {current_score} to {improved_score} (+75 points)",
                "impact": "Very High",
                "improvement_percentage": 20,
                "current_value": str(current_score),
                "suggested_value": str(improved_score),
                "reasoning": "Credit score is the single most important factor in loan decisions. Higher scores indicate better repayment history.",
                "probability_before": current_prob,
                "probability_after": new_prob,
                "risk_before": current_risk,
                "risk_after": "Low Risk" if new_prob > 0.8 else "Medium Risk" if new_prob > 0.5 else "High Risk"
            })
    
    # Scenario 4: Reduce Existing EMIs
    if 'existing_emi_obligations' in features:
        current_emi = features.get('existing_emi_obligations', 0)
        if current_emi:
            reduced_emi = current_emi * 0.5  # 50% reduction (pay off some loans)
            improvement = 0.18
            new_prob = calculate_improvement(current_prob, improvement)
            
            scenarios.append({
                "id": "scenario_existing_emi",
                "title": "💳 Reduce Existing EMIs",
                "description": f"Paying off existing loans to reduce monthly EMIs from ₹{current_emi:,.0f} to ₹{reduced_emi:,.0f}",
                "impact": "High",
                "improvement_percentage": 18,
                "current_value": f"₹{current_emi:,.0f}",
                "suggested_value": f"₹{reduced_emi:,.0f}",
                "reasoning": "Lower existing EMI obligations mean more disposable income for new loan repayments.",
                "probability_before": current_prob,
                "probability_after": new_prob,
                "risk_before": current_risk,
                "risk_after": "Low Risk" if new_prob > 0.8 else "Medium Risk" if new_prob > 0.5 else "High Risk"
            })
    
    # Scenario 5: Increase Income (if co-applicant available)
    if 'monthly_income' in features:
        current_income = features.get('monthly_income', 0)
        if current_income:
            increased_income = current_income * 1.2  # 20% increase (promotion/second job)
            improvement = 0.10
            new_prob = calculate_improvement(current_prob, improvement)
            
            scenarios.append({
                "id": "scenario_income",
                "title": "💼 Increase Monthly Income",
                "description": f"Increasing monthly income from ₹{current_income:,.0f} to ₹{increased_income:,.0f} (20% increase)",
                "impact": "Moderate",
                "improvement_percentage": 10,
                "current_value": f"₹{current_income:,.0f}",
                "suggested_value": f"₹{increased_income:,.0f}",
                "reasoning": "Higher income improves your debt-to-income ratio and repayment capacity.",
                "probability_before": current_prob,
                "probability_after": new_prob,
                "risk_before": current_risk,
                "risk_after": "Low Risk" if new_prob > 0.8 else "Medium Risk" if new_prob > 0.5 else "High Risk"
            })
    
    # Scenario 6: Add Co-applicant
    if 'co_applicant_income' in features:
        current_co_income = features.get('co_applicant_income', 0)
        if current_co_income == 0 or current_co_income < 10000:
            suggested_co_income = 25000
            improvement = 0.15
            new_prob = calculate_improvement(current_prob, improvement)
            
            scenarios.append({
                "id": "scenario_coapplicant",
                "title": "👥 Add Co-applicant",
                "description": f"Adding a co-applicant with monthly income of ₹{suggested_co_income:,.0f}",
                "impact": "Positive",
                "improvement_percentage": 15,
                "current_value": "No co-applicant",
                "suggested_value": f"₹{suggested_co_income:,.0f}",
                "reasoning": "A co-applicant with stable income strengthens your application and increases total household income.",
                "probability_before": current_prob,
                "probability_after": new_prob,
                "risk_before": current_risk,
                "risk_after": "Low Risk" if new_prob > 0.8 else "Medium Risk" if new_prob > 0.5 else "High Risk"
            })
    
    return scenarios

async def generate_llm_scenarios(features: Dict[str, Any], prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Use LLM to generate intelligent what-if scenarios for complex parameters
    """
    
    # Prepare features summary for LLM
    important_features = {k: v for k, v in features.items() 
                         if k in ["credit_score", "monthly_income", "loan_amount_requested", 
                                 "existing_emi_obligations", "employment_type", "education_level",
                                 "past_defaults_count", "credit_history_length_years", 
                                 "debt_to_income_ratio", "loan_purpose"]}
    
    # Enhanced prompt with stricter JSON instructions
    prompt = f"""You are a financial advisor specializing in loan eligibility optimization. 
Based on the applicant's current profile and loan prediction, provide 5 personalized what-if scenarios that would improve their chances of loan approval.

Current Applicant Profile:
{json.dumps(important_features, indent=2)}

Current Prediction:
- Risk Level: {prediction.get('risk_level', 'Unknown')}
- Confidence: {prediction.get('confidence', 0):.2%}
- Probability: {prediction.get('probability', 0):.2%}

For each scenario, provide:
1. A specific parameter to change (from the 77 features list)
2. The suggested new value
3. The reasoning why this change would help
4. Estimated improvement percentage (0-100%)
5. A brief explanation of how the user can achieve this change

Focus on parameters that are:
- Within the user's control (not fixed demographics)
- Financially feasible
- Have significant impact on loan eligibility
- Specific and actionable

IMPORTANT: Return ONLY a valid JSON array. No explanations, no markdown, no code blocks, no additional text. Just pure JSON.
The response must start with '[' and end with ']'.

Format:
[
    {{
        "parameter": "feature_name",
        "current_value": "current value",
        "suggested_value": "suggested new value",
        "improvement_percentage": 15,
        "reasoning": "Why this helps",
        "action_plan": "How to achieve this change"
    }}
]

Make scenarios realistic and personalized to the applicant's current profile."""
    
    try:
        # Call LLM service
        response = llm_service(prompt)
        print(f"LLM Response: {response[:200]}...")  # Debug: print first 200 chars
        
        # Parse response with robust method
        scenarios = parse_llm_json_response(response)
        
        if scenarios and isinstance(scenarios, list) and len(scenarios) > 0:
            # Format scenarios for response
            formatted_scenarios = []
            for i, scenario in enumerate(scenarios[:5]):
                # Ensure all required fields exist
                formatted_scenarios.append({
                    "id": f"llm_scenario_{i+1}",
                    "title": f"✨ {scenario.get('parameter', 'Parameter').replace('_', ' ').title()} Optimization",
                    "parameter": scenario.get('parameter', 'unknown'),
                    "current_value": str(scenario.get('current_value', 'Unknown')),
                    "suggested_value": str(scenario.get('suggested_value', 'Unknown')),
                    "improvement_percentage": float(scenario.get('improvement_percentage', 10)),
                    "reasoning": scenario.get('reasoning', 'No reasoning provided'),
                    "action_plan": scenario.get('action_plan', 'No action plan provided'),
                    "type": "llm_generated",
                    "impact": "Positive",  # Add required field
                    "description": f"Optimize {scenario.get('parameter', 'parameter').replace('_', ' ')}",  # Add description
                    "priority": "medium",
                    "source": "ai"
                })
            return formatted_scenarios
        else:
            print("No valid scenarios parsed from LLM response")
            
    except Exception as e:
        print(f"LLM scenario generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Fallback scenarios if LLM fails
    return get_fallback_llm_scenarios(features, prediction)

def parse_llm_json_response(response: str) -> List[Dict]:
    """
    Robust JSON parser for LLM responses
    """
    if not response:
        return []
    
    # Try to find JSON between triple backticks
    import re
    
    # Pattern for JSON in code blocks
    code_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    matches = re.findall(code_block_pattern, response)
    
    for match in matches:
        try:
            return json.loads(match.strip())
        except:
            continue
    
    # Try to find anything that looks like a JSON array
    array_pattern = r'(\[[\s\S]*\])'
    matches = re.findall(array_pattern, response)
    
    for match in matches:
        try:
            # Clean up common issues
            cleaned = match.strip()
            # Remove trailing commas before closing brackets
            cleaned = re.sub(r',\s*]', ']', cleaned)
            cleaned = re.sub(r',\s*}', '}', cleaned)
            return json.loads(cleaned)
        except:
            continue
    
    # Try direct parsing
    try:
        return json.loads(response)
    except:
        pass
    
    # If all else fails, try to manually extract JSON array
    try:
        start_idx = response.find('[')
        end_idx = response.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response[start_idx:end_idx]
            # Clean up
            json_str = re.sub(r',\s*]', ']', json_str)
            json_str = re.sub(r',\s*}', '}', json_str)
            return json.loads(json_str)
    except:
        pass
    
    return []

def get_fallback_llm_scenarios(features: Dict[str, Any], prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Provide fallback scenarios if LLM fails"""
    return [
        {
            "id": "llm_scenario_fallback_1",
            "title": "✨ Credit Utilization Optimization",
            "parameter": "credit_utilization_ratio",
            "current_value": str(features.get('credit_utilization_ratio', 0.6)),
            "suggested_value": "0.30",
            "improvement_percentage": 12,
            "reasoning": "Lower credit utilization shows you're not overly dependent on credit.",
            "action_plan": "Pay down credit card balances to keep utilization below 30% of your limit.",
            "type": "fallback",
            "impact": "Positive",  # Required field
            "description": "Optimize your credit utilization ratio to improve credit score",  # Required field
            "priority": "medium",
            "source": "ai"
        },
        {
            "id": "llm_scenario_fallback_2",
            "title": "✨ Employment Stability Improvement",
            "parameter": "years_in_current_job",
            "current_value": str(features.get('years_in_current_job', 1)),
            "suggested_value": "3",
            "improvement_percentage": 8,
            "reasoning": "Longer job tenure indicates stable income and lower risk.",
            "action_plan": "Consider waiting until you've completed 3 years in current job before applying.",
            "type": "fallback",
            "impact": "Positive",  # Required field
            "description": "Increase job tenure for better stability assessment",  # Required field
            "priority": "medium",
            "source": "ai"
        }
    ]

def merge_scenarios(logical_scenarios: List[Dict], llm_scenarios: List[Dict]) -> List[Dict]:
    """
    Merge and prioritize scenarios for frontend display
    """
    all_scenarios = []
    
    # Add logical scenarios (high priority)
    for scenario in logical_scenarios:
        scenario['priority'] = 'high'
        scenario['source'] = 'logical'
        all_scenarios.append(scenario)
    
    # Add LLM scenarios
    for scenario in llm_scenarios:
        scenario['priority'] = 'medium'
        scenario['source'] = 'ai'
        all_scenarios.append(scenario)
    
    # Sort by improvement percentage
    all_scenarios.sort(key=lambda x: x.get('improvement_percentage', 0), reverse=True)
    
    return all_scenarios[:10]  # Return top 10 scenarios







async def generate_risk_analysis(features: Dict[str, Any], prediction: Dict[str, Any], request_id: str) -> Dict[str, Any]:
    """
    Generate detailed risk analysis about which profiles are more likely to default
    and what economic factors could impact repayment capability
    """
    
    # Prepare comprehensive features for analysis
    risk_features = {k: v for k, v in features.items() 
                    if k in [
                        "employment_type", "occupation_type", "monthly_income", 
                        "years_in_current_job", "total_work_experience_years",
                        "income_volatility_index", "income_growth_rate_3years",
                        "salary_credit_regularity", "job_change_frequency",
                        "employer_sector_risk_score", "debt_to_income_ratio",
                        "existing_emi_obligations", "credit_score", 
                        "past_defaults_count", "loan_amount_requested",
                        "loan_purpose", "loan_term_months", "interest_rate",
                        "family_dependency_ratio", "cost_of_living_index",
                        "region_economic_risk_score", "region_default_rate_index",
                        "urban_rural_indicator", "local_employment_stability_index",
                        "business_vintage_years", "external_risk_score_1"
                    ]}
    
    prompt = f"""You are a senior credit risk analyst. Provide a detailed analysis of this loan applicant's vulnerability to default, focusing on:

1. **PROFESSION-SPECIFIC RISKS**: How their profession/industry affects repayment capacity
   - For salaried: Job stability, industry health, automation risk
   - For self-employed: Business cycle vulnerability, revenue consistency
   - For gig economy: Income volatility, career longevity

2. **INCOME TRAJECTORY IMPACT**: 
   - If income is declining → how quickly will repayment become difficult?
   - If income is volatile → probability of missed payments during low periods

3. **MACROECONOMIC SENSITIVITY**:
   - How would recession impact this applicant?
   - How would industry downturns affect them?
   - Interest rate hike sensitivity

Current Applicant Profile:
{json.dumps(risk_features, indent=2)}

Current Risk Assessment:
- Risk Level: {prediction.get('risk_level', 'Unknown')}
- Default Probability: {prediction.get('probability', 0):.2%}

Return a JSON with this exact structure:
{{
    "professional_risk": {{
        "category": "salaried/self-employed/gig/other",
        "industry_health": "stable/declining/growing",
        "income_stability_score": "0-100",
        "specific_concerns": ["concern1", "concern2"],
        "future_outlook": "positive/neutral/negative"
    }},
    "income_trajectory_analysis": {{
        "trend": "increasing/stable/declining/volatile",
        "stress_scenarios": [
            {{
                "scenario": "10% income drop",
                "impact": "description",
                "risk_level": "low/medium/high"
            }},
            {{
                "scenario": "20% income drop", 
                "impact": "description",
                "risk_level": "low/medium/high"
            }},
            {{
                "scenario": "Job loss for 3 months",
                "impact": "description",
                "survival_time_months": "X",
                "risk_level": "low/medium/high"
            }}
        ]
    }},
    "macroeconomic_sensitivity": {{
        "recession_impact": "low/medium/high",
        "industry_downturn_sensitivity": "low/medium/high",
        "interest_rate_hike_impact": "low/medium/high",
        "most_vulnerable_to": ["factor1", "factor2"]
    }},
    "early_warning_signs": [
        {{
            "trigger": "what to watch for",
            "threshold": "specific metric",
            "action": "recommended action"
        }}
    ],
    "recommendations": {{
        "risk_mitigation": ["suggestion1", "suggestion2"],
        "monitoring_focus": ["area1", "area2"]
    }}
}}

Return ONLY valid JSON, no other text."""
    
    try:
        print(f"   🔬 Generating detailed risk analysis...")
        response = llm_service(prompt)
        
        # Parse the response
        risk_analysis = parse_llm_json_response(response)
        
        if risk_analysis and isinstance(risk_analysis, dict):
            print(f"   ✅ Risk analysis generated successfully")
            return risk_analysis
        else:
            print(f"   ⚠️ Failed to parse risk analysis, using fallback")
            return get_fallback_risk_analysis(features, prediction)
            
    except Exception as e:
        print(f"   ❌ Risk analysis generation failed: {e}")
        return get_fallback_risk_analysis(features, prediction)

def get_fallback_risk_analysis(features: Dict[str, Any], prediction: Dict[str, Any]) -> Dict[str, Any]:
    """Provide fallback risk analysis if LLM fails"""
    
    employment = features.get('employment_type', 'Unknown')
    income = features.get('monthly_income', 0)
    dti = features.get('debt_to_income_ratio', 0.3)
    
    return {
        "professional_risk": {
            "category": "salaried" if employment == "Salaried" else "self-employed",
            "industry_health": "stable",
            "income_stability_score": 70 if employment == "Salaried" else 50,
            "specific_concerns": ["Income verification needed"] if employment != "Salaried" else ["Single income source"],
            "future_outlook": "neutral"
        },
        "income_trajectory_analysis": {
            "trend": "stable",
            "stress_scenarios": [
                {
                    "scenario": "10% income drop",
                    "impact": f"DTI increases to {dti*1.1:.2f}, still manageable",
                    "risk_level": "low"
                },
                {
                    "scenario": "20% income drop",
                    "impact": f"DTI becomes {dti*1.2:.2f}, entering moderate risk",
                    "risk_level": "medium"
                },
                {
                    "scenario": "Job loss for 3 months",
                    "impact": "Savings would last approximately 2 months",
                    "survival_time_months": "2",
                    "risk_level": "high"
                }
            ]
        },
        "macroeconomic_sensitivity": {
            "recession_impact": "medium",
            "industry_downturn_sensitivity": "medium",
            "interest_rate_hike_impact": "high" if features.get('interest_rate', 0) > 10 else "medium",
            "most_vulnerable_to": ["Income shock", "Interest rate hike"]
        },
        "early_warning_signs": [
            {
                "trigger": "Missed EMI on existing loans",
                "threshold": "1 missed payment",
                "action": "Immediate collection contact"
            }
        ],
        "recommendations": {
            "risk_mitigation": [
                "Consider lower loan amount",
                "Maintain emergency fund of 6 months EMI"
            ],
            "monitoring_focus": [
                "Income stability",
                "Industry news"
            ]
        }
    }








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







@router.get("/applications")
async def get_all_applications(limit: int = 50, skip: int = 0):
    """
    Fetch all loan applications from MongoDB
    - limit: maximum number of records to return (default 50)
    - skip: number of records to skip for pagination (default 0)
    """
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="MongoDB not connected")
        
        # Fetch applications, sort by newest first, exclude MongoDB _id
        applications = list(db.loan_applications.find(
            {}, 
            {"_id": 0}  # Exclude MongoDB internal ID
        ).sort("created_at", -1).skip(skip).limit(limit))
        
        # Get total count for pagination info
        total_count = db.loan_applications.count_documents({})
        
        return {
            "success": True,
            "total_count": total_count,
            "returned_count": len(applications),
            "skip": skip,
            "limit": limit,
            "applications": applications
        }
        
    except Exception as e:
        print(f"❌ Error fetching applications: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch applications: {str(e)}")

@router.get("/applications/{application_id}")
async def get_application_by_id(application_id: str):
    """
    Fetch a specific loan application by its application_id
    """
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="MongoDB not connected")
        
        application = db.loan_applications.find_one(
            {"application_id": application_id},
            {"_id": 0}  # Exclude MongoDB internal ID
        )
        
        if not application:
            raise HTTPException(status_code=404, detail=f"Application with ID {application_id} not found")
        
        return {
            "success": True,
            "application": application
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error fetching application {application_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch application: {str(e)}")
    
    
    
@router.post("/predict-direct")
async def predict_direct(features: Dict[str, Any]):
    """
    Directly pass feature values to the ML model and get prediction
    Accepts a dictionary of feature names and values
    """
    try:
        # Validate that features are provided
        if not features:
            raise HTTPException(status_code=400, detail="No features provided")
        
        print(f"\n{'='*60}")
        print(f"🤖 Direct ML Prediction Request")
        print('='*60)
        print(f"📊 Received {len(features)} features")
        
        # Get the feature list expected by the model
        expected_features = rag_engine.feature_list if hasattr(rag_engine, 'feature_list') else []
        
        if not expected_features:
            raise HTTPException(status_code=500, detail="Model feature list not available")
        
        # Validate and prepare features
        missing_features = []
        prepared_features = {}
        
        for feature in expected_features:
            if feature in features:
                prepared_features[feature] = features[feature]
            else:
                missing_features.append(feature)
                # Set default values based on feature type
                if feature in ["gender", "marital_status", "education_level", "loan_purpose", 
                             "employment_type", "income_verification_status", "property_area",
                             "home_ownership_status", "application_channel", "occupation_type",
                             "loan_type", "property_ownership_status", "urban_rural_indicator"]:
                    prepared_features[feature] = "Unknown"
                elif feature in ["fraud_flag", "past_default_flag", "has_credit_card_flag", 
                               "email_domain_risk_flag"]:
                    prepared_features[feature] = 0
                else:
                    prepared_features[feature] = 0.0
        
        if missing_features:
            print(f"⚠️ Missing {len(missing_features)} features: {missing_features[:10]}...")
            print(f"   Using default values for missing features")
        
        # Set the features in rag_engine temporarily
        original_features = rag_engine.extracted_features
        rag_engine.extracted_features = prepared_features
        
        # Get prediction
        prediction = rag_engine.predict_loan_risk()
        
        # Restore original features
        rag_engine.extracted_features = original_features
        
        print(f"\n📊 Prediction Result:")
        print(f"   • Risk Level: {prediction.get('risk_level', 'Unknown')}")
        print(f"   • Confidence: {prediction.get('confidence', 0):.2%}")
        print(f"   • Features Used: {len(prepared_features)}")
        print(f"   • Missing Features: {len(missing_features)}")
        print('='*60)
        
        return {
            "success": True,
            "prediction": prediction,
            "features_used": len(prepared_features),
            "features_missing": len(missing_features),
            "missing_features_list": missing_features[:20] if missing_features else [],
            "note": "Default values were used for missing features" if missing_features else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Direct prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/predict-batch")
async def predict_batch(features_list: List[Dict[str, Any]]):
    """
    Batch prediction - pass multiple sets of features
    """
    try:
        if not features_list:
            raise HTTPException(status_code=400, detail="No features provided")
        
        print(f"\n{'='*60}")
        print(f"🤖 Batch ML Prediction Request")
        print('='*60)
        print(f"📊 Received {len(features_list)} feature sets")
        
        results = []
        original_features = rag_engine.extracted_features
        
        for idx, features in enumerate(features_list):
            try:
                # Set features temporarily
                rag_engine.extracted_features = features
                
                # Get prediction
                prediction = rag_engine.predict_loan_risk()
                
                results.append({
                    "index": idx,
                    "success": True,
                    "prediction": prediction
                })
                
            except Exception as e:
                results.append({
                    "index": idx,
                    "success": False,
                    "error": str(e)
                })
        
        # Restore original features
        rag_engine.extracted_features = original_features
        
        # Count successes
        successes = sum(1 for r in results if r.get("success"))
        
        return {
            "success": True,
            "total": len(results),
            "successful": successes,
            "failed": len(results) - successes,
            "results": results
        }
        
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")