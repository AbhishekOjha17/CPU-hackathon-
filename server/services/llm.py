# from dotenv import load_dotenv


# import os, uuid, requests, logging, bcrypt
# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# from fastapi import HTTPException 

# # LLM Configuration
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")


# def llm_service(prompt: str) -> str:
#     """Call LLM via OpenRouter API."""
#     if not OPENROUTER_API_KEY:
#         raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

#     try:
#         response = requests.post(
#             "https://openrouter.ai/api/v1/chat/completions", 
#             headers={
#                 "Authorization": f"Bearer {OPENROUTER_API_KEY}",
#                 "Content-Type": "application/json",
#                 "HTTP-Referer": "http://localhost:3000",
#                 "X-Title": "GraphRAG Study Platform"
#             },
#             json={
#                 "model": OPENROUTER_MODEL,
#                 "messages": [
#                     {"role": "system", "content": "You are a helpful study assistant."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 "temperature": 0.2,
#                 "max_tokens": 1000
#             },
#             timeout=30
#         )

#         response.raise_for_status()
#         data = response.json()

#         if "choices" in data and len(data["choices"]) > 0:
#             return data["choices"][0]["message"]["content"]
#         else:
#             raise HTTPException(status_code=500, detail="Invalid response from LLM")

#     except requests.exceptions.RequestException as e:
#         logger.error(f"LLM API error: {e}")
#         raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")





from dotenv import load_dotenv
import os, uuid, requests, logging, bcrypt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import HTTPException

# LLM Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")

def llm_service(prompt: str) -> str:
    """Call LLM via OpenRouter API for loan document analysis."""
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OpenRouter API key not configured")

    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions", 
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Loan Risk Assessment Platform"
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a financial document analyzer specializing in loan applications. Extract structured information from loan documents, bank statements, income proofs, and identity documents. Return data in valid JSON format only."
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,  # Lower temperature for more consistent extraction
                "max_tokens": 2000    # Increased for longer document processing
            },
            timeout=45  # Increased timeout for longer documents
        )

        response.raise_for_status()
        data = response.json()

        if "choices" in data and len(data["choices"]) > 0:
            return data["choices"][0]["message"]["content"]
        else:
            raise HTTPException(status_code=500, detail="Invalid response from LLM")

    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM service error: {str(e)}")