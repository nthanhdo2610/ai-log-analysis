from contextlib import asynccontextmanager
import json
from fastapi import FastAPI, HTTPException, Request
from sentence_transformers import SentenceTransformer

# Register event handlers
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    print("🚀 Embedded FastAPI server is starting...")
    
    yield  # ⬅️ Let the app run here

    # Shutdown actions
    print("🛑 Embedded FastAPI server is shutting down...")


# Initialize FastAPI app
app = FastAPI(title="Embedded Log Analysis", version="1.0", lifespan=lifespan)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.post("/embed")
async def embed(request: Request):
    try:
        data = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {e}")
                            
    print("📥 [DEBUG] Incoming /embed request body (JSON):")
    print(json.dumps(data, indent=2))  # Pretty-printed JSON)

    # Accept both "texts" and "inputs"
    texts = data.get("texts") or data.get("inputs")
    if not texts:
        raise HTTPException(status_code=400, detail="Missing 'texts' or 'inputs' in request body")

    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise HTTPException(status_code=422, detail="'texts' or 'inputs' must be a list of strings.")

    vectors = model.encode(texts, convert_to_numpy=True).tolist()
    return {"vectors": vectors}

@app.get("/")
def root():
    return {"message": "Embedded Log Analysis API is running!"}

# ✅ Run the API server
if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")