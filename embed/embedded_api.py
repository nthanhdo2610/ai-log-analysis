from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
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
    data = await request.json()
    texts = data["texts"]
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