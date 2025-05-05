from fastapi import FastAPI, HTTPException, Header
from fastapi.concurrency import asynccontextmanager
from pydantic import BaseModel
from chatbot import ELKChatbot
from utils.config_manager import ConfigManager

# Initialize FastAPI app
app = FastAPI(title="ELK Log Analysis Chatbot API", version="1.0")
API_KEY = "Nx4bNayVyyXL6VR1YTkmaXTXKe75yilUxraXvSMq9l1unDIZoDnrFWYFte30PgNB"

# Configuration
config = ConfigManager("configs.json")

# Load configs
es_config = config.get_json("es_config")
qdrant_db = config.get_json("qdrant_db")
langfuse_keys = config.get_json("langfuse_keys")
claude_config = config.get_json("claude_config")

chatbot = ELKChatbot(
    es_config=es_config,
    embedding_model=config.get("embedding_model"),
    qdrant_db=qdrant_db,
    claude_config=claude_config,
    langfuse_keys=langfuse_keys
)

# Define a request model
class QueryRequest(BaseModel):
    question: str


@app.post("/chat")
def analyze_logs(request: QueryRequest, x_api_key: str = Header(...)):
    """
    Endpoint to query the chatbot for log analysis.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")

    try:
        response = chatbot.process_query(request.question)
        return {"query": request.question, "response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/about")
def root():
    return {"message": "ELK Log Analysis Chatbot API is running!"}

# ✅ Run the API server
if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="debug")
