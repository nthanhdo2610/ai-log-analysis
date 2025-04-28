import json
import uuid
from elasticsearch import Elasticsearch
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Distance, VectorParams
from models.langfuse import LangfusePromptManager
from models.embeddings import ModelEmbeddings
from models.anthropic import ChatAnthropic
from langchain_core.documents import Document

from elasticsearch import Elasticsearch, ConnectionError

class ELKLogRetriever:
    def __init__(self, es_host: str, es_user: str, es_pass: str, index: str):
        try:
            self.es = Elasticsearch(
                es_host,
                basic_auth=(es_user, es_pass),
                verify_certs=False
            )
            self.index = index
            self.mapping = json.dumps(self.get_mapping(), indent=2)  # Fetch mapping dynamically
        except Exception as e:
            print(f"❌ Unable to connect to Elasticsearch at {es_host}: {e}")
            self.es = None

    def get_mapping(self):
        """Retrieve index mapping for a data stream by checking its backing indices."""
        if not self.es:
            return {}

        try:
            # Get backing indices for the data stream
            data_stream_info = self.es.indices.get_data_stream(name=self.index)

            backing_indices = data_stream_info.get("data_streams", [])[0].get("indices", [])
            if not backing_indices:
                print(f"⚠️ No backing indices found for data stream: {self.index}")
                return {}

            # Get the latest backing index
            latest_index = backing_indices[-1]["index_name"]

            # Fetch mapping from the latest backing index
            mapping = self.es.indices.get_mapping(index=latest_index)
            return mapping.get(latest_index, {}).get("mappings", {})

        except Exception as e:
            print(f"❌ Error fetching index mapping for data stream {self.index}: {e}")
            return {}

    def search_logs(self, query: str):
        es_query = json.dumps(query, indent=2)
        print(f"🔎 Elasticsearch querying: {es_query}")
        """Search logs in a data stream with a flexible query."""
        if not self.es:
            return []
        # Ensure headers are set correctly
        headers = {
            "Content-Type": "application/json"
        }
        try:
            response = self.es.search(index=self.index, body={
                "query": {
                    "match": {"message": es_query}
                },
                "size": 100  # Fetch up to 100 logs
            })
            return [hit["_source"] for hit in response.get("hits", {}).get("hits", [])]
        except Exception as e:
            print(f"❌ Error querying Elasticsearch - error: {e}")
            return []



class LogEmbeddingProcessor:
    def __init__(self, embedding_model: str):
        self.embedder = ModelEmbeddings(model=embedding_model)

    def embed_logs(self, logs):
        documents = [Document(page_content=log["message"]) for log in logs]
        return self.embedder.embed_documents(documents)

    def embed_query(self, query: str):
        return self.embedder.embed_query(query)

class QDrantLogStore:
    
    def __init__(self, host: str, port: int, collection_name: str, api_key: str, prefer_grpc: bool, https:bool=False, vector_size: int = 768):
        self.client = QdrantClient(host=host, port=port,api_key=api_key, https=https, prefer_grpc=prefer_grpc)
        self.collection_name = collection_name
        self._ensure_collection_exists(vector_size)
    
    def __init__(self, host: str, port: int, collection_name: str, prefer_grpc: bool, https:bool=False, vector_size: int = 768):
        self.client = QdrantClient(host=host, port=port, https=https, prefer_grpc=prefer_grpc)
        self.collection_name = collection_name
        self._ensure_collection_exists(vector_size)

    def _ensure_collection_exists(self, vector_size: int):
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def store_logs(self, logs, embeddings):
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={"timestamp": log.get("timestamp", "unknown"), "user": log.get("user", "unknown"), "message": log.get("message", "")}
            )
            for i, log in enumerate(logs)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search_similar_logs(self, query_embedding, k=5):
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k
        )
        if not results:
            return [{"message": "No similar logs found.", "timestamp": None, "user": None}]
        return [
            {
                "timestamp": hit.payload.get("timestamp", "unknown"),
                "user": hit.payload.get("user", "unknown"),
                "message": hit.payload.get("message", "No message available")
            }
            for hit in results
        ]

class ClaudeAnalyzer:
    def __init__(self, claude_model: str, claude_api_key: str):
        self.claude = ChatAnthropic(model=claude_model, api_key=claude_api_key)

    def analyze_logs(self, logs):
        prompt = f"""
        Analyze the following logs and provide:
        - A summary of recurring issues
        - Predicted potential failures
        - Suggested fixes
        Logs: {logs}
        """
        response = self.claude.invoke(prompt)
        return response

class LangfuseLogger:
    def __init__(self, secret_key: str, public_key: str, host: str):
        self.langfuse = LangfusePromptManager(secret_key=secret_key, public_key=public_key, host=host)

    def log_interaction(self, query, response):
        self.langfuse.add(
            prompt=query,
            name="elk_chatbot",
            config={"response": response},
            labels=["log_analysis"]
        )