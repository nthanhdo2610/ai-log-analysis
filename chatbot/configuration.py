import base64
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

import json
from elasticsearch import Elasticsearch

class ELKLogRetriever:
    def __init__(self, es_host: str, es_user: str, es_pass: str, index: str):
        try:
            print(f"🔧 [DEBUG] Connecting to ES @ {es_host}")
            print(f"🔧 [DEBUG] Using index/data stream: '{index}'")
            self.es_user = es_user
            self.es_pass = es_pass
            self.es = Elasticsearch(
                es_host,
                basic_auth=(es_user, es_pass),
                verify_certs=False
            )
            self.index = index.strip()
            self.mapping = json.dumps(self.get_mapping(), indent=2)
            print("📌 [DEBUG] Loaded mapping from ES:")
            print(self.mapping)
        except Exception as e:
            print(f"❌ Unable to connect to Elasticsearch at {es_host}: {e}")
            self.es = None

    def get_mapping(self):
        """Fetch mapping for a regular index or the latest backing index of a data stream."""
        if not self.es:
            return {}

        # Prepare Basic Auth header
        encoded = base64.b64encode(f"{self.es_user}:{self.es_pass}".encode()).decode()
        auth_header = f"Basic {encoded}"
        headers = {
            "Accept": "application/json",
            "Authorization": auth_header
        }

        try:
            # Step 1: Check if it's a data stream
            ds_response = self.es.transport.perform_request(
                method="GET",
                target=f"/_data_stream/{self.index}",
                headers=headers
            ).body

            data_streams = ds_response.get("data_streams", [])
            if data_streams:
                backing_indices = data_streams[0].get("indices", [])
                if backing_indices:
                    # Get latest backing index
                    latest_index = sorted(backing_indices, key=lambda x: x["index_name"])[-1]["index_name"]
                    print(f"📌 [DEBUG] Resolved backing index for data stream: {latest_index}")

                    # Fetch mapping for backing index
                    mapping_response = self.es.transport.perform_request(
                        method="GET",
                        target=f"/{latest_index}/_mapping",
                        headers=headers
                    ).body
                    return mapping_response.get(latest_index, {}).get("mappings", {})
                else:
                    print(f"⚠️ No backing indices for data stream: {self.index}")
                    return {}
            else:
                raise Exception("Not a data stream")

        except Exception as e:
            # Fallback: assume it's a normal index
            print(f"ℹ️ Fallback to regular index mapping for '{self.index}': {e}")
            try:
                mapping_response = self.es.transport.perform_request(
                    method="GET",
                    target=f"/{self.index}/_mapping",
                    headers=headers
                ).body
                return mapping_response.get(self.index, {}).get("mappings", {})
            except Exception as e2:
                print(f"❌ Failed to fetch mapping for '{self.index}': {e2}")
                return {}


    def search_logs(self, query_body: dict):
        """Search logs using raw transport with manual authentication and full debug tracing."""
        if not self.es:
            print("❌ Elasticsearch client not initialized.")
            return []

        try:
            # Build Basic Auth header manually
            encoded = base64.b64encode(f"{self.es_user}:{self.es_pass}".encode()).decode()
            auth_header = f"Basic {encoded}"

            print("🔍 [TRACE] Sending Elasticsearch Query:")
            print(json.dumps(query_body, indent=2))

            response = self.es.transport.perform_request(
                method="POST",
                target=f"/{self.index}/_search",
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "Authorization": auth_header  # ✅ required
                },
                body=query_body
            ).body

            print("✅ [TRACE] Elasticsearch Response (truncated to 1000 chars):")
            raw_json = json.dumps(response, indent=2)
            print(raw_json[:1000] + ("..." if len(raw_json) > 1000 else ""))

            return [hit["_source"] for hit in response.get("hits", {}).get("hits", [])]

        except Exception as e:
            print("❌ [ERROR] Elasticsearch query failed.")
            print(f"❌ Exception: {e}")
            return []

class LogEmbeddingProcessor:
    def __init__(self, embedding_model: str):  # embedding_model = API URL
        self.embedder = ModelEmbeddings(model=embedding_model)

    def embed_logs(self, logs):
        documents = [Document(page_content=log.get("message", "")) for log in logs]
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