import requests
from langchain_core.documents import Document

class ModelEmbeddings:
    def __init__(self, model: str):
        self.api_url = model  # model is actually the embedding API URL

    def embed_documents(self, docs: list[Document]) -> list[list[float]]:
        texts = [doc.page_content for doc in docs]
        return self._post_embedding(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._post_embedding([query])[0]

    def _post_embedding(self, texts: list[str]) -> list[list[float]]:
        try:
            response = requests.post(self.api_url, json={"texts": texts}, timeout=30)
            response.raise_for_status()
            return response.json()["vectors"]
        except requests.RequestException as e:
            print(f"❌ Embedding API request failed: {e}")
            raise
