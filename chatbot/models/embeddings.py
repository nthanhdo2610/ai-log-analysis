import json
import os
from typing import Any, Dict, List, Optional
import asyncio
from langchain_core.documents import Document

from langchain_core.embeddings import Embeddings
from pydantic.v1 import BaseModel, Extra, root_validator

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VALID_TASKS = ("feature-extraction",)


class ModelEmbeddings(BaseModel, Embeddings):
    

    client: Any  #: :meta private:
    async_client: Any  #: :meta private:
    model: Optional[str] = None
    """Model name to use."""
    repo_id: Optional[str] = None
    """Huggingfacehub repository id, for backward compatibility."""
    task: Optional[str] = "feature-extraction"
    """Task to call the model with."""
    model_kwargs: Optional[dict] = None
    """Keyword arguments to pass to the model."""
    batch_size: int = 8  # Reduced batch size for better handling
    """Batch size for processing."""
    max_retries: int = 3
    """Maximum number of retries for failed requests."""
    retry_delay: float = 1.0
    """Delay between retries in seconds."""
    debug: bool = False  # Changed to False since we're removing logging
    max_text_length: int = 2048  # Maximum length for each text
    """Maximum length for each text to prevent payload size issues."""

    huggingfacehub_api_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        huggingfacehub_api_token = values["huggingfacehub_api_token"] or os.getenv(
            "HUGGINGFACEHUB_API_TOKEN"
        )

        try:
            from huggingface_hub import AsyncInferenceClient, InferenceClient

            if values["model"]:
                values["repo_id"] = values["model"]
            elif values["repo_id"]:
                values["model"] = values["repo_id"]
            else:
                values["model"] = DEFAULT_MODEL
                values["repo_id"] = DEFAULT_MODEL

            client = InferenceClient(
                model=values["model"],
                token=huggingfacehub_api_token,
            )

            async_client = AsyncInferenceClient(
                model=values["model"],
                token=huggingfacehub_api_token,
            )

            if values["task"] not in VALID_TASKS:
                raise ValueError(
                    f"Got invalid task {values['task']}, "
                    f"currently only {VALID_TASKS} are supported"
                )
            values["client"] = client
            values["async_client"] = async_client

        except ImportError:
            raise ImportError(
                "Could not import huggingface_hub python package. "
                "Please install it with `pip install huggingface_hub`."
            )
        return values

    def _truncate_text(self, text: str) -> str:
        """Truncate text to prevent payload size issues."""
        if len(text) > self.max_text_length:
            return text[:self.max_text_length]
        return text

    def _estimate_batch_size(self, texts: List[str]) -> int:
        """Estimate appropriate batch size based on text lengths."""
        avg_length = sum(len(text) for text in texts) / len(texts)
        if avg_length > 1000:
            return min(4, self.batch_size)
        elif avg_length > 500:
            return min(8, self.batch_size)
        return self.batch_size

    async def _async_embed_batch(self, batch: List[str], retry_count: int = 0) -> List[List[float]]:
        """Embed a batch of texts with retry logic."""
        truncated_batch = [self._truncate_text(text) for text in batch]
        
        try:
            result = await self.async_client.feature_extraction(truncated_batch)
            return result
        except Exception as e:
            if "413" in str(e) and retry_count == 0:
                if len(batch) > 1:
                    mid = len(batch) // 2
                    first_half = await self._async_embed_batch(batch[:mid])
                    second_half = await self._async_embed_batch(batch[mid:])
                    return first_half + second_half
                else:
                    truncated = self._truncate_text(batch[0])[:self.max_text_length//2]
                    return await self._async_embed_batch([truncated])
            
            if retry_count < self.max_retries:
                retry_time = self.retry_delay * (retry_count + 1)
                await asyncio.sleep(retry_time)
                return await self._async_embed_batch(batch, retry_count + 1)
            raise e

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Embed documents using async batching."""
        texts = [doc.page_content for doc in documents]
        self.batch_size = self._estimate_batch_size(texts)
        
        async def process_all_batches():
            tasks = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                tasks.append(self._async_embed_batch(batch))
            
            embeddings = []
            for batch_embeddings in await asyncio.gather(*tasks):
                embeddings.extend(batch_embeddings)
            return embeddings

        try:
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            pass

        try:
            return asyncio.run(process_all_batches())
        except Exception as e:
            raise

    def rerank(self, query: str, texts: List[str]) -> List[List[float]]:
        """Rerank documents using the model."""
        # texts schema: {"id": i, "text": doc.page_content, "meta": doc.metadata}
        text = [i["text"] for i in texts]
        
        try:
            responses = self.client.post(
                json={
                    "query": query,
                    "texts": text,
                }
            )
            responses = json.loads(responses.decode())

            # back the schema into json
            score_dict = {item['index']: item['score'] for item in responses}

            for item in texts:
                item_id = item['id']
                if item_id in score_dict:
                    item['score'] = score_dict[item_id]
            return sorted(texts, key=lambda x: x['score'], reverse=True)
        except Exception as e:
            print(f"Error in rerank: {e}")
            return texts  # Return original order if reranking fails

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed documents with batching and retry logic."""
        texts = [text.replace("\n", " ") for text in texts]
        
        tasks = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            tasks.append(self._async_embed_batch(batch))
        
        all_embeddings = []
        for batch_embeddings in await asyncio.gather(*tasks):
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text."""
        doc = Document(page_content=text, metadata={})
        return self.embed_documents([doc])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query text."""
        response = (await self.aembed_documents([text]))[0]
        return response