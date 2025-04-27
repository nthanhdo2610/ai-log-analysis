"""
QDrant Vector Database Integration Module

This module provides a wrapper class for interacting with QDrant vector database,
specifically designed for document storage and similarity search operations.

Key Features:
- Document storage with embeddings
- Similarity search
- Maximal Marginal Relevance (MMR) search
- Batch processing support
"""

from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages.base import BaseMessage
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

import numpy as np

class QDrantDB:
    """
    A wrapper class for QDrant vector database operations.
    
    This class provides an interface for storing documents with their embeddings,
    performing similarity searches, and implementing maximal marginal relevance search.
    
    Attributes:
        collection_name (str): Name of the QDrant collection
        embeddings (Embeddings): Embeddings instance for encoding text
        client (QdrantClient): QDrant client instance
    """

    def __init__(
        self,
        host: str,
        port: Optional[int] = 6333,
        collection_name: str = "support",
        embeddings: Optional[Embeddings] = None,
        vector_size: int = 768,
        prefer_grpc: bool = True,
    ) -> None:
        """
        Initialize QDrant database connection and setup.

        Args:
            host (str): QDrant server host address
            port (Optional[int]): QDrant server port number
            collection_name (str): Name of the collection to use
            embeddings (Optional[Embeddings]): Embeddings instance for text encoding
            vector_size (int): Dimension of the embedding vectors
            prefer_grpc (bool): Whether to use gRPC over HTTP for better performance
        """
        self._validate_init_params(host, port, collection_name, vector_size)
        
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.client = QdrantClient(host=host, port=port, prefer_grpc=prefer_grpc)
        
        self._ensure_collection_exists(vector_size)

    def _validate_init_params(
        self, host: str, port: Optional[int], collection_name: str, vector_size: int
    ) -> None:
        """Validate initialization parameters."""
        if not host:
            raise ValueError("Host cannot be empty")
        if port is not None and not (0 <= port <= 65535):
            raise ValueError("Port must be between 0 and 65535")
        if not collection_name:
            raise ValueError("Collection name cannot be empty")
        if vector_size <= 0:
            raise ValueError("Vector size must be positive")

    def _ensure_collection_exists(self, vector_size: int) -> None:
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]
        
        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def from_documents(
        self,
        documents: List[Document],
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Add documents to QDrant database with their embeddings.

        Args:
            documents (List[Document]): List of documents to add
            batch_size (int): Number of documents to process in each batch
            **kwargs: Additional arguments for future extensibility

        Raises:
            ValueError: If documents list is empty or embeddings are not configured
        """
        if not documents:
            raise ValueError("Documents list cannot be empty")
        if not self.embeddings:
            raise ValueError("Embeddings must be configured")

        self._process_documents_in_batches(documents, batch_size)

    def _process_documents_in_batches(self, documents: List[Document], batch_size: int) -> None:
        """Process and upload documents in batches."""
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            embeddings = self.embeddings.embed_documents(batch)
            points = self._create_points(batch, embeddings, start_id=i)
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

    def _create_points(
        self, documents: List[Document], embeddings: List[List[float]], start_id: int
    ) -> List[PointStruct]:
        """Create QDrant points from documents and their embeddings."""
        return [
            PointStruct(
                id=start_id + idx,
                payload={"page_content": doc.page_content, "metadata": doc.metadata},
                vector=self._prepare_vector(embedding)
            )
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings))
        ]

    @staticmethod
    def _prepare_vector(embedding: Any) -> List[float]:
        """Convert embedding to the correct format."""
        return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform similarity search for the given query.

        Args:
            query (str): Query text to search for
            k (int): Number of results to return
            filter (Optional[Dict[str, Any]]): Filter conditions for search

        Returns:
            List[Document]: List of similar documents
        """
        query_vector = self.embeddings.embed_query(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=filter
        )
        
        return [
            Document(
                page_content=result.payload["page_content"],
                metadata=result.payload["metadata"]
            )
            for result in results
        ]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform Maximal Marginal Relevance (MMR) search.

        MMR aims to maximize relevance while minimizing redundancy in search results.

        Args:
            query (str): Query text to search for
            k (int): Number of results to return
            fetch_k (int): Number of initial results to fetch before MMR
            lambda_mult (float): Trade-off between relevance and diversity (0-1)
            filter (Optional[Dict[str, Any]]): Filter conditions for search

        Returns:
            List[Document]: List of diverse and relevant documents
        """
        query_vector = self.embeddings.embed_query(query)
        results = self._fetch_search_results(query_vector, fetch_k, filter)
        
        documents, vectors = self._extract_documents_and_vectors(results)
        mmr_selected = self._maximal_marginal_relevance(
            query_vector, vectors, lambda_mult=lambda_mult, k=k
        )
        
        return [documents[i] for i in mmr_selected]

    def _fetch_search_results(
        self, query_vector: List[float], fetch_k: int, filter: Optional[Dict[str, Any]]
    ):
        """Fetch initial search results for MMR."""
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=fetch_k,
            query_filter=filter,
            with_vectors=True
        )

    @staticmethod
    def _extract_documents_and_vectors(results):
        """Extract documents and vectors from search results."""
        documents = []
        vectors = []
        for result in results:
            documents.append(
                Document(
                    page_content=result.payload["page_content"],
                    metadata=result.payload["metadata"]
                )
            )
            vectors.append(result.vector)
        return documents, vectors

    @staticmethod
    def _maximal_marginal_relevance(
        query_vector: List[float],
        vectors: List[List[float]],
        lambda_mult: float = 0.5,
        k: int = 4,
    ) -> List[int]:
        """
        Calculate maximal marginal relevance for diversity ranking.

        Args:
            query_vector (List[float]): Query embedding
            vectors (List[List[float]]): Document embeddings
            lambda_mult (float): Trade-off parameter
            k (int): Number of results to return

        Returns:
            List[int]: Indices of selected documents
        """
        query_vector = np.array(query_vector)
        vectors = np.array(vectors)
        
        similarities = np.dot(vectors, query_vector)
        selected_indices = []
        remaining_indices = list(range(len(vectors)))
        
        for _ in range(min(k, len(vectors))):
            if not remaining_indices:
                break
                
            mmr_scores = QDrantDB._calculate_mmr_scores(
                similarities, vectors, remaining_indices, selected_indices, lambda_mult
            )
            
            mmr_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(mmr_idx)
            remaining_indices.remove(mmr_idx)
            
        return selected_indices

    @staticmethod
    def _calculate_mmr_scores(
        similarities: np.ndarray,
        vectors: np.ndarray,
        remaining_indices: List[int],
        selected_indices: List[int],
        lambda_mult: float
    ) -> np.ndarray:
        """Calculate MMR scores for remaining documents."""
        if not selected_indices:
            return similarities[remaining_indices]
        
        selected_vectors = vectors[selected_indices]
        similarities_to_selected = np.max(
            np.dot(vectors[remaining_indices], selected_vectors.T),
            axis=1
        )
        
        return lambda_mult * similarities[remaining_indices] - \
               (1 - lambda_mult) * similarities_to_selected
