# Embeddings Module

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddings(ABC):
    """Base class for embeddings."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text."""
        pass


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings implementation."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI."""
        # TODO: Implement OpenAI embeddings
        pass
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using OpenAI."""
        # TODO: Implement OpenAI query embedding
        pass


class HuggingFaceEmbeddings(BaseEmbeddings):
    """HuggingFace embeddings implementation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using HuggingFace."""
        # TODO: Implement HuggingFace embeddings
        pass
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using HuggingFace."""
        # TODO: Implement HuggingFace query embedding
        pass
