# Text Splitters Module

from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document


class BaseSplitter(ABC):
    """Base class for text splitters."""
    
    @abstractmethod
    def split(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        pass


class RecursiveTextSplitter(BaseSplitter):
    """Recursive character text splitter."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, documents: List[Document]) -> List[Document]:
        """Split documents recursively."""
        # TODO: Implement recursive splitting logic
        pass


class TokenSplitter(BaseSplitter):
    """Token-based text splitter."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split(self, documents: List[Document]) -> List[Document]:
        """Split documents by tokens."""
        # TODO: Implement token-based splitting logic
        pass
