# Document Loaders Module

from abc import ABC, abstractmethod
from typing import List
from langchain.schema import Document
from .pdf_loader import PDFLoader


class BaseLoader(ABC):
    """Base class for document loaders."""
    
    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """Load documents from file path."""
        pass


class TextLoader(BaseLoader):
    """Text file loader."""
    
    def load(self, file_path: str) -> List[Document]:
        """Load text file from file path."""
        # TODO: Implement text loading logic
        pass
