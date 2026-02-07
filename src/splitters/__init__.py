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
        """
        初始化递归文本分割器
        
        Args:
            chunk_size: 每个文本块的最大字符数
            chunk_overlap: 相邻文本块之间的重叠字符数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        from langchain.text_splitter import RecursiveCharacterTextSplitter as LangChainRecursiveTextSplitter
        self._splitter = LangChainRecursiveTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表分割成更小的文本块
        
        Args:
            documents: 待分割的文档列表
            
        Returns:
            分割后的文档列表，每个文档包含原始metadata和chunk索引信息
        """
        if not documents:
            return []
        
        result = []
        
        for doc_idx, doc in enumerate(documents):
            # 使用LangChain的分割器分割文档内容
            chunks = self._splitter.split_text(doc.page_content)
            
            # 为每个chunk创建新的Document对象
            for chunk_idx, chunk in enumerate(chunks):
                # 复制原始metadata
                metadata = doc.metadata.copy() if doc.metadata else {}
                
                # 添加chunk相关信息
                metadata.update({
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                    "total_chunks": len(chunks)
                })
                
                # 创建新的Document对象
                chunk_doc = Document(page_content=chunk, metadata=metadata)
                result.append(chunk_doc)
        
        return result


class TokenSplitter(BaseSplitter):
    """Token-based text splitter."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化基于Token的文本分割器
        
        Args:
            chunk_size: 每个文本块的最大token数
            chunk_overlap: 相邻文本块之间的重叠token数
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        from langchain.text_splitter import TokenTextSplitter as LangChainTokenTextSplitter
        self._splitter = LangChainTokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
    
    def split(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表按token分割成更小的文本块
        
        Args:
            documents: 待分割的文档列表
            
        Returns:
            分割后的文档列表，每个文档包含原始metadata和chunk索引信息
        """
        if not documents:
            return []
        
        result = []
        
        for doc_idx, doc in enumerate(documents):
            # 使用LangChain的Token分割器分割文档内容
            chunks = self._splitter.split_text(doc.page_content)
            
            # 为每个chunk创建新的Document对象
            for chunk_idx, chunk in enumerate(chunks):
                # 复制原始metadata
                metadata = doc.metadata.copy() if doc.metadata else {}
                
                # 添加chunk相关信息
                metadata.update({
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                    "total_chunks": len(chunks),
                    "splitter_type": "token"
                })
                
                # 创建新的Document对象
                chunk_doc = Document(page_content=chunk, metadata=metadata)
                result.append(chunk_doc)
        
        return result
