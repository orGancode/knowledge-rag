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


class HeadingBasedSplitter(BaseSplitter):
    """基于标题的文本分割器，按一二级标题分块"""
    
    def __init__(self, max_chunk_size: int = 1500, chunk_overlap: int = 100):
        """
        初始化基于标题的文本分割器
        
        Args:
            max_chunk_size: 每个文本块的最大字符数（用于过长的标题块）
            chunk_overlap: 相邻文本块之间的重叠字符数
        """
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
    
    def _detect_headings(self, text: str) -> List[tuple]:
        """
        检测文本中的标题
        
        Args:
            text: 待检测的文本
            
        Returns:
            标题列表，每个元素为 (level, title, position)
        """
        headings = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            # 检测一级标题：数字开头 + 点 + 空格 + 标题
            # 例如: "1. 标题" 或 "第一章 标题"
            if self._is_level1_heading(stripped_line):
                headings.append((1, stripped_line, i))
            
            # 检测二级标题：数字.数字 + 空格 + 标题
            # 例如: "1.1 标题" 或 "1.1. 标题"
            elif self._is_level2_heading(stripped_line):
                headings.append((2, stripped_line, i))
        
        return headings
    
    def _is_level1_heading(self, line: str) -> bool:
        """判断是否为一级标题"""
        import re
        
        # 模式1: "1. 标题" 或 "1.标题"
        if re.match(r'^\d+\.?\s+', line):
            # 排除页码等数字行
            if len(line) > 10:  # 标题通常比页码长
                return True
        
        # 模式2: "第一章 标题" 或 "第1章 标题"
        if re.match(r'^第[一二三四五六七八九十\d]+[章节篇]\s+', line):
            return True
        
        # 模式3: 全大写或首字母大写的短行（可能是标题）
        if len(line) < 50 and line and line[0].isupper():
            # 检查是否包含常见标题关键词
            title_keywords = ['概述', '介绍', '说明', '规定', '制度', '流程', '要求', '标准']
            if any(keyword in line for keyword in title_keywords):
                return True
        
        return False
    
    def _is_level2_heading(self, line: str) -> bool:
        """判断是否为二级标题"""
        import re
        
        # 模式1: "1.1 标题" 或 "1.1. 标题"
        if re.match(r'^\d+\.\d+\.?\s+', line):
            if len(line) > 10:
                return True
        
        # 模式2: "（一）标题" 或 "(1) 标题"
        if re.match(r'^[（\(]\d+[）\)]\s+', line):
            if len(line) > 10:
                return True
        
        return False
    
    def _split_by_headings(self, text: str, headings: List[tuple]) -> List[str]:
        """
        根据标题分割文本
        
        Args:
            text: 待分割的文本
            headings: 标题列表
            
        Returns:
            分割后的文本块列表
        """
        if not headings:
            # 没有检测到标题，使用递归分割
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.max_chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", " ", ""]
            )
            return splitter.split_text(text)
        
        lines = text.split('\n')
        chunks = []
        
        # 添加第一个标题之前的内容（如果有）
        first_heading_pos = headings[0][2]
        if first_heading_pos > 0:
            pre_content = '\n'.join(lines[:first_heading_pos]).strip()
            if pre_content:
                chunks.append(pre_content)
        
        # 按标题分割
        for i in range(len(headings)):
            level, title, pos = headings[i]
            
            # 确定当前块的结束位置
            if i < len(headings) - 1:
                next_pos = headings[i + 1][2]
                chunk_lines = lines[pos:next_pos]
            else:
                chunk_lines = lines[pos:]
            
            chunk_text = '\n'.join(chunk_lines).strip()
            
            # 如果块太长，进一步分割
            if len(chunk_text) > self.max_chunk_size:
                sub_chunks = self._split_long_chunk(chunk_text, title)
                chunks.extend(sub_chunks)
            else:
                chunks.append(chunk_text)
        
        return chunks
    
    def _split_long_chunk(self, chunk: str, title: str) -> List[str]:
        """
        分割过长的文本块
        
        Args:
            chunk: 待分割的文本块
            title: 标题
            
        Returns:
            分割后的文本块列表
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        # 在标题后添加分割标记
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
        )
        
        sub_chunks = splitter.split_text(chunk)
        
        # 为每个子块添加标题信息
        result = []
        for i, sub_chunk in enumerate(sub_chunks):
            if i == 0:
                # 第一个块保留原标题
                result.append(sub_chunk)
            else:
                # 后续块添加标题引用
                result.append(f"[续自: {title}]\n{sub_chunk}")
        
        return result
    
    def split(self, documents: List[Document]) -> List[Document]:
        """
        将文档列表按标题分割成更小的文本块
        
        Args:
            documents: 待分割的文档列表
            
        Returns:
            分割后的文档列表，每个文档包含原始metadata和chunk索引信息
        """
        if not documents:
            return []
        
        result = []
        
        for doc_idx, doc in enumerate(documents):
            # 检测标题
            headings = self._detect_headings(doc.page_content)
            
            # 按标题分割
            chunks = self._split_by_headings(doc.page_content, headings)
            
            # 为每个chunk创建新的Document对象
            for chunk_idx, chunk in enumerate(chunks):
                # 复制原始metadata
                metadata = doc.metadata.copy() if doc.metadata else {}
                
                # 添加chunk相关信息
                metadata.update({
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                    "total_chunks": len(chunks),
                    "splitter_type": "heading_based",
                    "headings_count": len(headings)
                })
                
                # 创建新的Document对象
                chunk_doc = Document(page_content=chunk, metadata=metadata)
                result.append(chunk_doc)
        
        return result
