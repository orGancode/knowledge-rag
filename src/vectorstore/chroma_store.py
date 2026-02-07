from typing import List, Dict, Any, Optional, Callable
import uuid
import chromadb
from chromadb.config import Settings
from chromadb.api.types import EmbeddingFunction
from langchain.schema import Document


class LangChainEmbeddingFunction(EmbeddingFunction):
    """
    适配器类，将LangChain风格的嵌入函数转换为ChromaDB兼容的嵌入函数
    """
    
    def __init__(self, embed_function: Callable[[str], List[float]]):
        self.embed_function = embed_function
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB要求的接口
        
        Args:
            input: 文本列表
            
        Returns:
            嵌入向量列表
        """
        return [self.embed_function(text) for text in input]


class ChromaVectorStore:
    def __init__(self, collection_name: str, persist_directory: str):
        """
        初始化Chroma向量存储
        
        Args:
            collection_name: 集合名称
            persist_directory: 持久化目录路径
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self.embedding_function = None
    
    def create_collection(self, embedding_function: Callable[[str], List[float]]):
        """
        初始化Chroma客户端和集合
        
        Args:
            embedding_function: 嵌入函数，用于将文本转换为向量
        """
        # 保存原始嵌入函数
        self.embedding_function = embedding_function
        
        # 创建ChromaDB兼容的嵌入函数
        chroma_embedding_function = LangChainEmbeddingFunction(embedding_function)
        
        # 初始化Chroma客户端
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        
        # 获取或创建集合
        try:
            # 尝试获取现有集合
            self.collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=chroma_embedding_function
            )
            print(f"已加载现有集合: {self.collection_name}")
        except Exception:
            # 如果集合不存在，创建新集合
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=chroma_embedding_function,
                metadata={"hnsw:space": "cosine"}  # 使用余弦相似度
            )
            print(f"已创建新集合: {self.collection_name}")
    
    def add_documents(self, documents: List[Document], embeddings: List[List[float]]):
        """
        添加文档和对应向量到集合中
        
        Args:
            documents: 文档列表
            embeddings: 对应的向量列表
        """
        if not self.collection:
            raise ValueError("请先调用create_collection初始化集合")
        
        if len(documents) != len(embeddings):
            raise ValueError("文档数量与向量数量不匹配")
        
        # 准备数据
        ids = []
        texts = []
        metadatas = []
        
        for i, doc in enumerate(documents):
            # 生成唯一ID
            doc_id = f"chunk_{uuid.uuid4().hex[:8]}"
            ids.append(doc_id)
            
            # 文档内容
            texts.append(doc.page_content)
            
            # 元数据，确保包含chunk_index
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata["chunk_index"] = i
            metadatas.append(metadata)
        
        # 添加到Chroma集合
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"已添加 {len(documents)} 个文档到集合 {self.collection_name}")
    
    def similarity_search(self, query_embedding: List[float], k: int = 4) -> List[Dict[str, Any]]:
        """
        基于查询向量进行相似度搜索
        
        Args:
            query_embedding: 查询向量
            k: 返回的最相似文档数量
            
        Returns:
            包含文档内容和相似度分数的结果列表
        """
        if not self.collection:
            raise ValueError("请先调用create_collection初始化集合")
        
        # 执行相似度搜索
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results["ids"][0])):
            result = {
                "id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                # 将距离转换为相似度分数（0-1之间，越接近1越相似）
                "similarity_score": 1 - results["distances"][0][i]
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def persist(self):
        """
        显式持久化到磁盘
        注意：Chroma的PersistentClient会自动持久化，此方法主要用于确保数据已保存
        """
        if not self.client:
            raise ValueError("请先调用create_collection初始化集合")
        
        # Chroma的PersistentClient会自动持久化，这里主要用于确认
        print(f"数据已持久化到: {self.persist_directory}")
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        获取集合信息
        
        Returns:
            包含集合统计信息的字典
        """
        if not self.collection:
            raise ValueError("请先调用create_collection初始化集合")
        
        count = self.collection.count()
        return {
            "name": self.collection_name,
            "count": count,
            "persist_directory": self.persist_directory
        }
    
    def delete_collection(self):
        """
        删除当前集合
        """
        if not self.client:
            raise ValueError("请先调用create_collection初始化集合")
        
        self.client.delete_collection(name=self.collection_name)
        print(f"已删除集合: {self.collection_name}")
        self.collection = None