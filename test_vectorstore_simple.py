#!/usr/bin/env python3
"""
简化版ChromaVectorStore测试
"""

import os
import sys
import shutil

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchain.schema import Document
from src.vectorstore import ChromaVectorStore


def test_simple():
    """简化版测试，使用预定义的向量"""
    
    # 1. 创建测试文档
    print("创建测试文档...")
    documents = [
        Document(
            page_content="人工智能是计算机科学的一个分支。",
            metadata={"source": "test.txt", "page": 1}
        ),
        Document(
            page_content="机器学习是人工智能的一个子集。",
            metadata={"source": "test.txt", "page": 2}
        ),
        Document(
            page_content="深度学习使用神经网络来模拟人脑。",
            metadata={"source": "test.txt", "page": 3}
        )
    ]
    
    # 2. 创建模拟的嵌入向量（3维向量）
    print("创建模拟嵌入向量...")
    embeddings = [
        [0.1, 0.2, 0.3],  # 对应第一个文档
        [0.2, 0.3, 0.4],  # 对应第二个文档
        [0.3, 0.4, 0.5]   # 对应第三个文档
    ]
    
    # 3. 创建简单的嵌入函数
    def simple_embedding_function(text: str):
        """简单的嵌入函数，返回固定向量"""
        # 在实际应用中，这里会调用真实的嵌入模型
        return [0.25, 0.35, 0.45]  # 模拟查询向量
    
    # 4. 初始化ChromaVectorStore
    print("初始化ChromaVectorStore...")
    persist_directory = "./test_simple_chroma_db"
    collection_name = "simple_test"
    
    # 清理之前的测试数据
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # 5. 创建集合
    print("创建集合...")
    vector_store.create_collection(embedding_function=simple_embedding_function)
    
    # 6. 添加文档
    print("添加文档...")
    vector_store.add_documents(documents, embeddings)
    
    # 7. 获取集合信息
    print("\n集合信息:")
    info = vector_store.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 8. 执行相似度搜索
    print("\n执行相似度搜索...")
    query_embedding = [0.25, 0.35, 0.45]  # 模拟查询向量
    results = vector_store.similarity_search(query_embedding, k=2)
    
    print("搜索结果:")
    for i, result in enumerate(results):
        print(f"\n结果 {i+1}:")
        print(f"  文档ID: {result['id']}")
        print(f"  内容: {result['document']}")
        print(f"  元数据: {result['metadata']}")
        print(f"  相似度分数: {result['similarity_score']:.4f}")
    
    # 9. 持久化
    print("\n持久化数据...")
    vector_store.persist()
    
    print("\n✅ 简化版测试完成！")


if __name__ == "__main__":
    test_simple()