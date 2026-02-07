#!/usr/bin/env python3
"""
测试ChromaVectorStore的功能
"""

import os
import sys
import shutil
from typing import List

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchain.schema import Document
from src.vectorstore import ChromaVectorStore
from src.embeddings import HuggingFaceEmbeddings


def test_chroma_store():
    """测试ChromaVectorStore的基本功能"""
    
    # 1. 初始化嵌入模型
    print("初始化嵌入模型...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # 2. 创建测试文档
    print("创建测试文档...")
    documents = [
        Document(
            page_content="人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            metadata={"source": "test.txt", "page": 1}
        ),
        Document(
            page_content="机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
            metadata={"source": "test.txt", "page": 2}
        ),
        Document(
            page_content="深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。",
            metadata={"source": "test.txt", "page": 3}
        )
    ]
    
    # 3. 生成嵌入向量
    print("生成嵌入向量...")
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.embed_documents(texts)
    
    # 4. 初始化ChromaVectorStore
    print("初始化ChromaVectorStore...")
    persist_directory = "./test_chroma_db"
    collection_name = "test_collection"
    
    # 清理之前的测试数据
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
    
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # 5. 创建集合
    print("创建集合...")
    vector_store.create_collection(embedding_function=embedding_model.embed_query)
    
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
    query_text = "什么是深度学习？"
    query_embedding = embedding_model.embed_query(query_text)
    
    results = vector_store.similarity_search(query_embedding, k=2)
    
    print(f"查询: {query_text}")
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
    
    # 10. 测试重新加载
    print("\n测试重新加载...")
    vector_store2 = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    vector_store2.create_collection(embedding_function=embedding_model.embed_query)
    
    info2 = vector_store2.get_collection_info()
    print(f"重新加载后的集合信息: {info2}")
    
    # 11. 再次搜索验证数据完整性
    results2 = vector_store2.similarity_search(query_embedding, k=2)
    print(f"重新加载后搜索结果数量: {len(results2)}")
    
    print("\n测试完成！")


if __name__ == "__main__":
    test_chroma_store()