#!/usr/bin/env python3
"""
示例：使用ChromaVectorStore存储和检索PDF文档
"""

import os
import sys
import shutil

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.loaders import PDFLoader
from src.splitters import RecursiveTextSplitter
from src.embeddings import HuggingFaceEmbeddings
from src.vectorstore import ChromaVectorStore


def main():
    """主函数：演示完整的RAG流程中的向量存储部分"""
    
    # 1. 初始化嵌入模型
    print("步骤1: 初始化嵌入模型...")
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # 2. 加载PDF文档
    print("\n步骤2: 加载PDF文档...")
    pdf_path = "data/员工手册.pdf"
    if not os.path.exists(pdf_path):
        print(f"错误: 找不到文件 {pdf_path}")
        return
    
    loader = PDFLoader(pdf_path)
    documents = loader.load()
    print(f"已加载 {len(documents)} 页文档")
    
    # 3. 分割文档
    print("\n步骤3: 分割文档...")
    text_splitter = RecursiveTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    
    chunks = text_splitter.split(documents)
    print(f"已分割为 {len(chunks)} 个文本块")
    
    # 4. 生成嵌入向量
    print("\n步骤4: 生成嵌入向量...")
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.embed_documents(texts)
    print(f"已生成 {len(embeddings)} 个向量")
    
    # 5. 初始化向量存储
    print("\n步骤5: 初始化向量存储...")
    persist_directory = "./chroma_db"
    collection_name = "pdf_documents"
    
    # 清理之前的测试数据（可选）
    if os.path.exists(persist_directory):
        response = input(f"目录 {persist_directory} 已存在，是否删除？(y/n): ")
        if response.lower() == 'y':
            shutil.rmtree(persist_directory)
            print("已删除旧数据")
    
    vector_store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    
    # 6. 创建集合
    print("\n步骤6: 创建集合...")
    vector_store.create_collection(embedding_function=embedding_model.embed_query)
    
    # 7. 添加文档到向量存储
    print("\n步骤7: 添加文档到向量存储...")
    vector_store.add_documents(chunks, embeddings)
    
    # 8. 获取集合信息
    print("\n步骤8: 集合信息:")
    info = vector_store.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 9. 执行相似度搜索示例
    print("\n步骤9: 执行相似度搜索示例...")
    
    # 示例查询
    queries = [
        "什么是机器学习？",
        "深度学习的应用有哪些？",
        "如何训练神经网络？"
    ]
    
    for query in queries:
        print(f"\n查询: {query}")
        query_embedding = embedding_model.embed_query(query)
        results = vector_store.similarity_search(query_embedding, k=3)
        
        print("搜索结果:")
        for i, result in enumerate(results):
            print(f"\n结果 {i+1}:")
            print(f"  文档ID: {result['id']}")
            print(f"  内容片段: {result['document'][:100]}...")
            print(f"  来源: {result['metadata'].get('source', '未知')}")
            print(f"  页码: {result['metadata'].get('page', '未知')}")
            print(f"  块索引: {result['metadata'].get('chunk_index', '未知')}")
            print(f"  相似度分数: {result['similarity_score']:.4f}")
    
    # 10. 持久化
    print("\n步骤10: 持久化数据...")
    vector_store.persist()
    
    print("\n✅ 向量存储示例完成！")
    print(f"数据已保存到: {persist_directory}")
    print("您可以在后续会话中重新加载这些数据进行搜索。")


if __name__ == "__main__":
    main()