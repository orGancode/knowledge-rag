#!/usr/bin/env python3
"""
问答链使用示例
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chains.qa_chain import BasicQAChain
from embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
from vectorstore.chroma_store import ChromaVectorStore
from loaders.pdf_loader import PDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def create_sample_qa_system():
    """创建一个简单的问答系统示例"""
    print("创建问答系统示例...")
    
    # 1. 初始化嵌入模型
    print("1. 初始化嵌入模型...")
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # 2. 创建向量存储
    print("2. 创建向量存储...")
    vectorstore = ChromaVectorStore(
        collection_name="example_qa_collection",
        persist_directory="./example_db"
    )
    
    # 初始化集合
    vectorstore.create_collection(embeddings.embed_query)
    
    # 3. 加载和处理文档
    print("3. 加载和处理文档...")
    try:
        # 尝试加载PDF文档
        loader = PDFLoader()
        documents = loader.load("data/员工手册.pdf")
        
        # 如果没有找到文档，使用示例文档
        if not documents:
            print("未找到PDF文档，使用示例文档...")
            documents = [
                Document(
                    page_content="人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                    metadata={"source": "example_doc.pdf", "page": 1}
                ),
                Document(
                    page_content="机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
                    metadata={"source": "example_doc.pdf", "page": 2}
                ),
                Document(
                    page_content="深度学习是机器学习的一个子集，使用人工神经网络来模拟人脑的工作方式。",
                    metadata={"source": "example_doc.pdf", "page": 3}
                ),
                Document(
                    page_content="自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
                    metadata={"source": "example_doc.pdf", "page": 4}
                ),
                Document(
                    page_content="计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取有意义的信息。",
                    metadata={"source": "example_doc.pdf", "page": 5}
                )
            ]
    except Exception as e:
        print(f"加载文档时出错: {str(e)}，使用示例文档...")
        documents = [
            Document(
                page_content="人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                metadata={"source": "example_doc.pdf", "page": 1}
            ),
            Document(
                page_content="机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
                metadata={"source": "example_doc.pdf", "page": 2}
            ),
            Document(
                page_content="深度学习是机器学习的一个子集，使用人工神经网络来模拟人脑的工作方式。",
                metadata={"source": "example_doc.pdf", "page": 3}
            ),
            Document(
                page_content="自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
                metadata={"source": "example_doc.pdf", "page": 4}
            ),
            Document(
                page_content="计算机视觉是人工智能的一个分支，使计算机能够从数字图像或视频中获取有意义的信息。",
                metadata={"source": "example_doc.pdf", "page": 5}
            )
        ]
    
    # 4. 分割文档
    print("4. 分割文档...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # 5. 生成文档嵌入
    print("5. 生成文档嵌入...")
    doc_texts = [chunk.page_content for chunk in chunks]
    doc_embeddings = embeddings.embed_documents(doc_texts)
    
    # 6. 添加文档到向量存储
    print("6. 添加文档到向量存储...")
    vectorstore.add_documents(chunks, doc_embeddings)
    
    # 7. 创建检索器函数
    def retriever_function(query, k=3):
        query_embedding = embeddings.embed_query(query)
        return vectorstore.similarity_search(query_embedding, k=k)
    
    # 8. 创建问答链
    print("7. 创建问答链...")
    qa_chain = BasicQAChain(
        retriever=retriever_function, 
        llm_model="qwen:4b"
    )
    
    return qa_chain, vectorstore

def demo_qa_system():
    """演示问答系统"""
    print("\n=== 问答系统演示 ===\n")
    
    # 创建问答系统
    qa_chain, vectorstore = create_sample_qa_system()
    
    # 示例问题
    questions = [
        "什么是人工智能？",
        "机器学习和人工智能有什么关系？",
        "深度学习是什么？",
        "自然语言处理是做什么的？",
        "计算机视觉的应用有哪些？"
    ]
    
    # 处理每个问题
    for i, question in enumerate(questions, 1):
        print(f"\n--- 问题 {i} ---")
        print(f"问题: {question}")
        
        # 执行问答
        result = qa_chain.run(question)
        
        # 显示结果
        print(f"答案: {result['answer']}")
        print(f"源文档数量: {len(result['source_documents'])}")
        
        # 显示源文档信息
        if result['source_documents']:
            print("源文档:")
            for j, doc in enumerate(result['source_documents'], 1):
                print(f"  {j}. 相似度: {doc['similarity_score']:.4f}")
                print(f"     内容: {doc['content'][:100]}...")
                if doc['metadata']:
                    print(f"     元数据: {doc['metadata']}")
    
    # 清理资源
    print("\n清理资源...")
    vectorstore.delete_collection()
    print("演示完成！")

if __name__ == "__main__":
    demo_qa_system()