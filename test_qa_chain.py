#!/usr/bin/env python3
"""
测试问答链功能的脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chains.qa_chain import BasicQAChain
from embeddings import OllamaEmbeddings
from vectorstore.chroma_store import ChromaVectorStore
from langchain.schema import Document

class MockRetriever:
    """模拟检索器，用于测试"""
    
    def __init__(self):
        # 准备一些测试文档
        self.documents = [
            {
                "document": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
                "metadata": {"source": "test_doc.pdf", "page": 1},
                "similarity_score": 0.95
            },
            {
                "document": "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
                "metadata": {"source": "test_doc.pdf", "page": 2},
                "similarity_score": 0.90
            },
            {
                "document": "深度学习是机器学习的一个子集，使用人工神经网络来模拟人脑的工作方式。",
                "metadata": {"source": "test_doc.pdf", "page": 3},
                "similarity_score": 0.85
            }
        ]
    
    def similarity_search(self, query, k=4):
        """模拟相似度搜索"""
        # 简单返回所有文档，实际应用中会根据查询相似度排序
        return self.documents[:k]

def test_qa_chain_basic():
    """测试基础问答链功能"""
    print("测试 BasicQAChain 基础功能...")
    
    try:
        # 创建模拟检索器
        retriever = MockRetriever()
        
        # 创建问答链（使用本地模型）
        qa_chain = BasicQAChain(retriever=retriever, llm_model="qwen2.5:7b")
        
        # 测试问题
        test_question = "什么是人工智能？"
        
        # 执行问答
        result = qa_chain.run(test_question)
        
        # 打印调试信息
        print(f"调试信息 - result keys: {result.keys()}")
        print(f"调试信息 - question: {result.get('question', 'N/A')}")
        print(f"调试信息 - answer: {result.get('answer', 'N/A')}")
        print(f"调试信息 - source_documents: {result.get('source_documents', 'N/A')}")
        
        # 验证结果
        assert "question" in result
        assert "answer" in result
        assert "source_documents" in result
        assert result["question"] == test_question
        
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"源文档数量: {len(result['source_documents'])}")
        
        print("✅ BasicQAChain 基础功能测试成功!")
        return True
    except Exception as e:
        import traceback
        print(f"❌ BasicQAChain 基础功能测试失败: {str(e)}")
        print(f"错误详情: {traceback.format_exc()}")
        return False

def test_qa_chain_with_real_components():
    """测试问答链与真实组件的集成"""
    print("\n测试 BasicQAChain 与真实组件集成...")
    
    try:
        # 初始化嵌入模型（使用Ollama）
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # 创建向量存储
        vectorstore = ChromaVectorStore(
            collection_name="test_qa_collection",
            persist_directory="./test_db"
        )
        
        # 初始化集合
        vectorstore.create_collection(embeddings.embed_query)
        
        # 准备测试文档
        documents = [
            Document(
                page_content="Python是一种高级编程语言，由Guido van Rossum于1991年首次发布。",
                metadata={"source": "python_doc.pdf", "page": 1}
            ),
            Document(
                page_content="Python具有简洁明了的语法，使得代码更易于阅读和维护。",
                metadata={"source": "python_doc.pdf", "page": 2}
            ),
            Document(
                page_content="Python广泛应用于数据科学、人工智能、Web开发等领域。",
                metadata={"source": "python_doc.pdf", "page": 3}
            )
        ]
        
        # 生成文档嵌入
        doc_texts = [doc.page_content for doc in documents]
        doc_embeddings = embeddings.embed_documents(doc_texts)
        
        # 添加文档到向量存储
        vectorstore.add_documents(documents, doc_embeddings)
        
        # 创建检索器函数
        def retriever_function(query, k=3):
            query_embedding = embeddings.embed_query(query)
            return vectorstore.similarity_search(query_embedding, k=k)
        
        # 创建问答链（使用本地模型）
        qa_chain = BasicQAChain(retriever=retriever_function, llm_model="llama3:8b")
        
        # 测试问题
        test_question = "Python是什么时候发布的？"
        
        # 执行问答
        result = qa_chain.run(test_question)
        
        # 验证结果
        assert "question" in result
        assert "answer" in result
        assert "source_documents" in result
        
        print(f"问题: {result['question']}")
        print(f"答案: {result['answer']}")
        print(f"源文档数量: {len(result['source_documents'])}")
        
        # 清理测试数据
        vectorstore.delete_collection()
        
        print("✅ BasicQAChain 与真实组件集成测试成功!")
        return True
    except Exception as e:
        print(f"❌ BasicQAChain 与真实组件集成测试失败: {str(e)}")
        return False

def test_qa_chain_error_handling():
    """测试问答链的错误处理"""
    print("\n测试 BasicQAChain 错误处理...")
    
    try:
        # 创建一个会抛出异常的模拟检索器
        class ErrorRetriever:
            def similarity_search(self, query, k=4):
                raise Exception("模拟检索器错误")
        
        retriever = ErrorRetriever()
        qa_chain = BasicQAChain(retriever=retriever, llm_model="llama3:8b")
        
        # 执行问答
        result = qa_chain.run("测试问题")
        
        # 验证错误处理
        assert "question" in result
        assert "answer" in result
        assert "source_documents" in result
        assert "处理问题时发生错误" in result["answer"]
        
        print(f"问题: {result['question']}")
        print(f"错误答案: {result['answer']}")
        
        print("✅ BasicQAChain 错误处理测试成功!")
        return True
    except Exception as e:
        print(f"❌ BasicQAChain 错误处理测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试问答链功能...\n")
    
    # 运行所有测试
    results = []
    results.append(test_qa_chain_basic())
    results.append(test_qa_chain_with_real_components())
    results.append(test_qa_chain_error_handling())
    
    # 总结测试结果
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n测试总结: {success_count}/{total_count} 个测试通过")
    
    if success_count == total_count:
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")