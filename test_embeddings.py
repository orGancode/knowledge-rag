#!/usr/bin/env python3
"""
测试向量化功能的脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
# from embeddings import OllamaEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings

def test_ollama_embeddings():
    """测试Ollama向量化功能"""
    print("测试 OllamaEmbeddings (nomic-embed-text)...")
    
    try:
        # 初始化Ollama嵌入模型，使用专门的嵌入模型
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # 测试单个文本向量化
        test_text = "这是一个测试文本，用于验证向量化功能。"
        query_embedding = embeddings.embed_query(test_text)
        print(f"查询文本向量维度: {len(query_embedding)}")
        print(f"查询文本向量前5个值: {query_embedding[:5]}")
        
        # 测试批量文本向量化
        test_texts = [
            "这是第一个测试文档。",
            "这是第二个测试文档，内容略有不同。",
            "人工智能是计算机科学的一个分支。"
        ]
        doc_embeddings = embeddings.embed_documents(test_texts)
        print(f"文档向量数量: {len(doc_embeddings)}")
        print(f"每个文档向量维度: {len(doc_embeddings[0])}")
        print(f"第一个文档向量前5个值: {doc_embeddings[0][:5]}")
        
        print("✅ OllamaEmbeddings 测试成功!")
        return True
    except Exception as e:
        print(f"❌ OllamaEmbeddings 测试失败: {str(e)}")
        return False

# def test_openai_embeddings():
#     """测试OpenAI向量化功能"""
#     print("\n测试 OpenAIEmbeddings...")
#
#     try:
#         # 检查是否有API密钥
#         if not os.getenv("OPENAI_API_KEY"):
#             print("⚠️ 未找到 OPENAI_API_KEY 环境变量，跳过 OpenAI 测试")
#             return True
#
#         # 初始化OpenAI嵌入模型
#         embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
#
#         # 测试单个文本向量化
#         test_text = "This is a test text for embedding verification."
#         query_embedding = embeddings.embed_query(test_text)
#         print(f"查询文本向量维度: {len(query_embedding)}")
#         print(f"查询文本向量前5个值: {query_embedding[:5]}")
#
#         # 测试批量文本向量化
#         test_texts = [
#             "This is the first test document.",
#             "This is the second test document with slightly different content.",
#             "Artificial intelligence is a branch of computer science."
#         ]
#         doc_embeddings = embeddings.embed_documents(test_texts)
#         print(f"文档向量数量: {len(doc_embeddings)}")
#         print(f"每个文档向量维度: {len(doc_embeddings[0])}")
#         print(f"第一个文档向量前5个值: {doc_embeddings[0][:5]}")
#
#         print("✅ OpenAIEmbeddings 测试成功!")
#         return True
#     except Exception as e:
#         print(f"❌ OpenAIEmbeddings 测试失败: {str(e)}")
#         return False

def test_huggingface_embeddings():
    """测试HuggingFace向量化功能"""
    print("\n测试 HuggingFaceEmbeddings...")
    
    try:
        # 初始化HuggingFace嵌入模型
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 测试单个文本向量化
        test_text = "This is a test text for embedding verification."
        query_embedding = embeddings.embed_query(test_text)
        print(f"查询文本向量维度: {len(query_embedding)}")
        print(f"查询文本向量前5个值: {query_embedding[:5]}")
        
        # 测试批量文本向量化
        test_texts = [
            "This is the first test document.",
            "This is the second test document with slightly different content.",
            "Artificial intelligence is a branch of computer science."
        ]
        doc_embeddings = embeddings.embed_documents(test_texts)
        print(f"文档向量数量: {len(doc_embeddings)}")
        print(f"每个文档向量维度: {len(doc_embeddings[0])}")
        print(f"第一个文档向量前5个值: {doc_embeddings[0][:5]}")
        
        # 测试模型信息获取
        model_info = embeddings.get_model_info()
        print(f"模型信息: {model_info}")
        
        # 测试相似度计算
        similarity_score = embeddings.similarity(
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning is a type of machine learning."
        )
        print(f"相似度分数: {similarity_score:.4f}")
        
        print("✅ HuggingFaceEmbeddings 测试成功!")
        return True
    except Exception as e:
        print(f"❌ HuggingFaceEmbeddings 测试失败: {str(e)}")
        return False

def test_huggingface_embeddings_advanced():
    """测试HuggingFace向量化高级功能"""
    print("\n测试 HuggingFaceEmbeddings 高级功能...")
    
    try:
        # 测试不同的模型配置
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="auto",
            encode_kwargs={
                "batch_size": 16,
                "show_progress_bar": True,
                "normalize_embeddings": True
            }
        )
        
        # 测试中文文本
        chinese_texts = [
            "人工智能是计算机科学的一个分支。",
            "机器学习是人工智能的一个重要组成部分。",
            "深度学习使用神经网络来模拟人脑的工作方式。"
        ]
        
        # 嵌入中文文本
        doc_embeddings = embeddings.embed_documents(chinese_texts)
        print(f"中文文档向量数量: {len(doc_embeddings)}")
        print(f"每个文档向量维度: {len(doc_embeddings[0])}")
        
        # 测试中文相似度计算
        similarity_score = embeddings.similarity(
            "人工智能和机器学习密切相关。",
            "机器学习是人工智能的一个子领域。"
        )
        print(f"中文文本相似度分数: {similarity_score:.4f}")
        
        # 获取模型信息
        model_info = embeddings.get_model_info()
        print(f"多语言模型信息: {model_info}")
        
        print("✅ HuggingFaceEmbeddings 高级功能测试成功!")
        return True
    except Exception as e:
        print(f"❌ HuggingFaceEmbeddings 高级功能测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    print("开始测试向量化功能...\n")
    
    # 测试各种嵌入模型
    results = []
    results.append(test_ollama_embeddings())
    # results.append(test_openai_embeddings())  # 暂时注释掉OpenAI测试
    results.append(test_huggingface_embeddings())
    results.append(test_huggingface_embeddings_advanced())
    
    # 总结测试结果
    success_count = sum(results)
    total_count = len(results)
    
    print(f"\n测试总结: {success_count}/{total_count} 个测试通过")
    
    if success_count == total_count:
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 部分测试失败，请检查错误信息")