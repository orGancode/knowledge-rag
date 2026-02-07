#!/usr/bin/env python3
"""
使用向量化功能的示例脚本
"""

import sys
import os

# 添加src目录到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from embeddings import OllamaEmbeddings, OpenAIEmbeddings, HuggingFaceEmbeddings

def main():
    print("向量化功能使用示例\n")
    
    # 示例文本
    sample_text = "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。"
    sample_documents = [
        "机器学习是人工智能的一个子集，它使计算机能够在没有明确编程的情况下学习和改进。",
        "深度学习是机器学习的一个子集，它使用神经网络来模拟人脑的工作方式。",
        "自然语言处理是人工智能的一个分支，专注于计算机与人类语言之间的交互。"
    ]
    
    # 1. 使用 Ollama 嵌入模型
    print("1. 使用 Ollama 嵌入模型 (nomic-embed-text)")
    print("-" * 50)
    try:
        ollama_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # 嵌入单个查询
        query_vector = ollama_embeddings.embed_query(sample_text)
        print(f"查询文本: {sample_text}")
        print(f"向量维度: {len(query_vector)}")
        print(f"向量前5个值: {query_vector[:5]}")
        
        # 嵌入多个文档
        doc_vectors = ollama_embeddings.embed_documents(sample_documents)
        print(f"\n文档数量: {len(doc_vectors)}")
        print(f"每个文档向量维度: {len(doc_vectors[0])}")
        print("第一个文档向量前5个值:", doc_vectors[0][:5])
        
        print("\n✅ Ollama 嵌入示例成功完成!")
    except Exception as e:
        print(f"❌ Ollama 嵌入示例失败: {str(e)}")
    
    print("\n" + "="*60 + "\n")
    
    # 2. 使用 HuggingFace 嵌入模型
    print("2. 使用 HuggingFace 嵌入模型 (sentence-transformers/all-MiniLM-L6-v2)")
    print("-" * 50)
    try:
        hf_embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # 嵌入单个查询
        query_vector = hf_embeddings.embed_query(sample_text)
        print(f"查询文本: {sample_text}")
        print(f"向量维度: {len(query_vector)}")
        print(f"向量前5个值: {query_vector[:5]}")
        
        # 嵌入多个文档
        doc_vectors = hf_embeddings.embed_documents(sample_documents)
        print(f"\n文档数量: {len(doc_vectors)}")
        print(f"每个文档向量维度: {len(doc_vectors[0])}")
        print("第一个文档向量前5个值:", doc_vectors[0][:5])
        
        # 获取模型信息
        model_info = hf_embeddings.get_model_info()
        print(f"\n模型信息: {model_info}")
        
        # 计算文本相似度
        similarity_score = hf_embeddings.similarity(
            "机器学习是人工智能的一个子集",
            "深度学习是机器学习的一个分支"
        )
        print(f"\n文本相似度分数: {similarity_score:.4f}")
        
        print("\n✅ HuggingFace 嵌入示例成功完成!")
    except Exception as e:
        print(f"❌ HuggingFace 嵌入示例失败: {str(e)}")
    
    print("\n" + "="*60 + "\n")
    
    # 3. 使用 HuggingFace 高级功能
    print("3. 使用 HuggingFace 高级功能 (多语言模型)")
    print("-" * 50)
    try:
        # 使用多语言模型，支持中文
        hf_advanced = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            device="auto",
            encode_kwargs={
                "batch_size": 16,
                "show_progress_bar": True,
                "normalize_embeddings": True
            }
        )
        
        # 中文文本示例
        chinese_texts = [
            "人工智能正在改变我们的生活方式。",
            "机器学习算法可以自动从数据中学习模式。",
            "深度学习是机器学习的一个分支，使用神经网络。"
        ]
        
        # 嵌入中文文本
        chinese_vectors = hf_advanced.embed_documents(chinese_texts)
        print(f"中文文档数量: {len(chinese_vectors)}")
        print(f"每个文档向量维度: {len(chinese_vectors[0])}")
        
        # 计算中文文本相似度
        similarity_score = hf_advanced.similarity(
            "人工智能和机器学习密切相关。",
            "机器学习是人工智能的一个重要组成部分。"
        )
        print(f"\n中文文本相似度分数: {similarity_score:.4f}")
        
        # 获取高级模型信息
        advanced_model_info = hf_advanced.get_model_info()
        print(f"\n多语言模型信息: {advanced_model_info}")
        
        print("\n✅ HuggingFace 高级功能示例成功完成!")
    except Exception as e:
        print(f"❌ HuggingFace 高级功能示例失败: {str(e)}")
    
    print("\n" + "="*60 + "\n")
    
    # 4. 使用 OpenAI 嵌入模型 (如果有API密钥)
    print("4. 使用 OpenAI 嵌入模型 (text-embedding-3-small)")
    print("-" * 50)
    try:
        if os.getenv("OPENAI_API_KEY"):
            openai_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            
            # 嵌入单个查询
            query_vector = openai_embeddings.embed_query(sample_text)
            print(f"查询文本: {sample_text}")
            print(f"向量维度: {len(query_vector)}")
            print(f"向量前5个值: {query_vector[:5]}")
            
            # 嵌入多个文档
            doc_vectors = openai_embeddings.embed_documents(sample_documents)
            print(f"\n文档数量: {len(doc_vectors)}")
            print(f"每个文档向量维度: {len(doc_vectors[0])}")
            print("第一个文档向量前5个值:", doc_vectors[0][:5])
            
            print("\n✅ OpenAI 嵌入示例成功完成!")
        else:
            print("⚠️ 未找到 OPENAI_API_KEY 环境变量，跳过 OpenAI 示例")
    except Exception as e:
        print(f"❌ OpenAI 嵌入示例失败: {str(e)}")
    
    print("\n" + "="*60)
    print("\n📝 使用说明:")
    print("1. OllamaEmbeddings: 用于本地 Ollama 模型，默认使用 nomic-embed-text")
    print("2. HuggingFaceEmbeddings: 用于 HuggingFace 上的预训练模型")
    print("3. HuggingFaceEmbeddings 高级功能: 支持多语言、自定义配置、相似度计算等")
    print("4. OpenAIEmbeddings: 用于 OpenAI 的嵌入模型，需要设置 OPENAI_API_KEY 环境变量")
    print("\n🚀 HuggingFaceEmbeddings 新功能:")
    print("- 自动设备选择 (CPU/GPU)")
    print("- 模型缓存和批处理优化")
    print("- 文本相似度计算")
    print("- 模型信息获取")
    print("- 多语言支持")
    print("- 自定义编码参数")

if __name__ == "__main__":
    main()