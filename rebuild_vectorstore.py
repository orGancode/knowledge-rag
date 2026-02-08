#!/usr/bin/env python3
# 重建向量数据库脚本

import os
import shutil
from dotenv import load_dotenv

# 添加src目录到Python路径
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.loaders.pdf_loader import PDFLoader
from src.splitters import HeadingBasedSplitter
from src.embeddings import OllamaEmbeddings
from src.vectorstore import ChromaVectorStore

# 加载环境变量
load_dotenv()


def rebuild_vectorstore(
    pdf_path: str = "data/员工手册.pdf",
    collection_name: str = "employee_handbook",
    persist_directory: str = "./chroma_db",
    embedding_model: str = "bge-m3:latest",
    max_chunk_size: int = 1500,
    chunk_overlap: int = 100
):
    """
    重建向量数据库
    
    Args:
        pdf_path: PDF文件路径
        collection_name: 集合名称
        persist_directory: 持久化目录
        embedding_model: 嵌入模型
        max_chunk_size: 最大块大小
        chunk_overlap: 块重叠大小
    """
    print("=" * 60)
    print("🔄 重建向量数据库")
    print("=" * 60)
    
    # 1. 删除旧的向量数据库
    print(f"\n🗑️  删除旧的向量数据库: {persist_directory}")
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("✅ 旧向量数据库已删除")
    else:
        print("⚠️  旧的向量数据库不存在，跳过删除步骤")
    
    # 2. 初始化嵌入模型
    print(f"\n🔤 加载嵌入模型: {embedding_model}")
    embedder = OllamaEmbeddings(model=embedding_model)
    
    # 3. 初始化向量存储
    print(f"\n💾 初始化向量数据库: {collection_name}")
    store = ChromaVectorStore(
        collection_name=collection_name,
        persist_directory=persist_directory
    )
    store.create_collection(embedder.embed_query)
    
    # 4. 加载PDF文档
    print(f"\n📄 加载PDF文档: {pdf_path}")
    loader = PDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ 共加载 {len(documents)} 页文档")
    
    # 5. 分割文档
    print(f"\n✂️  按标题分割文档 (max_chunk_size={max_chunk_size}, overlap={chunk_overlap})")
    splitter = HeadingBasedSplitter(
        max_chunk_size=max_chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = splitter.split(documents)
    print(f"✅ 共生成 {len(chunks)} 个文本块")
    
    # 6. 生成嵌入向量
    print(f"\n🔢 生成嵌入向量...")
    texts = [doc.page_content for doc in chunks]
    embeddings = embedder.embed_documents(texts)
    print(f"✅ 嵌入向量维度: {len(embeddings[0])}")
    
    # 7. 存储到向量数据库
    print(f"\n💾 存储到向量数据库...")
    store.add_documents(chunks, embeddings)
    store.persist()
    print(f"✅ 数据已持久化到: {persist_directory}")
    
    # 8. 验证
    print(f"\n🔍 验证向量数据库...")
    collection_info = store.get_collection_info()
    print(f"✅ 向量数据库包含 {collection_info['count']} 个文档块")
    
    print("\n" + "=" * 60)
    print("✅ 向量数据库重建完成！")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="重建向量数据库")
    parser.add_argument(
        "--pdf", "-p",
        type=str,
        default="data/员工手册.pdf",
        help="PDF文件路径"
    )
    parser.add_argument(
        "--collection", "-c",
        type=str,
        default="employee_handbook",
        help="集合名称"
    )
    parser.add_argument(
        "--persist-dir", "-d",
        type=str,
        default="./chroma_db",
        help="持久化目录"
    )
    parser.add_argument(
        "--embedding-model", "-e",
        type=str,
        default="bge-m3:latest",
        help="嵌入模型"
    )
    parser.add_argument(
        "--max-chunk-size", "-s",
        type=int,
        default=1500,
        help="最大块大小"
    )
    parser.add_argument(
        "--chunk-overlap", "-o",
        type=int,
        default=100,
        help="块重叠大小"
    )
    
    args = parser.parse_args()
    
    rebuild_vectorstore(
        pdf_path=args.pdf,
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        embedding_model=args.embedding_model,
        max_chunk_size=args.max_chunk_size,
        chunk_overlap=args.chunk_overlap
    )
