# RAG Demo - Main Entry Point

import os
from dotenv import load_dotenv
# 直接从pdf_loader模块导入PDFLoader
from loaders.pdf_loader import PDFLoader
# Load environment variables
load_dotenv()


def main():
    """RAG Demo main function."""
    print("🚀 Welcome to RAG Demo!")
    print("=" * 50)
    
    # TODO: Implement your RAG pipeline here
    # 1. Load documents from data/
    # 2. Split documents into chunks
    # 3. Create embeddings
    # 4. Store in vector database
    # 5. Create retrieval chain
    # 6. Run queries
    
    print("\nProject structure initialized successfully!")
    print("Next steps:")
    print("1. Add your PDF documents to data/")
    print("2. Implement document loaders in src/loaders/")
    print("3. Implement text splitters in src/splitters/")
    print("4. Implement embeddings in src/embeddings/")
    print("5. Implement vector store in src/vectorstore/")
    print("6. Implement chains in src/chains/")
    print("7. Run: python src/main.py")
   

    # 使用绝对路径指向数据文件
    data_path = os.path.join(os.getcwd(), 'data', '员工手册.pdf')
    print(f"数据文件路径: {data_path}")
    loader = PDFLoader(data_path)
    docs = loader.load()
    print(f"加载了{len(docs)}页")
    print(f"第一页前200字符：{docs[0].page_content[:200]}")
    print(f"Metadata：{docs[0].metadata}")


if __name__ == "__main__":
    main()
