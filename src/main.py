# RAG Demo - 基于员工手册的问答系统

import os
import sys
from dotenv import load_dotenv

# 禁用 ChromaDB 遥测功能以避免错误
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loaders.pdf_loader import PDFLoader
from splitters import HeadingBasedSplitter
from embeddings import OllamaEmbeddings
from vectorstore import ChromaVectorStore
from chains.qa_chain import BasicQAChain

# 加载环境变量
load_dotenv()


class EmployeeHandbookQA:
    """员工手册问答系统"""
    
    def __init__(
        self,
        pdf_path: str = "data/员工手册.pdf",
        collection_name: str = "employee_handbook",
        persist_directory: str = "./chroma_db",
        max_chunk_size: int = 1500,
        chunk_overlap: int = 100,
        embedding_model: str = "BAAI/bge-m3",
        llm_model: str = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    ):
        """
        初始化员工手册问答系统
        
        Args:
            pdf_path: 员工手册PDF文件路径
            collection_name: 向量数据库集合名称
            persist_directory: 向量数据库持久化目录
            max_chunk_size: 文本块最大字符数（用于过长的标题块）
            chunk_overlap: 文本块重叠大小
            embedding_model: 嵌入模型名称
            llm_model: LLM模型名称
        """
        self.pdf_path = pdf_path
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # 初始化组件
        self.embedder = None
        self.store = None
        self.qa_chain = None
        
        print("=" * 60)
        print("🚀 员工手册问答系统")
        print("=" * 60)
    
    def initialize(self, force_rebuild: bool = False):
        """
        初始化系统
        
        Args:
            force_rebuild: 是否强制重建向量数据库
        """
        print("\n📋 正在初始化系统...")
        
        # 1. 初始化嵌入模型
        print(f"\n🔤 加载嵌入模型: {self.embedding_model}")
        self.embedder = OllamaEmbeddings(model=self.embedding_model)
        
        # 2. 初始化向量存储
        print(f"\n💾 初始化向量数据库: {self.collection_name}")
        self.store = ChromaVectorStore(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory
        )
        self.store.create_collection(self.embedder.embed_query)
        
        # 3. 检查是否需要重建向量数据库
        collection_info = self.store.get_collection_info()
        
        if collection_info["count"] > 0 and not force_rebuild:
            print(f"✅ 向量数据库已存在，包含 {collection_info['count']} 个文档块")
            print("   如需重建，请使用 force_rebuild=True")
        else:
            print("📄 正在处理PDF文档...")
            self._build_vector_store()
        
        # 4. 初始化问答链
        print(f"\n🤖 初始化问答链: {self.llm_model}")
        self.qa_chain = BasicQAChain(
            retriever=self._create_retriever(),
            llm_model=self.llm_model
        )
        
        print("\n✅ 系统初始化完成！")
        print("=" * 60)
    
    def _build_vector_store(self):
        """构建向量数据库"""
        # 1. 加载PDF文档
        print(f"   - 加载PDF: {self.pdf_path}")
        loader = PDFLoader(self.pdf_path)
        documents = loader.load()
        print(f"   - 共加载 {len(documents)} 页文档")
        
        # 2. 分割文档（使用基于标题的分割器）
        print(f"   - 按标题分割文档 (max_chunk_size={self.max_chunk_size}, overlap={self.chunk_overlap})")
        splitter = HeadingBasedSplitter(
            max_chunk_size=self.max_chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split(documents)
        print(f"   - 共生成 {len(chunks)} 个文本块")
        
        # 3. 生成嵌入向量
        print(f"   - 生成嵌入向量...")
        texts = [doc.page_content for doc in chunks]
        embeddings = self.embedder.embed_documents(texts)
        print(f"   - 嵌入向量维度: {len(embeddings[0])}")
        
        # 4. 存储到向量数据库
        print(f"   - 存储到向量数据库...")
        self.store.add_documents(chunks, embeddings)
        self.store.persist()
        print(f"   - 数据已持久化到: {self.persist_directory}")
    
    def _create_retriever(self):
        """创建检索器适配器"""
        class VectorStoreRetriever:
            def __init__(self, store, embedder):
                self.store = store
                self.embedder = embedder
            
            def similarity_search(self, question: str, k: int = 4):
                """将问题转换为嵌入向量，然后进行相似度搜索"""
                query_embedding = self.embedder.embed_query(question)
                return self.store.similarity_search(query_embedding, k)
        
        return VectorStoreRetriever(self.store, self.embedder)
    
    def ask(self, question: str, k: int = 7) -> dict:
        """
        提问
        
        Args:
            question: 用户问题
            k: 检索的文档数量
            
        Returns:
            包含答案和参考文档的字典
        """
        if not self.qa_chain:
            raise RuntimeError("系统未初始化，请先调用 initialize() 方法")
        
        print(f"\n❓ 问题: {question}")
        print("-" * 60)
        
        result = self.qa_chain.run(question, k=k)
        
        print(f"📝 答案: {result['answer']}")
        
        if result['source_documents']:
            print(f"\n📚 参考文档:")
            for i, doc in enumerate(result['source_documents'], 1):
                metadata = doc['metadata']
                page = metadata.get('page', '未知')
                similarity = doc.get('similarity_score', 0)
                print(f"   [{i}] 页码: {page} | 相似度: {similarity:.3f}")
                # print(f"       内容: {doc['content'][:100]}...")
        
        print("=" * 60)
        
        return result
    
    def interactive_mode(self):
        """交互式问答模式"""
        print("\n🎯 进入交互式问答模式")
        print("输入问题开始提问，输入 'quit' 或 'exit' 退出")
        print("=" * 60)
        
        while True:
            try:
                question = input("\n❓ 请输入问题: ").strip()
                
                if not question:
                    continue
                
                if question.lower() in ['quit', 'exit', '退出']:
                    print("\n👋 感谢使用，再见！")
                    break
                
                self.ask(question)
                
            except KeyboardInterrupt:
                print("\n\n👋 感谢使用，再见！")
                break
            except Exception as e:
                print(f"\n❌ 发生错误: {str(e)}")


def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="员工手册问答系统")
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="进入交互式问答模式"
    )
    parser.add_argument(
        "--question", "-q",
        type=str,
        help="直接提问"
    )
    parser.add_argument(
        "--rebuild", "-r",
        action="store_true",
        help="强制重建向量数据库"
    )
    parser.add_argument(
        "--example", "-e",
        action="store_true",
        help="运行示例问题"
    )
    args = parser.parse_args()
    
    # 创建问答系统实例
    qa_system = EmployeeHandbookQA(
        pdf_path="data/员工手册.pdf",
        collection_name="employee_handbook",
        persist_directory="./chroma_db",
        max_chunk_size=1500,
        chunk_overlap=100,
        embedding_model="bge-m3:latest",
        llm_model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    )
    
    # 初始化系统
    # 首次运行时，force_rebuild=False 会自动检测并构建向量数据库
    # 如果需要重建，设置 force_rebuild=True
    qa_system.initialize(force_rebuild=args.rebuild)
    
    # 示例问题
    example_questions = [
        "员工的午休时间是几点？",
        "公司有哪些福利待遇？",
        "请假制度是怎样的？",
        "员工手册中关于加班的规定是什么？"
    ]
    
    # 根据参数执行不同操作
    if args.question:
        # 直接提问
        qa_system.ask(args.question)
    elif args.interactive:
        # 交互式问答模式
        print("\n💡 示例问题:")
        for i, q in enumerate(example_questions, 1):
            print(f"   {i}. {q}")
        qa_system.interactive_mode()
    elif args.example:
        # 运行示例问题
        print("\n💡 运行示例问题:")
        for i, q in enumerate(example_questions, 1):
            print(f"\n[{i}] {q}")
            qa_system.ask(q)
    else:
        # 默认：显示帮助信息并运行一个示例
        print("\n💡 示例问题:")
        for i, q in enumerate(example_questions, 1):
            print(f"   {i}. {q}")
        
        print("\n" + "=" * 60)
        print("📖 使用说明:")
        print("   python src/main.py                    # 运行示例问题")
        print("   python src/main.py -i                 # 交互式问答模式")
        print("   python src/main.py -q '你的问题'      # 直接提问")
        print("   python src/main.py -e                 # 运行所有示例问题")
        print("   python src/main.py -r                # 强制重建向量数据库")
        print("=" * 60)
        
        # 运行一个示例问题
        print("\n运行示例问题...")
        qa_system.ask(example_questions[0])


if __name__ == "__main__":
    main()
