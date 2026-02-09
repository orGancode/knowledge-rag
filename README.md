# RAG Demo 项目

一个用于学习和演示 RAG (Retrieval-Augmented Generation) 技术的完整项目，实现了基于员工手册的智能问答系统。

## 📋 项目简介

本项目展示了如何构建一个完整的 RAG 系统，包括文档加载、文本分割、向量化、向量存储和问答生成等核心组件。系统支持本地和云端语言模型，使用 ChromaDB 作为向量数据库，并提供了丰富的示例和文档。

## ✨ 主要特性

- **完整的 RAG 流程**: 从文档加载到问答生成的完整实现
- **多种模型支持**: 支持 Ollama 本地模型和云端 API（OpenAI、SiliconFlow 等）
- **智能文本分割**: 基于标题的文档分割策略，保持语义完整性
- **高效向量检索**: 使用 ChromaDB 进行本地向量存储和相似度搜索
- **灵活的问答链**: 支持自定义检索器和提示模板
- **交互式问答**: 提供命令行交互式问答模式
- **丰富的文档**: 详细的使用指南和示例代码

## 🛠️ 技术栈

### 核心框架
- **LangChain 0.3.0**: LLM 应用开发框架
- **LangChain Community**: 社区扩展组件
- **LangChain Ollama**: 本地模型集成
- **LangChain OpenAI**: OpenAI API 集成
- **LangChain HuggingFace**: HuggingFace 模型集成

### 文档处理
- **PyPDF 4.2.0**: PDF 文档解析
- **Python-docx 1.1.0**: Word 文档支持
- **Unstructured 0.14.0**: 万能文档解析

### 向量数据库
- **ChromaDB 0.5.0**: 本地向量数据库

### 嵌入模型
- **Sentence-Transformers 3.0.0**: 文本向量化
- **Torch 2.2.0**: PyTorch 深度学习框架
- **Transformers 4.40.0**: HuggingFace 模型库

### 工具库
- **Python-dotenv**: 环境变量管理
- **Tiktoken**: Token 计数
- **NumPy/Pandas**: 数据处理
- **Scikit-learn**: 相似度计算
- **TQDM**: 进度条显示

## 📁 项目结构

```
rag-demo/
├── data/                           # 数据目录
│   ├── 员工手册.pdf                # 示例文档
│   └── README.md                   # 数据说明
├── docs/                           # 文档目录
│   ├── BasicQAChain_使用指南.md    # 问答链使用指南
│   ├── ChromaVectorStore_使用指南.md # 向量存储使用指南
│   ├── HuggingFaceEmbeddings_使用指南.md # 嵌入模型使用指南
│   └── 性能优化方案.md             # 性能优化建议
├── notebooks/                      # Jupyter 笔记本
│   ├── debug_retrieval.ipynb       # 检索调试笔记本
│   └── setup_path.py               # 路径设置
├── src/                            # 源代码目录
│   ├── chains/                     # 问答链模块
│   │   ├── __init__.py
│   │   └── qa_chain.py             # 基础问答链实现
│   ├── embeddings/                 # 嵌入模型模块
│   │   ├── __init__.py
│   │   └── ollama.py               # Ollama 嵌入模型
│   ├── loaders/                    # 文档加载模块
│   │   ├── __init__.py
│   │   └── pdf_loader.py           # PDF 加载器
│   ├── llms/                       # 语言模型模块
│   │   ├── ollama.py               # Ollama LLM
│   │   └── siliconflow.py          # SiliconFlow LLM
│   ├── splitters/                  # 文本分割模块
│   │   ├── __init__.py
│   │   └── heading_splitter.py    # 基于标题的分割器
│   ├── vectorstore/                # 向量存储模块
│   │   ├── __init__.py
│   │   └── chroma_store.py         # ChromaDB 向量存储
│   └── main.py                     # 主程序入口
├── example_*.py                    # 示例代码
├── test_*.py                       # 测试代码
├── requirements.txt                # Python 依赖
├── setup.py                       # 安装配置
└── README.md                      # 项目说明
```

## 🚀 快速开始

### 1. 环境要求

- Python 3.8 或更高版本
- pip 包管理器
- （可选）Ollama 用于本地模型

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd rag-demo

# 安装依赖
pip install -r requirements.txt

# 或使用 setup.py 安装
pip install -e .
```

### 3. 配置环境变量

创建 `.env` 文件（如果使用云端 API）：

```bash
# OpenAI API（可选）
OPENAI_API_KEY=your_openai_api_key

# SiliconFlow API（可选）
SILICONFLOW_API_KEY=your_siliconflow_api_key
```

### 4. 准备文档

将你的 PDF 文档放入 `data/` 目录。项目默认使用 `data/员工手册.pdf`。

### 5. 运行系统

```bash
# 运行示例问题
python src/main.py

# 交互式问答模式
python src/main.py -i

# 直接提问
python src/main.py -q "员工的午休时间是几点？"

# 运行所有示例问题
python src/main.py -e

# 强制重建向量数据库
python src/main.py -r
```

## 📖 使用指南

### 基础用法

```python
from src.main import EmployeeHandbookQA

# 创建问答系统实例
qa_system = EmployeeHandbookQA(
    pdf_path="data/员工手册.pdf",
    collection_name="employee_handbook",
    persist_directory="./chroma_db",
    embedding_model="bge-m3:latest",
    llm_model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)

# 初始化系统
qa_system.initialize()

# 提问
result = qa_system.ask("员工的午休时间是几点？")
print(result['answer'])
```

### 交互式模式

```bash
python src/main.py -i
```

进入交互式模式后，可以连续提问，输入 `quit` 或 `exit` 退出。

### 自定义配置

```python
# 使用不同的嵌入模型
qa_system = EmployeeHandbookQA(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

# 使用不同的 LLM
qa_system = EmployeeHandbookQA(
    llm_model="gpt-4"  # 或 "llama3:8b"
)

# 调整文本分割参数
qa_system = EmployeeHandbookQA(
    max_chunk_size=2000,
    chunk_overlap=200
)
```

## 🔧 开发流程

### 1. 文档加载 (`src/loaders/`)

实现 PDF、文本等文档的加载和解析。

```python
from src.loaders import PDFLoader

loader = PDFLoader("data/document.pdf")
documents = loader.load()
```

### 2. 文本分割 (`src/splitters/`)

将长文档切分成适合处理的块，保持语义完整性。

```python
from src.splitters import HeadingBasedSplitter

splitter = HeadingBasedSplitter(
    max_chunk_size=1500,
    chunk_overlap=100
)
chunks = splitter.split(documents)
```

### 3. 向量化 (`src/embeddings/`)

将文本转换为向量表示。

```python
from src.embeddings import OllamaEmbeddings

embedder = OllamaEmbeddings(model="bge-m3:latest")
embeddings = embedder.embed_documents(texts)
```

### 4. 向量存储 (`src/vectorstore/`)

存储和检索向量数据。

```python
from src.vectorstore import ChromaVectorStore

store = ChromaVectorStore(
    collection_name="my_collection",
    persist_directory="./chroma_db"
)
store.create_collection(embedder.embed_query)
store.add_documents(chunks, embeddings)
```

### 5. 问答链 (`src/chains/`)

组合所有组件形成 RAG 流程。

```python
from src.chains.qa_chain import BasicQAChain

qa_chain = BasicQAChain(
    retriever=retriever_function,
    llm_model="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)
result = qa_chain.run(question)
```

## 📚 文档

详细的使用指南请参考：

- [BasicQAChain 使用指南](docs/BasicQAChain_使用指南.md) - 问答链的详细说明
- [ChromaVectorStore 使用指南](docs/ChromaVectorStore_使用指南.md) - 向量存储的使用方法
- [HuggingFaceEmbeddings 使用指南](docs/HuggingFaceEmbeddings_使用指南.md) - 嵌入模型的配置
- [性能优化方案](docs/性能优化方案.md) - 性能优化建议

## 🧪 示例代码

项目提供了丰富的示例代码：

- `example_embeddings_usage.py` - 嵌入模型使用示例
- `example_vectorstore_usage.py` - 向量存储使用示例
- `example_qa_chain_usage.py` - 问答链使用示例

## 🧪 测试

运行测试脚本验证功能：

```bash
# 测试嵌入模型
python test_embeddings.py

# 测试向量存储
python test_chroma_store.py

# 测试问答链
python test_qa_chain.py

# 测试标题分割器
python test_heading_splitter.py
```

## 🎯 支持的模型

### 嵌入模型

- **BGE-M3**: `bge-m3:latest` (推荐，支持多语言)
- **BGE-Small**: `BAAI/bge-small-zh-v1.5` (中文专用)
- **MiniLM**: `sentence-transformers/all-MiniLM-L6-v2` (轻量级)
- **多语言**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### 语言模型

#### 本地模型（Ollama）
- `llama3:8b` - Meta Llama 3
- `qwen2.5:7b` - 阿里通义千问
- `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B` - DeepSeek R1

#### 云端模型
- OpenAI: `gpt-4`, `gpt-3.5-turbo`
- SiliconFlow: 支持多种开源模型

## ⚙️ 配置说明

### 向量数据库配置

向量数据库默认存储在 `./chroma_db` 目录，包含以下内容：

- 文档向量
- 元数据信息
- 索引文件

### 文本分割配置

- `max_chunk_size`: 文本块最大字符数（默认 1500）
- `chunk_overlap`: 文本块重叠大小（默认 100）

### 检索配置

- `k`: 检索的文档数量（默认 5）
- 相似度阈值：自动计算，返回最相关的文档

## 🔍 调试技巧

### 使用 Jupyter Notebook

```bash
jupyter notebook notebooks/debug_retrieval.ipynb
```

在笔记本中可以：
- 可视化检索结果
- 调试各个组件
- 测试不同的参数

### 查看向量数据库信息

```python
info = store.get_collection_info()
print(f"文档数量: {info['count']}")
```

### 检查嵌入向量

```python
vector = embedder.embed_query("测试文本")
print(f"向量维度: {len(vector)}")
```

## ⚠️ 注意事项

1. **环境变量**: 请勿将 `.env` 文件提交到版本控制
2. **向量数据库**: 向量数据库文件会在 `chroma_db/` 目录生成
3. **模型下载**: 首次使用嵌入模型时会自动下载，需要网络连接
4. **内存使用**: 大文档处理时注意内存使用，可以调整批处理大小
5. **GPU 支持**: 如有 GPU，可以设置 `device="cuda"` 提升性能

## 🚧 已知问题

- ChromaDB 遥测功能可能导致错误，已在代码中禁用
- 大文档处理可能需要较长时间，建议使用 SSD 存储
- 某些 PDF 文档可能需要特殊处理

## 🤝 贡献指南

欢迎贡献代码、报告问题或提出建议！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 LICENSE 文件

## 📞 联系方式

- 项目主页: https://github.com/yourusername/rag-demo
- 问题反馈: https://github.com/yourusername/rag-demo/issues

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 强大的 LLM 应用开发框架
- [ChromaDB](https://github.com/chroma-core/chroma) - 开源向量数据库
- [HuggingFace](https://huggingface.co/) - 丰富的预训练模型
- [Ollama](https://ollama.ai/) - 本地 LLM 运行工具

---

**注意**: 本项目仅用于学习和演示目的，生产环境使用请根据实际需求进行调整和优化。
