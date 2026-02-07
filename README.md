# RAG Demo 项目

这是一个用于学习和演示 RAG (Retrieval-Augmented Generation) 技术的项目。

## 项目结构

```
rag-demo/
├── data/                    # 存放测试文档
│   └── sample.pdf          # 你的测试PDF
├── src/
│   ├── loaders/            # 文档加载模块
│   ├── splitters/          # 文本分割模块  
│   ├── embeddings/         # 向量化模块
│   ├── vectorstore/        # 向量数据库模块
│   ├── chains/             # LangChain链定义
│   └── main.py             # 入口文件
├── notebooks/              # 调试笔记本（重要！）
│   └── debug_retrieval.ipynb
├── .env                    # 环境变量
└── requirements.txt
```

## 快速开始

### 1. 安装依赖

```bash
cd rag-demo
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env` 文件并填入你的 API 密钥：

```bash
cp .env .env.local
# 编辑 .env.local，填入你的 OPENAI_API_KEY
```

### 3. 添加测试文档

将你的 PDF 文件放入 `data/` 目录。

### 4. 运行调试笔记本

```bash
jupyter notebook notebooks/debug_retrieval.ipynb
```

### 5. 运行主程序

```bash
python src/main.py
```

## 开发流程

1. **文档加载** (`src/loaders/`) - 实现 PDF、文本等文档的加载
2. **文本分割** (`src/splitters/`) - 将长文档切分成适合处理的块
3. **向量化** (`src/embeddings/`) - 将文本转换为向量表示
4. **向量存储** (`src/vectorstore/`) - 存储和检索向量数据
5. **链定义** (`src/chains/`) - 组合所有组件形成 RAG 流程

## 注意事项

- 请勿将 `.env.local` 文件提交到版本控制
- 向量数据库文件会在 `chroma_db/` 目录生成
- 使用笔记本进行组件级别的调试和测试
