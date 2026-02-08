# BasicQAChain 使用指南

## 概述

`BasicQAChain` 是一个基础问答链实现，用于构建检索增强生成（RAG）系统。它结合了文档检索和语言模型生成，能够基于提供的文档内容回答用户问题。

## 功能特点

- 支持本地和云端语言模型
- 灵活的检索器接口
- 完善的错误处理机制
- 支持源文档追踪
- 无记忆设计（每次查询独立）

## 安装依赖

确保已安装以下依赖：

```bash
pip install langchain langchain-openai langchain-ollama langchain-huggingface
pip install sentence-transformers chromadb
```

## 基本使用

### 1. 初始化问答链

```python
from chains.qa_chain import BasicQAChain
from embeddings import HuggingFaceEmbeddings
from vectorstore.chroma_store import ChromaVectorStore

# 创建检索器函数
def retriever_function(query, k=3):
    query_embedding = embeddings.embed_query(query)
    return vectorstore.similarity_search(query_embedding, k=k)

# 使用本地模型（Ollama）
qa_chain = BasicQAChain(
    retriever=retriever_function, 
    use_local=True, 
    llm_model="llama3:8b"
)

# 使用云端模型（OpenAI）
qa_chain = BasicQAChain(
    retriever=retriever_function, 
    use_local=False, 
    llm_model="gpt-3.5-turbo"
)
```

### 2. 执行问答

```python
# 提出问题
question = "什么是人工智能？"

# 获取答案
result = qa_chain.run(question)

# 查看结果
print(f"问题: {result['question']}")
print(f"答案: {result['answer']}")
print(f"源文档数量: {len(result['source_documents'])}")

# 查看源文档
for doc in result['source_documents']:
    print(f"内容: {doc['content']}")
    print(f"相似度: {doc['similarity_score']}")
    print(f"元数据: {doc['metadata']}")
```

## 完整示例

### 创建问答系统

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from chains.qa_chain import BasicQAChain
from embeddings import HuggingFaceEmbeddings
from vectorstore.chroma_store import ChromaVectorStore
from langchain.schema import Document

# 1. 初始化组件
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = ChromaVectorStore(
    collection_name="my_collection",
    persist_directory="./db"
)
vectorstore.create_collection(embeddings.embed_query)

# 2. 准备文档
documents = [
    Document(
        page_content="人工智能是计算机科学的一个分支...",
        metadata={"source": "doc.pdf", "page": 1}
    ),
    # 更多文档...
]

# 3. 处理文档
doc_texts = [doc.page_content for doc in documents]
doc_embeddings = embeddings.embed_documents(doc_texts)
vectorstore.add_documents(documents, doc_embeddings)

# 4. 创建检索器
def retriever_function(query, k=3):
    query_embedding = embeddings.embed_query(query)
    return vectorstore.similarity_search(query_embedding, k=k)

# 5. 创建问答链
qa_chain = BasicQAChain(
    retriever=retriever_function,
    use_local=True,
    llm_model="llama3:8b"
)

# 6. 使用问答链
result = qa_chain.run("什么是人工智能？")
print(result['answer'])
```

## 检索器接口

`BasicQAChain` 支持两种类型的检索器：

### 1. 对象检索器

```python
class MyRetriever:
    def similarity_search(self, query, k=4):
        # 实现检索逻辑
        return results

retriever = MyRetriever()
qa_chain = BasicQAChain(retriever=retriever)
```

### 2. 函数检索器

```python
def my_retriever_function(query, k=4):
    # 实现检索逻辑
    return results

qa_chain = BasicQAChain(retriever=my_retriever_function)
```

## 返回结果格式

`qa_chain.run()` 方法返回一个字典，包含以下字段：

```python
{
    "question": "用户问题",
    "answer": "生成的答案",
    "source_documents": [
        {
            "content": "文档内容",
            "metadata": {"source": "doc.pdf", "page": 1},
            "similarity_score": 0.95
        },
        # 更多源文档...
    ]
}
```

## 提示模板

默认提示模板包含以下要求：

1. 只基于提供的文档内容回答
2. 如果文档中没有相关信息，明确说明
3. 回答简洁准确，控制在200字以内
4. 在回答末尾列出参考的文档页码

## 错误处理

`BasicQAChain` 内置了错误处理机制：

- 检索器错误会被捕获并返回错误信息
- LLM 调用错误会被捕获并返回错误信息
- 所有错误都会在 `answer` 字段中返回，格式为："处理问题时发生错误: {错误详情}"

## 性能优化建议

1. **批量处理**：对于大量文档，考虑批量生成嵌入向量
2. **缓存机制**：对频繁查询的结果进行缓存
3. **并行处理**：使用多线程/多进程处理文档
4. **模型选择**：根据需求选择合适的嵌入模型和语言模型

## 常见问题

### Q: 如何更换语言模型？

A: 修改 `llm_model` 参数和 `use_local` 设置：

```python
# 使用不同的本地模型
qa_chain = BasicQAChain(retriever=retriever, use_local=True, llm_model="qwen2.5:7b")

# 使用不同的云端模型
qa_chain = BasicQAChain(retriever=retriever, use_local=False, llm_model="gpt-4")
```

### Q: 如何调整检索的文档数量？

A: 在检索器函数中修改 `k` 参数：

```python
def retriever_function(query, k=5):  # 检索5个文档
    query_embedding = embeddings.embed_query(query)
    return vectorstore.similarity_search(query_embedding, k=k)
```

### Q: 如何自定义提示模板？

A: 修改 `_create_prompt` 方法中的模板内容：

```python
def _create_prompt(self):
    template = """
    自定义提示模板...
    {context}
    {question}
    """
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )
```

## 扩展功能

`BasicQAChain` 是一个基础实现，可以在此基础上扩展：

1. **记忆功能**：添加对话历史记录
2. **多轮对话**：支持上下文相关的多轮问答
3. **文档过滤**：根据元数据过滤检索结果
4. **答案评分**：对生成的答案进行质量评分
5. **多语言支持**：支持多语言问答

## 测试

运行测试脚本验证功能：

```bash
python test_qa_chain.py
```

## 示例代码

完整的使用示例请参考 `example_qa_chain_usage.py` 文件。