# ChromaVectorStore 使用指南

## 概述

ChromaVectorStore 是基于 ChromaDB 实现的本地向量数据库模块，用于存储和检索文档的向量表示。它是 RAG 系统中的关键组件，负责高效地存储文档向量并支持相似度搜索。

## 主要特性

- **本地存储**: 数据存储在本地磁盘，无需外部服务
- **自动持久化**: 支持数据的自动和手动持久化
- **相似度搜索**: 基于余弦相似度的高效向量检索
- **元数据支持**: 保留文档的原始元数据信息
- **集合管理**: 支持多个命名集合的创建和管理

## 数据模型

ChromaVectorStore 使用以下数据结构存储文档：

```python
{
    "ids": ["chunk_0", "chunk_1", ...],           # 唯一ID
    "embeddings": [[0.1, 0.2, ...], ...],         # 向量
    "documents": ["文本内容1", "文本内容2", ...],  # 原始文本
    "metadatas": [
        {"source": "sample.pdf", "page": 1, "chunk_index": 0},
        ...
    ]
}
```

## 核心接口

### 初始化

```python
from src.vectorstore import ChromaVectorStore

vector_store = ChromaVectorStore(
    collection_name="my_collection",
    persist_directory="./chroma_db"
)
```

### 创建集合

```python
# 需要提供一个嵌入函数
vector_store.create_collection(embedding_function=my_embedding_function)
```

### 添加文档

```python
from langchain.schema import Document

documents = [
    Document(page_content="文档内容1", metadata={"source": "doc1.pdf"}),
    Document(page_content="文档内容2", metadata={"source": "doc2.pdf"})
]

embeddings = [
    [0.1, 0.2, 0.3],  # 对应第一个文档的向量
    [0.4, 0.5, 0.6]   # 对应第二个文档的向量
]

vector_store.add_documents(documents, embeddings)
```

### 相似度搜索

```python
query_embedding = [0.2, 0.3, 0.4]  # 查询向量
results = vector_store.similarity_search(query_embedding, k=5)

for result in results:
    print(f"文档ID: {result['id']}")
    print(f"内容: {result['document']}")
    print(f"元数据: {result['metadata']}")
    print(f"相似度分数: {result['similarity_score']}")
```

### 持久化

```python
# 显式持久化（ChromaDB会自动持久化，这里主要用于确认）
vector_store.persist()
```

### 获取集合信息

```python
info = vector_store.get_collection_info()
print(f"集合名称: {info['name']}")
print(f"文档数量: {info['count']}")
print(f"存储路径: {info['persist_directory']}")
```

## 完整示例

```python
#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from langchain.schema import Document
from src.vectorstore import ChromaVectorStore
from src.embeddings import HuggingFaceEmbeddings

def main():
    # 1. 初始化嵌入模型
    embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    
    # 2. 创建测试文档
    documents = [
        Document(
            page_content="人工智能是计算机科学的一个分支。",
            metadata={"source": "test.txt", "page": 1}
        ),
        Document(
            page_content="机器学习是人工智能的一个子集。",
            metadata={"source": "test.txt", "page": 2}
        )
    ]
    
    # 3. 生成嵌入向量
    texts = [doc.page_content for doc in documents]
    embeddings = embedding_model.embed_documents(texts)
    
    # 4. 初始化向量存储
    vector_store = ChromaVectorStore(
        collection_name="example_collection",
        persist_directory="./chroma_db"
    )
    
    # 5. 创建集合
    vector_store.create_collection(embedding_function=embedding_model.embed_query)
    
    # 6. 添加文档
    vector_store.add_documents(documents, embeddings)
    
    # 7. 执行搜索
    query_text = "什么是机器学习？"
    query_embedding = embedding_model.embed_query(query_text)
    results = vector_store.similarity_search(query_embedding, k=2)
    
    # 8. 显示结果
    for result in results:
        print(f"内容: {result['document']}")
        print(f"相似度: {result['similarity_score']}")
    
    # 9. 持久化
    vector_store.persist()

if __name__ == "__main__":
    main()
```

## 注意事项

1. **嵌入函数兼容性**: ChromaVectorStore 内部使用适配器将 LangChain 风格的嵌入函数转换为 ChromaDB 兼容的格式。

2. **唯一ID**: 每个文档块都会自动生成唯一ID，格式为 `chunk_{uuid}`。

3. **元数据保留**: 原始文档的元数据会被完整保留，并自动添加 `chunk_index` 字段。

4. **相似度分数**: 返回的相似度分数范围是 0-1，越接近 1 表示越相似。

5. **持久化目录**: 指定的持久化目录会自动创建，如果不存在的话。

## 性能优化建议

1. **批量操作**: 尽量批量添加文档，而不是逐个添加。

2. **合适的块大小**: 根据文档内容和查询需求选择合适的文本块大小。

3. **向量维度**: 使用与嵌入模型匹配的向量维度。

4. **存储位置**: 将持久化目录放在快速的存储设备上以提高性能。

## 故障排除

### 常见问题

1. **嵌入函数错误**: 确保提供的嵌入函数接受单个文本字符串并返回向量列表。

2. **维度不匹配**: 确保所有向量的维度一致。

3. **权限问题**: 确保对持久化目录有读写权限。

### 调试技巧

1. 使用 `get_collection_info()` 检查集合状态。

2. 检查返回的相似度分数是否合理。

3. 验证元数据是否正确保留。

## 扩展功能

ChromaVectorStore 还提供了以下扩展功能：

- `delete_collection()`: 删除整个集合
- 支持自定义距离度量（默认使用余弦相似度）
- 支持集合元数据配置

这些功能可以根据具体需求进行使用和扩展。