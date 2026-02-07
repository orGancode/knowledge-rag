# HuggingFaceEmbeddings 使用指南

## 概述

`HuggingFaceEmbeddings` 是一个基于 HuggingFace 模型的向量化实现，提供了丰富的功能和灵活的配置选项。

## 主要功能

### 1. 基础向量化功能
- `embed_documents()`: 批量文档向量化
- `embed_query()`: 单个查询文本向量化

### 2. 高级功能
- **自动设备选择**: 自动检测并使用可用的 GPU 或 CPU
- **模型缓存**: 支持模型缓存，避免重复下载
- **批处理优化**: 高效的批量处理，支持自定义批大小
- **文本相似度计算**: 内置余弦相似度计算功能
- **模型信息获取**: 获取模型详细信息
- **多语言支持**: 支持多语言模型，包括中文

## 使用方法

### 基础使用

```python
from src.embeddings import HuggingFaceEmbeddings

# 初始化模型
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 单个文本向量化
text = "这是一个测试文本"
vector = embeddings.embed_query(text)
print(f"向量维度: {len(vector)}")

# 批量文档向量化
documents = ["文档1", "文档2", "文档3"]
vectors = embeddings.embed_documents(documents)
print(f"文档数量: {len(vectors)}")
```

### 高级配置

```python
# 使用高级配置
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    device="auto",  # 自动选择设备
    cache_folder="./models_cache",  # 模型缓存目录
    encode_kwargs={
        "batch_size": 16,  # 批大小
        "show_progress_bar": True,  # 显示进度条
        "normalize_embeddings": True  # 归一化嵌入
    }
)
```

### 文本相似度计算

```python
# 计算两个文本的相似度
similarity = embeddings.similarity(
    "机器学习是人工智能的一个分支",
    "深度学习是机器学习的一个子集"
)
print(f"相似度分数: {similarity:.4f}")
```

### 获取模型信息

```python
# 获取模型详细信息
model_info = embeddings.get_model_info()
print(model_info)
# 输出示例:
# {
#     "model_name": "sentence-transformers/all-MiniLM-L6-v2",
#     "device": "cpu",
#     "embedding_dimension": 384,
#     "max_sequence_length": 256,
#     "cache_folder": None
# }
```

## 推荐模型

### 英文模型
- `sentence-transformers/all-MiniLM-L6-v2`: 轻量级英文模型，384维
- `sentence-transformers/all-mpnet-base-v2`: 高质量英文模型，768维

### 多语言模型
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: 多语言模型，支持中文
- `sentence-transformers/distiluse-base-multilingual-cased-v1`: 多语言模型

### 中文专用模型
- `shibing624/text2vec-base-chinese`: 中文专用模型
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`: 对中文支持良好

## 性能优化建议

1. **设备选择**: 如果有 GPU，设置 `device="cuda"` 可以显著提升性能
2. **批处理**: 对于大量文档，适当调整 `batch_size` 可以平衡内存使用和速度
3. **模型缓存**: 设置 `cache_folder` 避免重复下载模型
4. **模型选择**: 根据任务需求选择合适大小的模型，平衡质量和速度

## 错误处理

```python
try:
    embeddings = HuggingFaceEmbeddings(model_name="invalid-model-name")
    vector = embeddings.embed_query("test")
except ImportError as e:
    print(f"依赖库缺失: {e}")
except RuntimeError as e:
    print(f"模型加载失败: {e}")
```

## 依赖要求

- `sentence-transformers>=3.0.0`
- `torch>=2.2.0`
- `transformers>=4.40.0`
- `numpy>=1.24.0`
- `scikit-learn>=1.3.0` (用于相似度计算)

## 示例代码

完整示例请参考:
- `example_embeddings_usage.py`: 基础和高级功能示例
- `test_embeddings.py`: 功能测试代码

## 常见问题

### Q: 如何切换到 GPU？
A: 设置 `device="cuda"` 或 `device="auto"` (自动检测)

### Q: 如何处理中文文本？
A: 使用多语言模型如 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

### Q: 模型下载很慢怎么办？
A: 设置 `cache_folder` 参数，模型只需下载一次

### Q: 如何提高批处理速度？
A: 适当增加 `batch_size`，但注意不要超过内存限制