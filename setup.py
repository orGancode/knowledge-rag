# setup.py
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="rag-demo",
    version="0.1.0",
    author="RAG Demo Team",
    author_email="team@ragdemo.com",
    description="一个用于学习和演示 RAG (Retrieval-Augmented Generation) 技术的项目",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/rag-demo",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        # 核心框架
        "langchain==0.3.0",
        "langchain-community==0.3.0",
        "langchain-ollama==0.2.0",
        
        # 文档解析
        "pypdf==4.2.0",
        "python-docx==1.1.0",
        "unstructured==0.14.0",
        
        # 向量数据库
        "chromadb==0.5.0",
        
        # Embedding模型
        "sentence-transformers==3.0.0",
        "torch==2.2.0",
        "transformers==4.40.0",
        
        # 环境变量
        "python-dotenv>=1.0.0",
        
        # 工具依赖
        "tiktoken>=0.5.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "tqdm==4.66.0",
        
        # PDF处理
        "pypdf",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipython>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rag-demo=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
)