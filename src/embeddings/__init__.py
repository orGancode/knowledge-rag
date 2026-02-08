# Embeddings Module

from abc import ABC, abstractmethod
from typing import List, Optional
from langchain_ollama import OllamaEmbeddings as LangChainOllamaEmbeddings


class BaseEmbeddings(ABC):
    """Base class for embeddings."""
    
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        pass
    
    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a query text."""
        pass


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embeddings implementation."""
    
    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using OpenAI."""
        # TODO: Implement OpenAI embeddings
        pass
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using OpenAI."""
        # TODO: Implement OpenAI query embedding
        pass


class OllamaEmbeddings(BaseEmbeddings):
    """Ollama embeddings implementation for local models.
    
    For embedding functionality, use models like 'nomic-embed-text' or other embedding-specific models.
    """
    
    def __init__(self, model: str = "nomic-embed-text", base_url: Optional[str] = None):
        """
        Initialize Ollama embeddings.
        
        Args:
            base_url: Base URL for Ollama API (default: http://localhost:11434)
        """
        self.model = model
        self.base_url = base_url or "http://localhost:11434"
        self._client = LangChainOllamaEmbeddings(
            model=model,
            base_url=self.base_url
        )
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using Ollama."""
        return self._client.embed_documents(texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query using Ollama."""
        return self._client.embed_query(text)


class HuggingFaceEmbeddings(BaseEmbeddings):
    """HuggingFace embeddings implementation with advanced features."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "auto",
        cache_folder: Optional[str] = None,
        model_kwargs: Optional[dict] = None,
        encode_kwargs: Optional[dict] = None,
        multi_process: bool = False,
        show_progress: bool = True
    ):
        """
        Initialize HuggingFace embeddings.
        
        Args:
            model_name: Model name from HuggingFace Hub
            device: Device to run model on ("cpu", "cuda", "auto")
            cache_folder: Path to cache models
            model_kwargs: Additional kwargs for model initialization
            encode_kwargs: Additional kwargs for encoding
            multi_process: Whether to use multiple processes for encoding
            show_progress: Whether to show progress bar
        """
        self.model_name = model_name
        self.device = self._determine_device(device)
        self.cache_folder = cache_folder
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.multi_process = multi_process
        self.show_progress = show_progress
        
        # Lazy loading - model will be loaded when first used
        self._model = None
        self._tokenizer = None
    
    def _determine_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_model(self):
        """Load the model and tokenizer if not already loaded."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                import warnings
                
                # 过滤 huggingface_hub 的 FutureWarning
                warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
                
                # Set default model kwargs
                default_model_kwargs = {
                    'device': self.device,
                    'cache_folder': self.cache_folder,
                }
                default_model_kwargs.update(self.model_kwargs)
                
                # Load model
                self._model = SentenceTransformer(
                    self.model_name,
                    **default_model_kwargs
                )
                
                # Set default encode kwargs
                default_encode_kwargs = {
                    'batch_size': 32,
                    'show_progress_bar': self.show_progress,
                    'normalize_embeddings': True,
                }
                default_encode_kwargs.update(self.encode_kwargs)
                
                print(f"✅ 成功加载模型: {self.model_name} (设备: {self.device})")
                
            except ImportError as e:
                raise ImportError(
                    "请安装 sentence-transformers 库: pip install sentence-transformers"
                ) from e
            except Exception as e:
                raise RuntimeError(f"加载模型失败: {str(e)}") from e
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents using HuggingFace models.
        
        Args:
            texts: List of document texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        self._load_model()
        
        try:
            # Convert to list if it's not already
            if not isinstance(texts, list):
                texts = list(texts)
            
            # Encode documents
            embeddings = self._model.encode(
                texts,
                **self.encode_kwargs
            )
            
            # Convert to list of lists if needed
            if hasattr(embeddings, 'tolist'):
                embeddings = embeddings.tolist()
            
            return embeddings
            
        except Exception as e:
            raise RuntimeError(f"文档嵌入失败: {str(e)}") from e
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text using HuggingFace models.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        if not text:
            return []
        
        self._load_model()
        
        try:
            # Encode query
            embedding = self._model.encode(
                text,
                **self.encode_kwargs
            )
            
            # Convert to list if needed
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            raise RuntimeError(f"查询嵌入失败: {str(e)}") from e
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Cosine similarity score
        """
        try:
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
            
            emb1 = self.embed_query(text1)
            emb2 = self.embed_query(text2)
            
            # Reshape for cosine_similarity
            emb1 = np.array(emb1).reshape(1, -1)
            emb2 = np.array(emb2).reshape(1, -1)
            
            return float(cosine_similarity(emb1, emb2)[0][0])
            
        except ImportError:
            raise ImportError(
                "请安装 numpy 和 scikit-learn 库: pip install numpy scikit-learn"
            )
        except Exception as e:
            raise RuntimeError(f"相似度计算失败: {str(e)}") from e
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        self._load_model()
        
        try:
            # Get embedding dimension
            sample_embedding = self.embed_query("test")
            embedding_dimension = len(sample_embedding)
            
            return {
                "model_name": self.model_name,
                "device": self.device,
                "embedding_dimension": embedding_dimension,
                "max_sequence_length": getattr(self._model, 'max_seq_length', 'Unknown'),
                "cache_folder": self.cache_folder,
            }
        except Exception as e:
            return {"error": f"获取模型信息失败: {str(e)}"}
