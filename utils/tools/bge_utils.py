"""
BGE Embedding Utilities

A utility class for computing text embeddings using BGE (BAAI General Embedding) models.
Supports FlagEmbedding library for semantic vector computation.
"""

import numpy as np
from typing import List, Optional
from pathlib import Path
import os


# Check if FlagEmbedding is available
try:
    from FlagEmbedding import FlagModel
    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False
    FlagModel = None


class BGEEmbeddingUtils:
    """
    Utility class for BGE model embeddings.
    Provides lazy loading and text embedding computation.
    """
    
    # Default model path (relative to project root)
    _PROJECT_ROOT = Path(__file__).parent.parent.parent
    _DEFAULT_MODEL_PATH = _PROJECT_ROOT / 'component' / 'bge-base-en-v1.5'
    
    # Model instance (lazy loading)
    _model_instance: Optional[FlagModel] = None
    _model_path: Optional[Path] = None
    
    @classmethod
    def _get_model(cls, model_path: Optional[Path] = None) -> FlagModel:
        """
        Get or initialize the FlagModel instance (lazy loading).
        
        Args:
            model_path: Optional path to model directory. If None, uses default path.
            
        Returns:
            FlagModel instance
            
        Raises:
            ImportError: If FlagEmbedding is not available
            FileNotFoundError: If model path does not exist
        """
        if not FLAG_EMBEDDING_AVAILABLE:
            raise ImportError(
                "FlagEmbedding is not available. "
                "Please install it with: pip install FlagEmbedding"
            )
        
        # Use provided path or default path
        if model_path is None:
            model_path = cls._DEFAULT_MODEL_PATH
        else:
            model_path = Path(model_path)
        
        # Check if we need to reload the model
        if cls._model_instance is None or cls._model_path != model_path:
            model_path_abs = model_path.resolve()
            if not model_path_abs.exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path_abs}")
            
            cls._model_instance = FlagModel(
                str(model_path_abs),
                query_instruction_for_retrieval="Generate embeddings for the following sentences for retrieval:",
                use_fp16=True
            )
            cls._model_path = model_path
        
        return cls._model_instance
    
    @classmethod
    def compute_embeddings(
        cls,
        texts: List[str],
        model_path: Optional[Path] = None,
        normalize: bool = True,
        fallback_to_random: bool = True
    ) -> List[np.ndarray]:
        """
        Compute semantic vectors for a list of texts using BGE model.
        
        Args:
            texts: List of text strings to embed
            model_path: Optional path to model directory. If None, uses default path.
            normalize: Whether to normalize embeddings to unit vectors (default: True)
            fallback_to_random: Whether to use random embeddings if model fails (default: True)
            
        Returns:
            List of numpy arrays representing embeddings (normalized to unit vectors if normalize=True)
        """
        if not texts:
            return []
        
        if FLAG_EMBEDDING_AVAILABLE:
            try:
                model = cls._get_model(model_path)
                embeddings = model.encode(texts)
                embeddings_list = []
                for emb in embeddings:
                    emb_array = np.array(emb)
                    if normalize:
                        emb_array = emb_array / np.linalg.norm(emb_array)
                    embeddings_list.append(emb_array)
                return embeddings_list
            except Exception as e:
                if fallback_to_random:
                    print(f"Warning: Failed to load FlagModel, using placeholder embeddings: {e}")
                else:
                    raise
        
        # Fallback to random embeddings if model is not available or failed
        if fallback_to_random:
            embeddings = []
            for text in texts:
                embedding = np.random.rand(768)
                if normalize:
                    embedding = embedding / np.linalg.norm(embedding)
                embeddings.append(embedding)
            return embeddings
        else:
            raise RuntimeError("BGE model is not available and fallback is disabled")
    
    @classmethod
    def compute_embedding(
        cls,
        text: str,
        model_path: Optional[Path] = None,
        normalize: bool = True,
        fallback_to_random: bool = True
    ) -> np.ndarray:
        """
        Compute semantic vector for a single text using BGE model.
        
        Args:
            text: Text string to embed
            model_path: Optional path to model directory. If None, uses default path.
            normalize: Whether to normalize embedding to unit vector (default: True)
            fallback_to_random: Whether to use random embedding if model fails (default: True)
            
        Returns:
            Numpy array representing the embedding (normalized to unit vector if normalize=True)
        """
        embeddings = cls.compute_embeddings(
            [text],
            model_path=model_path,
            normalize=normalize,
            fallback_to_random=fallback_to_random
        )
        if embeddings:
            return embeddings[0]
        else:
            # Fallback
            embedding = np.random.rand(768)
            if normalize:
                embedding = embedding / np.linalg.norm(embedding)
            return embedding
    
    @classmethod
    def is_available(cls) -> bool:
        """
        Check if BGE embedding is available.
        
        Returns:
            True if FlagEmbedding is available, False otherwise
        """
        return FLAG_EMBEDDING_AVAILABLE
    
    @classmethod
    def reset_model(cls) -> None:
        """
        Reset the cached model instance. Useful for reloading with a different model path.
        """
        cls._model_instance = None
        cls._model_path = None
