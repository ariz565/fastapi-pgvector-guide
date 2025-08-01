# Embedding Generation Functions
# Clean function-based approach for text embedding generation

import logging
import numpy as np
from typing import List, Optional, Union
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache to avoid reloading
_embedding_model = None
_model_name = None

def initialize_embedding_model(model_name: str = 'all-MiniLM-L6-v2') -> bool:
    """
    Initialize the sentence transformer model for embedding generation
    
    Args:
        model_name: Name of the sentence transformer model to use
        
    Returns:
        True if successful, False otherwise
    """
    global _embedding_model, _model_name
    
    try:
        # Check if model is already loaded
        if _embedding_model is not None and _model_name == model_name:
            logger.info(f"✅ Embedding model '{model_name}' already loaded")
            return True
        
        # Try to import sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"🔄 Loading embedding model: {model_name}")
            
            # Load the model
            _embedding_model = SentenceTransformer(model_name)
            _model_name = model_name
            
            # Test the model with a simple example
            test_embedding = _embedding_model.encode("test sentence")
            embedding_dim = len(test_embedding)
            
            logger.info(f"✅ Embedding model loaded successfully")
            logger.info(f"   Model: {model_name}")
            logger.info(f"   Embedding dimension: {embedding_dim}")
            
            return True
            
        except ImportError:
            logger.warning("⚠️ sentence-transformers not available, falling back to TF-IDF")
            return _initialize_tfidf_fallback()
            
    except Exception as e:
        logger.error(f"❌ Error loading embedding model: {e}")
        return False

def _initialize_tfidf_fallback() -> bool:
    """
    Initialize TF-IDF as fallback when sentence-transformers is not available
    
    Returns:
        True if successful, False otherwise
    """
    global _embedding_model, _model_name
    
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        
        # Create a TF-IDF vectorizer with fixed vocabulary size
        _embedding_model = TfidfVectorizer(
            max_features=384,  # Match sentence-transformer dimension
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        _model_name = "tfidf_fallback"
        
        logger.info("✅ TF-IDF fallback model initialized")
        logger.info("   Embedding dimension: 384 (TF-IDF)")
        
        return True
        
    except ImportError:
        logger.error("❌ Both sentence-transformers and scikit-learn are unavailable")
        return False

def generate_embedding(text: str) -> Optional[np.ndarray]:
    """
    Generate embedding for a single text
    
    Args:
        text: Input text to encode
        
    Returns:
        Numpy array containing the embedding, or None if failed
    """
    global _embedding_model, _model_name
    
    if _embedding_model is None:
        logger.warning("⚠️ Embedding model not initialized, initializing now...")
        if not initialize_embedding_model():
            return None
    
    try:
        # Preprocess text
        cleaned_text = preprocess_text(text)
        if not cleaned_text:
            logger.warning("⚠️ Empty text after preprocessing")
            return None
        
        # Generate embedding based on model type
        if _model_name == "tfidf_fallback":
            return _generate_tfidf_embedding(cleaned_text)
        else:
            return _generate_transformer_embedding(cleaned_text)
            
    except Exception as e:
        logger.error(f"❌ Error generating embedding: {e}")
        return None

def _generate_transformer_embedding(text: str) -> np.ndarray:
    """
    Generate embedding using sentence transformer model
    
    Args:
        text: Preprocessed text
        
    Returns:
        Numpy array containing the embedding
    """
    embedding = _embedding_model.encode(text, convert_to_numpy=True)
    
    # Normalize the embedding for cosine similarity
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    
    return embedding.astype(np.float32)

def _generate_tfidf_embedding(text: str) -> np.ndarray:
    """
    Generate embedding using TF-IDF fallback
    
    Args:
        text: Preprocessed text
        
    Returns:
        Numpy array containing the TF-IDF embedding
    """
    # For TF-IDF, we need to fit on some data first
    # This is a simplified approach - in production, you'd fit on your corpus
    try:
        # Transform the text
        tfidf_matrix = _embedding_model.transform([text])
        embedding = tfidf_matrix.toarray()[0]
        
        # Pad or truncate to ensure consistent dimension
        if len(embedding) < 384:
            embedding = np.pad(embedding, (0, 384 - len(embedding)))
        elif len(embedding) > 384:
            embedding = embedding[:384]
        
        return embedding.astype(np.float32)
        
    except:
        # If transform fails (model not fitted), fit on the single text
        _embedding_model.fit([text])
        return _generate_tfidf_embedding(text)

def generate_embeddings_batch(texts: List[str], batch_size: int = 32) -> List[Optional[np.ndarray]]:
    """
    Generate embeddings for multiple texts efficiently
    
    Args:
        texts: List of texts to encode
        batch_size: Number of texts to process at once
        
    Returns:
        List of embeddings (or None for failed texts)
    """
    global _embedding_model, _model_name
    
    if _embedding_model is None:
        if not initialize_embedding_model():
            return [None] * len(texts)
    
    embeddings = []
    
    try:
        # Process texts in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Preprocess batch
            cleaned_batch = [preprocess_text(text) for text in batch_texts]
            
            # Filter out empty texts
            valid_texts = [(idx, text) for idx, text in enumerate(cleaned_batch) if text]
            
            if not valid_texts:
                # All texts in batch are empty
                embeddings.extend([None] * len(batch_texts))
                continue
            
            # Generate embeddings for valid texts
            batch_indices, valid_text_list = zip(*valid_texts)
            
            if _model_name == "tfidf_fallback":
                batch_embeddings = [_generate_tfidf_embedding(text) for text in valid_text_list]
            else:
                batch_embeddings = _embedding_model.encode(
                    list(valid_text_list), 
                    convert_to_numpy=True,
                    batch_size=min(batch_size, len(valid_text_list))
                )
                
                # Normalize embeddings
                batch_embeddings = [
                    emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb 
                    for emb in batch_embeddings
                ]
            
            # Map back to original batch order
            batch_result = [None] * len(batch_texts)
            for batch_idx, embedding in zip(batch_indices, batch_embeddings):
                batch_result[batch_idx] = embedding.astype(np.float32)
            
            embeddings.extend(batch_result)
            
            # Log progress for large batches
            if len(texts) > 50:
                processed = min(i + batch_size, len(texts))
                logger.info(f"🔄 Generated embeddings: {processed}/{len(texts)}")
        
        logger.info(f"✅ Generated {sum(1 for emb in embeddings if emb is not None)}/{len(texts)} embeddings")
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Error in batch embedding generation: {e}")
        return [None] * len(texts)

def preprocess_text(text: str) -> str:
    """
    Preprocess text before embedding generation
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned and preprocessed text
    """
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Basic cleaning
    text = text.strip()
    
    # Remove excessive whitespace
    import re
    text = re.sub(r'\s+', ' ', text)
    
    # Remove very short texts (less than 3 characters)
    if len(text) < 3:
        return ""
    
    # Truncate very long texts (over 8000 characters for efficiency)
    if len(text) > 8000:
        text = text[:8000] + "..."
        logger.warning(f"⚠️ Text truncated to 8000 characters")
    
    return text

def get_embedding_model_info() -> dict:
    """
    Get information about the currently loaded embedding model
    
    Returns:
        Dictionary containing model information
    """
    global _embedding_model, _model_name
    
    if _embedding_model is None:
        return {
            'loaded': False,
            'model_name': None,
            'embedding_dimension': None,
            'model_type': None
        }
    
    # Determine embedding dimension
    if _model_name == "tfidf_fallback":
        embedding_dim = 384
        model_type = "TF-IDF"
    else:
        try:
            # Test with a sample text to get dimension
            test_embedding = _embedding_model.encode("test")
            embedding_dim = len(test_embedding)
            model_type = "Sentence Transformer"
        except:
            embedding_dim = "Unknown"
            model_type = "Unknown"
    
    return {
        'loaded': True,
        'model_name': _model_name,
        'embedding_dimension': embedding_dim,
        'model_type': model_type
    }

def calculate_text_similarity(text1: str, text2: str) -> Optional[float]:
    """
    Calculate cosine similarity between two texts using embeddings
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Cosine similarity score (0.0 to 1.0), or None if failed
    """
    try:
        # Generate embeddings for both texts
        embedding1 = generate_embedding(text1)
        embedding2 = generate_embedding(text2)
        
        if embedding1 is None or embedding2 is None:
            return None
        
        # Calculate cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure similarity is in [0, 1] range
        similarity = max(0.0, min(1.0, similarity))
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"❌ Error calculating text similarity: {e}")
        return None

def benchmark_embedding_generation(num_samples: int = 100) -> dict:
    """
    Benchmark embedding generation performance
    
    Args:
        num_samples: Number of sample texts to process
        
    Returns:
        Dictionary containing benchmark results
    """
    try:
        # Generate sample texts
        sample_texts = [
            f"This is sample text number {i} for benchmarking embedding generation performance. "
            f"It contains various words and phrases to test the embedding model thoroughly." 
            for i in range(num_samples)
        ]
        
        # Single embedding benchmark
        start_time = time.time()
        single_embedding = generate_embedding(sample_texts[0])
        single_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Batch embedding benchmark
        start_time = time.time()
        batch_embeddings = generate_embeddings_batch(sample_texts)
        batch_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Calculate statistics
        successful_embeddings = sum(1 for emb in batch_embeddings if emb is not None)
        
        return {
            'num_samples': num_samples,
            'successful_embeddings': successful_embeddings,
            'success_rate': successful_embeddings / num_samples,
            'single_embedding_time_ms': round(single_time, 2),
            'batch_total_time_ms': round(batch_time, 2),
            'batch_avg_time_per_text_ms': round(batch_time / num_samples, 2),
            'speedup_factor': round(single_time * num_samples / batch_time, 2),
            'model_info': get_embedding_model_info()
        }
        
    except Exception as e:
        logger.error(f"❌ Error in benchmark: {e}")
        return {'error': str(e)}

# Initialize the embedding model on module import for better performance
def auto_initialize():
    """
    Automatically initialize the embedding model when module is imported
    """
    try:
        initialize_embedding_model()
    except Exception as e:
        logger.warning(f"⚠️ Could not auto-initialize embedding model: {e}")

# Uncomment the line below to auto-initialize on import
# auto_initialize()
