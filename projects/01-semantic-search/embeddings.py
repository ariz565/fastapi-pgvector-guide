# Text embedding module using sentence-transformers
# This module converts text into vector embeddings for semantic search

from sentence_transformers import SentenceTransformer
import numpy as np
from config import EMBEDDING_MODEL

class TextEmbedder:
    """Handles text to vector conversion using sentence transformers"""
    
    def __init__(self):
        """Initialize the text embedder with pre-trained model"""
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        
        try:
            # Load the sentence transformer model
            self.model = SentenceTransformer(EMBEDDING_MODEL)
            print("Embedding model loaded successfully")
            
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            self.model = None
    
    def embed_text(self, text):
        """Convert a single text string into an embedding vector"""
        if not self.model:
            print("Error: Embedding model not loaded")
            return None
        
        try:
            # Generate embedding for the text
            embedding = self.model.encode(text)
            
            # Convert to numpy array if not already
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def embed_texts(self, texts):
        """Convert multiple texts into embedding vectors"""
        if not self.model:
            print("Error: Embedding model not loaded")
            return None
        
        try:
            # Generate embeddings for all texts at once (more efficient)
            embeddings = self.model.encode(texts)
            
            # Convert to numpy array if not already
            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)
            
            return embeddings
            
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return None
    
    def calculate_similarity(self, embedding1, embedding2):
        """Calculate cosine similarity between two embeddings"""
        try:
            # Calculate cosine similarity
            # Formula: dot(a,b) / (norm(a) * norm(b))
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            similarity = dot_product / (norm1 * norm2)
            return similarity
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

# Helper function to create text embedder
def get_text_embedder():
    """Create and return a text embedder instance"""
    embedder = TextEmbedder()
    if embedder.model:
        return embedder
    else:
        return None

# Example usage and testing functions
def test_embeddings():
    """Test the embedding functionality with sample texts"""
    print("Testing text embeddings...")
    
    # Create embedder
    embedder = get_text_embedder()
    if not embedder:
        print("Failed to create embedder")
        return
    
    # Test texts
    test_texts = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleepy dog",
        "Python is a great programming language",
        "Machine learning is fascinating"
    ]
    
    # Generate embeddings
    embeddings = embedder.embed_texts(test_texts)
    
    if embeddings is not None:
        print(f"Generated {len(embeddings)} embeddings")
        print(f"Embedding dimension: {embeddings[0].shape}")
        
        # Test similarity between first two texts (should be high)
        similarity = embedder.calculate_similarity(embeddings[0], embeddings[1])
        print(f"Similarity between similar texts: {similarity:.3f}")
        
        # Test similarity between different texts (should be lower)
        similarity2 = embedder.calculate_similarity(embeddings[0], embeddings[2])
        print(f"Similarity between different texts: {similarity2:.3f}")
    
    else:
        print("Failed to generate embeddings")

if __name__ == "__main__":
    test_embeddings()
