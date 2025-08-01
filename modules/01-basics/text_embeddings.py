"""
Text Embeddings - Converting Words to Vectors

Learn how real text gets converted into vectors that capture semantic meaning.
This is the foundation of modern search engines and AI applications!

Learning Goals:
- Understand how text becomes vectors
- Try different embedding methods
- See how embeddings capture meaning
- Build a semantic search system
"""

import numpy as np
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re

# Try to import sentence transformers, provide fallback if not available
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️  sentence-transformers not available. Some examples will use simpler methods.")

class SimpleTextEmbedder:
    """A simple text embedding system using TF-IDF."""
    
    def __init__(self, method: str = "tfidf"):
        """
        Initialize the embedder.
        
        Args:
            method: Either 'tfidf' or 'count'
        """
        self.method = method
        if method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=100,  # Limit vocabulary size
                stop_words='english',
                lowercase=True
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=100,
                stop_words='english', 
                lowercase=True
            )
        
        self.is_fitted = False
        print(f"📝 SimpleTextEmbedder initialized with {method} method")
    
    def fit(self, texts: List[str]):
        """Fit the embedder on a collection of texts."""
        self.vectorizer.fit(texts)
        self.is_fitted = True
        print(f"✅ Embedder fitted on {len(texts)} texts")
        
        # Show some vocabulary
        vocab = list(self.vectorizer.vocabulary_.keys())[:10]
        print(f"   Sample vocabulary: {vocab}")
    
    def embed(self, texts: List[str]) -> np.ndarray:
        """Convert texts to vectors."""
        if not self.is_fitted:
            raise ValueError("Embedder must be fitted first!")
        
        vectors = self.vectorizer.transform(texts).toarray()
        return vectors
    
    def embed_single(self, text: str) -> np.ndarray:
        """Convert a single text to a vector."""
        return self.embed([text])[0]

def demonstrate_basic_embeddings():
    """Show how simple text becomes vectors."""
    print("🔤 Basic Text to Vector Conversion")
    print("=" * 37)
    
    # Sample texts
    texts = [
        "The cat sits on the mat",
        "A dog runs in the park", 
        "Cats and dogs are pets",
        "The car drives fast",
        "Fast cars are expensive"
    ]
    
    print("Original texts:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    print()
    
    # Create embedder and fit on texts
    embedder = SimpleTextEmbedder("tfidf")
    embedder.fit(texts)
    
    # Convert to vectors
    vectors = embedder.embed(texts)
    
    print(f"Text vectors (shape: {vectors.shape}):")
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        # Show only first 10 dimensions for readability
        vector_preview = vector[:10]
        non_zero_count = np.count_nonzero(vector)
        print(f"  {i+1}. \"{text}\"")
        print(f"     Vector preview: {vector_preview}")
        print(f"     Non-zero elements: {non_zero_count}/{len(vector)}")
        print()
    
    return embedder, texts, vectors

def find_similar_texts(embedder: SimpleTextEmbedder, texts: List[str], vectors: np.ndarray):
    """Find similar texts using vector similarity."""
    print("🔍 Finding Similar Texts")
    print("=" * 25)
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    # Test with a new query
    query = "pets and animals"
    print(f"Query: \"{query}\"")
    
    query_vector = embedder.embed_single(query)
    print(f"Query vector preview: {query_vector[:10]}")
    print()
    
    # Calculate similarities
    similarities = []
    for i, (text, vector) in enumerate(zip(texts, vectors)):
        similarity = cosine_similarity(query_vector, vector)
        similarities.append((text, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    print("Most similar texts:")
    for i, (text, sim) in enumerate(similarities, 1):
        print(f"  {i}. \"{text}\" (similarity: {sim:.3f})")
    print()

def compare_embedding_methods():
    """Compare different embedding approaches."""
    print("⚖️ Comparing Embedding Methods")
    print("=" * 33)
    
    sample_texts = [
        "Machine learning is amazing",
        "AI and machine learning are related", 
        "Dogs are loyal pets",
        "Cats are independent animals",
        "Cars need fuel to run"
    ]
    
    print("Sample texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"  {i}. {text}")
    print()
    
    # Test TF-IDF vs Count vectors
    methods = ["tfidf", "count"]
    
    for method in methods:
        print(f"📊 Using {method.upper()} embeddings:")
        
        embedder = SimpleTextEmbedder(method)
        embedder.fit(sample_texts)
        vectors = embedder.embed(sample_texts)
        
        # Test similarity between ML-related texts (1 and 2)
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        ml_similarity = cosine_similarity(vectors[0], vectors[1])
        pet_similarity = cosine_similarity(vectors[2], vectors[3])
        
        print(f"   ML texts similarity: {ml_similarity:.3f}")
        print(f"   Pet texts similarity: {pet_similarity:.3f}")
        print()

def advanced_embeddings_demo():
    """Demonstrate advanced embeddings if available."""
    print("🚀 Advanced Embeddings (Sentence Transformers)")
    print("=" * 48)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("⚠️  sentence-transformers not installed.")
        print("   Install with: pip install sentence-transformers")
        print("   Using simple TF-IDF instead...")
        
        # Fallback to simple embeddings
        simple_semantic_search()
        return
    
    # Use sentence transformers for high-quality embeddings
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("✅ Loaded pre-trained sentence transformer model")
    
    # Test texts with subtle semantic differences
    texts = [
        "The dog is running in the park",
        "A canine is jogging through the garden", 
        "The cat is sleeping on the couch",
        "A feline is resting on the sofa",
        "I love eating pizza for dinner",
        "Pizza is my favorite evening meal"
    ]
    
    print("\nTest texts:")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text}")
    
    # Generate embeddings
    embeddings = model.encode(texts)
    print(f"\nGenerated embeddings shape: {embeddings.shape}")
    
    # Calculate similarities
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("\n📊 Semantic similarities:")
    pairs = [
        (0, 1, "dog running vs canine jogging"),
        (2, 3, "cat sleeping vs feline resting"),
        (4, 5, "love pizza vs favorite meal"),
        (0, 2, "dog running vs cat sleeping"),
        (0, 4, "dog running vs love pizza")
    ]
    
    for i, j, description in pairs:
        similarity = cosine_similarity(embeddings[i], embeddings[j])
        print(f"   {description}: {similarity:.3f}")
    
    print("\n💡 Notice how semantically similar phrases get high scores!")

def simple_semantic_search():
    """Build a semantic search system using available tools."""
    print("🔍 Building Semantic Search System")
    print("=" * 35)
    
    # Knowledge base (simulate a document collection)
    documents = [
        "Python is a programming language used for web development and data science",
        "Machine learning algorithms can predict patterns in data",
        "Cats are independent pets that require minimal maintenance",
        "Dogs are loyal companions that need regular exercise and attention",
        "Cooking healthy meals requires fresh ingredients and proper techniques",
        "Exercise and nutrition are important for maintaining good health",
        "Solar panels convert sunlight into electrical energy for homes",
        "Electric cars are becoming more popular due to environmental concerns",
        "Reading books expands knowledge and improves vocabulary",
        "Music has therapeutic effects and can improve mood and concentration"
    ]
    
    print(f"Knowledge base: {len(documents)} documents")
    print()
    
    # Create embedder and process documents
    embedder = SimpleTextEmbedder("tfidf")
    embedder.fit(documents)
    doc_vectors = embedder.embed(documents)
    
    # Search function
    def search(query: str, top_k: int = 3):
        query_vector = embedder.embed_single(query)
        
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        similarities = []
        for i, doc_vector in enumerate(doc_vectors):
            similarity = cosine_similarity(query_vector, doc_vector)
            similarities.append((documents[i], similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # Test searches
    test_queries = [
        "I want to learn about programming and coding",
        "How to take care of pets and animals",
        "Information about renewable energy and environment",
        "Tips for staying healthy and fit"
    ]
    
    print("🔍 Search Results:")
    for query in test_queries:
        print(f"\nQuery: \"{query}\"")
        results = search(query, top_k=2)
        
        for i, (doc, score) in enumerate(results, 1):
            print(f"  {i}. {doc} (score: {score:.3f})")

def practical_tips():
    """Provide practical tips for working with text embeddings."""
    print("💡 Practical Tips for Text Embeddings")
    print("=" * 38)
    
    tips = [
        {
            "topic": "Preprocessing",
            "tip": "Clean your text: remove special characters, normalize case, handle stopwords",
            "example": "Convert 'Hello, World!!!' to 'hello world'"
        },
        {
            "topic": "Vocabulary Size", 
            "tip": "Balance vocabulary size - too small loses information, too large creates noise",
            "example": "Start with 1000-10000 features for TF-IDF"
        },
        {
            "topic": "Domain Adaptation",
            "tip": "Train embeddings on your domain-specific data for better results",
            "example": "Medical texts need medical vocabulary, not general English"
        },
        {
            "topic": "Dimensionality",
            "tip": "Higher dimensions capture more nuance but require more computation",
            "example": "384-768 dimensions are common for sentence transformers"
        },
        {
            "topic": "Similarity Metrics",
            "tip": "Cosine similarity works well for text, handles different text lengths",
            "example": "Short tweet vs long article can still be compared fairly"
        }
    ]
    
    for tip in tips:
        print(f"🔸 {tip['topic']}")
        print(f"   {tip['tip']}")
        print(f"   Example: {tip['example']}")
        print()

def main():
    """Run the complete text embeddings demonstration."""
    print("📝 Text Embeddings - From Words to Vectors")
    print("=" * 42)
    print("Learn how text becomes vectors that capture meaning!")
    print()
    
    # Run demonstrations
    embedder, texts, vectors = demonstrate_basic_embeddings()
    find_similar_texts(embedder, texts, vectors)
    compare_embedding_methods()
    advanced_embeddings_demo()
    simple_semantic_search()
    practical_tips()
    
    print("\n🎉 Fantastic! You now understand how text becomes meaningful vectors!")
    print("🔜 Next: Move to modules/02-fastapi/ to build web APIs with vector search!")

if __name__ == "__main__":
    main()
