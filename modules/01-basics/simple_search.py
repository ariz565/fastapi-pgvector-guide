"""
Simple In-Memory Vector Search

Build your first similarity search engine! This demonstrates the core
concept behind all vector databases before we add complexity.

Learning Goals:
- Implement basic vector search
- Understand linear search vs indexed search
- See how search quality depends on embeddings
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import time

class SimpleVectorSearch:
    """A basic in-memory vector search engine."""
    
    def __init__(self, metric: str = "cosine"):
        """
        Initialize the search engine.
        
        Args:
            metric: Similarity metric to use ('cosine', 'euclidean', 'dot')
        """
        self.vectors: List[np.ndarray] = []
        self.metadata: List[Dict] = []
        self.metric = metric
        print(f"🔍 SimpleVectorSearch initialized with {metric} similarity")
    
    def add_vector(self, vector: np.ndarray, metadata: Dict):
        """Add a vector and its metadata to the search index."""
        self.vectors.append(vector.copy())
        self.metadata.append(metadata.copy())
        print(f"➕ Added vector: {metadata.get('title', 'Untitled')}")
    
    def _calculate_similarity(self, query: np.ndarray, target: np.ndarray) -> float:
        """Calculate similarity between two vectors."""
        if self.metric == "cosine":
            dot_product = np.dot(query, target)
            magnitude_query = np.linalg.norm(query)
            magnitude_target = np.linalg.norm(target)
            
            if magnitude_query == 0 or magnitude_target == 0:
                return 0.0
            
            return dot_product / (magnitude_query * magnitude_target)
        
        elif self.metric == "euclidean":
            # For distance, we convert to similarity (smaller distance = higher similarity)
            distance = np.linalg.norm(query - target)
            return 1.0 / (1.0 + distance)  # Convert distance to similarity
        
        elif self.metric == "dot":
            return np.dot(query, target)
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Search for the most similar vectors.
        
        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            
        Returns:
            List of (metadata, similarity_score) tuples, sorted by similarity
        """
        if len(self.vectors) == 0:
            return []
        
        # Calculate similarity with all vectors (linear search)
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = self._calculate_similarity(query_vector, vector)
            similarities.append((self.metadata[i], similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k results
        return similarities[:top_k]
    
    def get_stats(self) -> Dict:
        """Get statistics about the search index."""
        if len(self.vectors) == 0:
            return {"count": 0}
        
        vector_dims = [len(v) for v in self.vectors]
        return {
            "count": len(self.vectors),
            "dimensions": vector_dims[0] if vector_dims else 0,
            "metric": self.metric
        }

def create_sample_documents():
    """Create sample documents with simple vector representations."""
    print("📚 Creating Sample Documents")
    print("=" * 30)
    
    # Simple document vectors (imagine these are from a real embedding model)
    # Dimensions represent different topics: [tech, animals, food, sports, travel]
    documents = [
        {
            "vector": np.array([0.9, 0.1, 0.1, 0.2, 0.1]),
            "metadata": {"title": "iPhone Review", "category": "tech", "content": "Latest smartphone features"}
        },
        {
            "vector": np.array([0.8, 0.2, 0.1, 0.1, 0.2]), 
            "metadata": {"title": "Android vs iOS", "category": "tech", "content": "Mobile operating systems compared"}
        },
        {
            "vector": np.array([0.1, 0.9, 0.1, 0.1, 0.1]),
            "metadata": {"title": "Cat Care Guide", "category": "animals", "content": "How to take care of your cat"}
        },
        {
            "vector": np.array([0.1, 0.8, 0.2, 0.1, 0.1]),
            "metadata": {"title": "Dog Training Tips", "category": "animals", "content": "Training your dog effectively"}
        },
        {
            "vector": np.array([0.1, 0.1, 0.9, 0.1, 0.2]),
            "metadata": {"title": "Italian Recipes", "category": "food", "content": "Authentic pasta and pizza recipes"}
        },
        {
            "vector": np.array([0.2, 0.1, 0.8, 0.1, 0.1]),
            "metadata": {"title": "Healthy Cooking", "category": "food", "content": "Nutritious meal preparation tips"}
        },
        {
            "vector": np.array([0.1, 0.1, 0.1, 0.9, 0.1]),
            "metadata": {"title": "Football Tactics", "category": "sports", "content": "Modern football strategies"}
        },
        {
            "vector": np.array([0.1, 0.2, 0.1, 0.1, 0.9]),
            "metadata": {"title": "Paris Travel Guide", "category": "travel", "content": "Best places to visit in Paris"}
        }
    ]
    
    print("Documents created:")
    for doc in documents:
        print(f"  📄 {doc['metadata']['title']} ({doc['metadata']['category']})")
    print()
    
    return documents

def demonstrate_search(search_engine: SimpleVectorSearch):
    """Show how the search engine works with different queries."""
    print("🔍 Search Demonstrations")
    print("=" * 25)
    
    # Query vectors representing user interests
    queries = [
        {
            "description": "Looking for technology content",
            "vector": np.array([0.9, 0.1, 0.1, 0.1, 0.1])  # High tech interest
        },
        {
            "description": "Interested in pet care", 
            "vector": np.array([0.1, 0.9, 0.1, 0.1, 0.1])  # High animal interest
        },
        {
            "description": "Mixed interests: tech and food",
            "vector": np.array([0.5, 0.1, 0.5, 0.1, 0.1])  # Both tech and food
        }
    ]
    
    for query in queries:
        print(f"🎯 Query: {query['description']}")
        print(f"   Vector: {query['vector']}")
        
        results = search_engine.search(query['vector'], top_k=3)
        
        print("   Results:")
        for i, (metadata, score) in enumerate(results, 1):
            print(f"   {i}. {metadata['title']} (score: {score:.3f})")
        print()

def compare_similarity_metrics():
    """Compare how different metrics affect search results."""
    print("⚖️ Comparing Similarity Metrics")
    print("=" * 35)
    
    documents = create_sample_documents()
    
    # Test with different metrics
    metrics = ["cosine", "euclidean", "dot"]
    query_vector = np.array([0.7, 0.3, 0.2, 0.1, 0.1])  # Mixed tech/animal interest
    
    print(f"Query vector: {query_vector}")
    print("(High tech, some animal interest)")
    print()
    
    for metric in metrics:
        print(f"📊 Using {metric} similarity:")
        
        # Create new search engine with this metric
        engine = SimpleVectorSearch(metric=metric)
        for doc in documents:
            engine.add_vector(doc['vector'], doc['metadata'])
        
        results = engine.search(query_vector, top_k=3)
        
        for i, (metadata, score) in enumerate(results, 1):
            print(f"   {i}. {metadata['title']} (score: {score:.3f})")
        print()

def performance_analysis():
    """Analyze search performance with different dataset sizes."""
    print("⚡ Performance Analysis")
    print("=" * 22)
    
    # Create search engines with different dataset sizes
    sizes = [100, 1000, 5000]
    
    for size in sizes:
        print(f"📈 Testing with {size} documents...")
        
        # Create random documents
        np.random.seed(42)  # For reproducible results
        engine = SimpleVectorSearch("cosine")
        
        for i in range(size):
            random_vector = np.random.rand(50)  # 50-dimensional vectors
            metadata = {"title": f"Document {i}", "id": i}
            engine.add_vector(random_vector, metadata)
        
        # Test search performance
        query_vector = np.random.rand(50)
        
        start_time = time.time()
        results = engine.search(query_vector, top_k=10)
        end_time = time.time()
        
        search_time_ms = (end_time - start_time) * 1000
        
        print(f"   Search time: {search_time_ms:.2f} ms")
        print(f"   Top result score: {results[0][1]:.3f}")
        print()
    
    print("💡 Observations:")
    print("• Linear search time grows with dataset size")
    print("• Real vector databases use indexing for faster search")
    print("• This simple approach works well for small datasets")

def build_mini_search_app():
    """Build a mini search application."""
    print("🚀 Mini Search Application")
    print("=" * 27)
    
    # Create a search engine with real-ish data
    engine = SimpleVectorSearch("cosine")
    
    # Add some articles (simplified vectors)
    articles = [
        {
            "vector": np.array([0.8, 0.1, 0.2, 0.1, 0.3, 0.1]),
            "metadata": {
                "title": "Machine Learning Basics",
                "author": "Dr. Smith", 
                "tags": ["AI", "ML", "Programming"],
                "content_preview": "Introduction to machine learning concepts..."
            }
        },
        {
            "vector": np.array([0.9, 0.2, 0.1, 0.1, 0.2, 0.1]),
            "metadata": {
                "title": "Deep Learning with Python",
                "author": "Jane Doe",
                "tags": ["Deep Learning", "Python", "Neural Networks"],
                "content_preview": "Building neural networks from scratch..."
            }
        },
        {
            "vector": np.array([0.1, 0.1, 0.9, 0.2, 0.1, 0.1]),
            "metadata": {
                "title": "Healthy Breakfast Ideas",
                "author": "Chef Mike",
                "tags": ["Food", "Health", "Breakfast"],
                "content_preview": "Start your day with nutritious meals..."
            }
        },
        {
            "vector": np.array([0.2, 0.1, 0.1, 0.8, 0.1, 0.2]),
            "metadata": {
                "title": "Olympic Swimming Techniques", 
                "author": "Coach Wilson",
                "tags": ["Sports", "Swimming", "Olympics"],
                "content_preview": "Professional swimming training methods..."
            }
        }
    ]
    
    print("Adding articles to search index...")
    for article in articles:
        engine.add_vector(article['vector'], article['metadata'])
    
    print(f"\n📊 Search Index Stats: {engine.get_stats()}")
    
    # Simulate user searches
    user_queries = [
        {
            "text": "I want to learn about AI and programming",
            "vector": np.array([0.9, 0.1, 0.1, 0.1, 0.2, 0.1])
        },
        {
            "text": "Looking for cooking and food recipes",
            "vector": np.array([0.1, 0.1, 0.9, 0.1, 0.1, 0.1])
        }
    ]
    
    print("\n🔍 Search Results:")
    for query in user_queries:
        print(f"\n💬 User query: \"{query['text']}\"")
        results = engine.search(query['vector'], top_k=2)
        
        for i, (metadata, score) in enumerate(results, 1):
            print(f"   {i}. 📄 {metadata['title']}")
            print(f"      👤 By {metadata['author']}")
            print(f"      🏷️ Tags: {', '.join(metadata['tags'])}")
            print(f"      📊 Relevance: {score:.3f}")
            print(f"      📝 Preview: {metadata['content_preview']}")

def main():
    """Run the complete simple search demonstration."""
    print("🔍 Simple Vector Search Engine")
    print("=" * 35)
    print("Let's build a basic search engine to understand")
    print("how vector databases work under the hood!")
    print()
    
    # Create sample data
    documents = create_sample_documents()
    
    # Create search engine and add documents
    search_engine = SimpleVectorSearch("cosine")
    for doc in documents:
        search_engine.add_vector(doc['vector'], doc['metadata'])
    
    print(f"📊 Search index created: {search_engine.get_stats()}")
    print()
    
    # Run demonstrations
    demonstrate_search(search_engine)
    compare_similarity_metrics()
    performance_analysis()
    build_mini_search_app()
    
    print("\n🎉 Excellent! You've built your first vector search engine!")
    print("🔜 Next: Run 'python text_embeddings.py' to learn about real text embeddings!")

if __name__ == "__main__":
    main()
