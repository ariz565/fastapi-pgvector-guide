"""
FAISS Basics - Introduction to Facebook AI Similarity Search

FAISS is the gold standard for high-performance vector similarity search.
Learn the fundamentals of this powerful library used by Facebook, Spotify, and more!

Learning Goals:
- Understand what FAISS is and why it's revolutionary
- Learn basic index types and operations
- See performance differences compared to simple search
- Build your first production-grade search system
"""

import numpy as np
import time
import faiss
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

def understand_faiss_basics():
    """Learn what FAISS is and why it's special."""
    print("🚀 Welcome to FAISS - Facebook AI Similarity Search")
    print("=" * 55)
    print("FAISS revolutionized vector search by making it:")
    print("• ⚡ Ultra-fast: Search millions of vectors in milliseconds")
    print("• 🎯 Accurate: Multiple algorithms for precision/speed trade-offs")
    print("• 💾 Memory-efficient: Smart compression and indexing")
    print("• 🔧 Production-ready: Used by Facebook, Spotify, Pinterest")
    print()
    
    print("💡 Key Concepts:")
    print("• Index: Data structure optimized for similarity search")
    print("• Quantization: Compress vectors to save memory")
    print("• Approximate Search: Trade accuracy for speed")
    print("• GPU Acceleration: Leverage graphics cards for speed")
    print()

def create_sample_dataset(n_vectors: int = 10000, dimension: int = 128) -> np.ndarray:
    """Create a sample dataset for FAISS examples."""
    print(f"📊 Creating sample dataset: {n_vectors} vectors, {dimension} dimensions")
    
    # Create random vectors (in practice, these would be embeddings from your data)
    np.random.seed(42)  # For reproducible results
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    
    # FAISS requires float32 for best performance
    print(f"✅ Dataset created: shape {vectors.shape}, type {vectors.dtype}")
    return vectors

def flat_index_example():
    """Demonstrate the basic Flat index - exact search."""
    print("\n🔍 FAISS Flat Index - Exact Search")
    print("=" * 40)
    print("The Flat index does exact search (brute force).")
    print("It's slow but gives perfect results - great for small datasets!")
    print()
    
    # Create dataset
    vectors = create_sample_dataset(n_vectors=1000, dimension=64)
    dimension = vectors.shape[1]
    
    # Create FAISS index
    print("🏗️  Building Flat index...")
    index = faiss.IndexFlatL2(dimension)  # L2 (Euclidean) distance
    
    print(f"📈 Index info before adding vectors:")
    print(f"   Trained: {index.is_trained}")
    print(f"   Total vectors: {index.ntotal}")
    
    # Add vectors to index
    print("➕ Adding vectors to index...")
    start_time = time.time()
    index.add(vectors)
    build_time = time.time() - start_time
    
    print(f"✅ Index built in {build_time:.3f} seconds")
    print(f"   Total vectors: {index.ntotal}")
    print()
    
    # Search for similar vectors
    query_vector = vectors[0:1]  # Use first vector as query
    k = 5  # Find 5 most similar vectors
    
    print(f"🔍 Searching for {k} most similar vectors...")
    start_time = time.time()
    distances, indices = index.search(query_vector, k)
    search_time = time.time() - start_time
    
    print(f"⚡ Search completed in {search_time*1000:.2f} milliseconds")
    print(f"📊 Results:")
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"   {i+1}. Index {idx}, Distance: {distance:.4f}")
    print()
    
    return index, vectors

def ivf_index_example():
    """Demonstrate IVF index - faster approximate search."""
    print("\n⚡ FAISS IVF Index - Fast Approximate Search")
    print("=" * 47)
    print("IVF (Inverted File) index trades some accuracy for much better speed.")
    print("It clusters vectors and only searches relevant clusters!")
    print()
    
    # Create larger dataset to see speed benefits
    vectors = create_sample_dataset(n_vectors=50000, dimension=128)
    dimension = vectors.shape[1]
    
    # IVF parameters
    nlist = 100  # Number of clusters
    
    print(f"🏗️  Building IVF index with {nlist} clusters...")
    
    # Create quantizer (for clustering)
    quantizer = faiss.IndexFlatL2(dimension)
    
    # Create IVF index
    index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    
    print(f"📈 Index info before training:")
    print(f"   Trained: {index.is_trained}")
    print(f"   Total vectors: {index.ntotal}")
    
    # Train the index (required for IVF)
    print("🎓 Training index (clustering vectors)...")
    start_time = time.time()
    index.train(vectors)
    train_time = time.time() - start_time
    
    print(f"✅ Training completed in {train_time:.3f} seconds")
    print(f"   Trained: {index.is_trained}")
    
    # Add vectors
    print("➕ Adding vectors to trained index...")
    start_time = time.time()
    index.add(vectors)
    build_time = time.time() - start_time
    
    print(f"✅ Index built in {build_time:.3f} seconds")
    print(f"   Total vectors: {index.ntotal}")
    print()
    
    # Search with different nprobe values
    query_vector = vectors[0:1]
    k = 10
    
    nprobe_values = [1, 5, 10, 20]
    
    print("🔍 Testing different search accuracies:")
    for nprobe in nprobe_values:
        index.nprobe = nprobe  # Number of clusters to search
        
        start_time = time.time()
        distances, indices = index.search(query_vector, k)
        search_time = time.time() - start_time
        
        print(f"   nprobe={nprobe:2d}: {search_time*1000:5.2f}ms, "
              f"distances: {distances[0][:3]} ...")
    
    print("\n💡 Observation: Higher nprobe = more accurate but slower")
    print()
    
    return index, vectors

def performance_comparison():
    """Compare FAISS with naive search implementation."""
    print("\n📊 Performance Comparison: FAISS vs Naive Search")
    print("=" * 50)
    
    sizes = [1000, 5000, 10000, 20000]
    dimension = 128
    k = 10
    
    faiss_times = []
    naive_times = []
    
    print(f"Testing search performance with {dimension}D vectors:")
    print(f"{'Dataset Size':<12} {'Naive (ms)':<12} {'FAISS (ms)':<12} {'Speedup':<10}")
    print("-" * 50)
    
    for size in sizes:
        # Create dataset
        vectors = np.random.random((size, dimension)).astype('float32')
        query = vectors[0:1]
        
        # FAISS search
        index = faiss.IndexFlatL2(dimension)
        index.add(vectors)
        
        start_time = time.time()
        distances, indices = index.search(query, k)
        faiss_time = (time.time() - start_time) * 1000
        faiss_times.append(faiss_time)
        
        # Naive search
        start_time = time.time()
        naive_distances = []
        for i, vector in enumerate(vectors):
            distance = np.linalg.norm(query[0] - vector)
            naive_distances.append((distance, i))
        naive_distances.sort()
        naive_time = (time.time() - start_time) * 1000
        naive_times.append(naive_time)
        
        speedup = naive_time / faiss_time if faiss_time > 0 else float('inf')
        
        print(f"{size:<12} {naive_time:<12.2f} {faiss_time:<12.2f} {speedup:<10.1f}x")
    
    print(f"\n🚀 FAISS is consistently faster, especially for larger datasets!")
    return faiss_times, naive_times

def memory_usage_example():
    """Demonstrate memory-efficient indexing."""
    print("\n💾 Memory Usage and Optimization")
    print("=" * 35)
    
    dimension = 128
    n_vectors = 100000
    
    print(f"Comparing memory usage for {n_vectors} vectors of {dimension} dimensions:")
    
    # Original vectors memory usage
    vectors = np.random.random((n_vectors, dimension)).astype('float32')
    original_memory = vectors.nbytes / (1024 * 1024)  # MB
    
    print(f"📊 Original vectors: {original_memory:.1f} MB")
    
    # Flat index memory
    index_flat = faiss.IndexFlatL2(dimension)
    index_flat.add(vectors)
    # FAISS Flat index stores vectors as-is, so similar memory usage
    print(f"📊 FAISS Flat index: ~{original_memory:.1f} MB (stores full vectors)")
    
    # IVF index memory (approximately)
    nlist = 1000
    quantizer = faiss.IndexFlatL2(dimension)
    index_ivf = faiss.IndexIVFFlat(quantizer, dimension, nlist)
    index_ivf.train(vectors)
    index_ivf.add(vectors)
    
    print(f"📊 FAISS IVF index: ~{original_memory:.1f} MB + cluster overhead")
    
    print("\n💡 Memory optimization strategies:")
    print("• Use IndexIVFPQ for product quantization (much smaller)")
    print("• Use IndexHNSW for good speed/memory balance")
    print("• Consider GPU memory for very large datasets")
    print("• Use binary vectors for extreme memory savings")

def practical_example():
    """Build a practical similarity search system."""
    print("\n🎯 Practical Example: Document Similarity Search")
    print("=" * 47)
    
    # Simulate document embeddings (in practice, use sentence transformers)
    print("📚 Creating simulated document embeddings...")
    n_documents = 10000
    embedding_dim = 384  # Common dimension for sentence transformers
    
    # Create document embeddings
    np.random.seed(42)
    document_embeddings = np.random.random((n_documents, embedding_dim)).astype('float32')
    
    # Simulate document metadata
    documents = [
        f"Document {i}: Sample content about topic {i % 10}" 
        for i in range(n_documents)
    ]
    
    print(f"✅ Created {n_documents} document embeddings ({embedding_dim}D)")
    
    # Build FAISS index
    print("🏗️  Building search index...")
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(document_embeddings)
    
    print(f"✅ Index built with {index.ntotal} documents")
    
    # Simulate search queries
    print("\n🔍 Testing search queries:")
    
    # Query 1: Find similar to document 42
    query_doc_id = 42
    query_embedding = document_embeddings[query_doc_id:query_doc_id+1]
    
    distances, indices = index.search(query_embedding, k=5)
    
    print(f"\nQuery: '{documents[query_doc_id]}'")
    print("Most similar documents:")
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"   {i+1}. {documents[idx]} (distance: {distance:.3f})")
    
    # Query 2: Random query
    print(f"\n🎲 Random query test:")
    random_query = np.random.random((1, embedding_dim)).astype('float32')
    distances, indices = index.search(random_query, k=3)
    
    print("Top matches for random query:")
    for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
        print(f"   {i+1}. {documents[idx]} (distance: {distance:.3f})")
    
    print(f"\n💡 In production, you would:")
    print("• Use real text embeddings (sentence-transformers, OpenAI, etc.)")
    print("• Store document metadata in a separate database")
    print("• Use IVF or HNSW indexes for better performance")
    print("• Add filtering and ranking logic")

def next_steps_guide():
    """Show what to learn next."""
    print("\n🎓 Congratulations! You've learned FAISS basics!")
    print("=" * 45)
    
    print("🔜 Next steps in your FAISS journey:")
    print()
    print("1. 📈 Advanced Indexing:")
    print("   python faiss_indexing.py")
    print("   • Learn IVF, HNSW, PQ, and more")
    print("   • Understand index selection criteria")
    print()
    print("2. ⚡ Performance Optimization:")
    print("   python faiss_performance.py")
    print("   • Benchmarking and tuning")
    print("   • GPU acceleration")
    print()
    print("3. 🏭 Production Systems:")
    print("   python faiss_production.py")
    print("   • Saving/loading indexes")
    print("   • Building scalable services")
    print()
    print("4. 🔍 OpenSearch Integration:")
    print("   python opensearch_vectors.py")
    print("   • Compare FAISS with OpenSearch")
    print("   • Learn when to use each")
    
    print("\n💪 Skills you've gained:")
    print("✅ FAISS index creation and usage")
    print("✅ Performance comparison techniques")
    print("✅ Memory usage understanding")
    print("✅ Practical similarity search implementation")

def main():
    """Run all FAISS basic examples."""
    print("🎯 FAISS Basics - Master High-Performance Vector Search")
    print("=" * 60)
    print("Learn the fundamentals of the world's most popular")
    print("vector similarity search library!")
    print()
    
    # Run examples
    understand_faiss_basics()
    flat_index_example()
    ivf_index_example()
    performance_comparison()
    memory_usage_example()
    practical_example()
    next_steps_guide()
    
    print("\n🌟 You're now ready for advanced FAISS techniques!")

if __name__ == "__main__":
    main()
