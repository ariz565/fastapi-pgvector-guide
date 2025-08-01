"""
Similarity Metrics - The Heart of Vector Search

Learn the different ways to measure how similar vectors are.
This is crucial for vector databases and search systems!

Learning Goals:
- Understand cosine similarity, euclidean distance, and dot product
- See when to use each metric
- Compare their behavior with real examples
"""

import numpy as np
from typing import List, Tuple, Dict
import math

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity measures the angle between vectors.
    Range: -1 to 1 (1 = identical direction, 0 = perpendicular, -1 = opposite)
    Good for: Text, when magnitude doesn't matter
    """
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    
    return dot_product / (magnitude_a * magnitude_b)

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Euclidean distance is the straight-line distance between vectors.
    Range: 0 to infinity (0 = identical, larger = more different)
    Good for: When actual distances matter (coordinates, measurements)
    """
    return np.linalg.norm(a - b)

def dot_product_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Dot product considers both angle and magnitude.
    Range: -infinity to infinity
    Good for: When both direction and magnitude are important
    """
    return np.dot(a, b)

def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Manhattan distance sums the absolute differences.
    Also called L1 distance or taxicab distance.
    Good for: High-dimensional sparse data
    """
    return np.sum(np.abs(a - b))

def compare_metrics_simple_example():
    """Compare all metrics with simple 2D vectors."""
    print("📊 Comparing Similarity Metrics")
    print("=" * 35)
    
    # Simple 2D vectors for easy visualization
    v1 = np.array([1.0, 0.0])  # Points right
    v2 = np.array([0.0, 1.0])  # Points up
    v3 = np.array([2.0, 0.0])  # Points right (longer)
    v4 = np.array([-1.0, 0.0]) # Points left (opposite)
    
    vectors = {
        "Right (1,0)": v1,
        "Up (0,1)": v2, 
        "Right Long (2,0)": v3,
        "Left (-1,0)": v4
    }
    
    print("Vectors:")
    for name, vec in vectors.items():
        print(f"  {name}: {vec}")
    print()
    
    # Compare v1 (right) with all others
    reference = "Right (1,0)"
    ref_vector = vectors[reference]
    
    print(f"Comparing '{reference}' with others:")
    print(f"{'Vector':<15} {'Cosine':<8} {'Euclidean':<10} {'Dot Product':<12} {'Manhattan':<10}")
    print("-" * 60)
    
    for name, vec in vectors.items():
        if name != reference:
            cos_sim = cosine_similarity(ref_vector, vec)
            euc_dist = euclidean_distance(ref_vector, vec)
            dot_prod = dot_product_similarity(ref_vector, vec)
            man_dist = manhattan_distance(ref_vector, vec)
            
            print(f"{name:<15} {cos_sim:<8.3f} {euc_dist:<10.3f} {dot_prod:<12.3f} {man_dist:<10.3f}")
    
    print("\n🔍 Key Observations:")
    print("• Cosine: Same direction (Right Long) = 1.0, perpendicular (Up) = 0.0, opposite (Left) = -1.0")
    print("• Euclidean: Measures actual distance - longer vectors are farther apart")
    print("• Dot Product: Considers both direction AND magnitude")
    print("• Manhattan: Sum of coordinate differences")
    print()

def text_similarity_example():
    """Demonstrate why cosine similarity is great for text."""
    print("📝 Text Similarity Example")
    print("=" * 27)
    
    # Imagine these represent word frequencies in documents
    # Dimensions: [cat, dog, animal, car, vehicle, fast]
    documents = {
        "Short cat article": np.array([5, 1, 3, 0, 0, 0]),
        "Long cat article": np.array([50, 10, 30, 0, 0, 0]),  # Same content, longer
        "Dog article": np.array([2, 8, 5, 0, 0, 0]),
        "Car review": np.array([0, 0, 0, 10, 8, 6])
    }
    
    print("Document word counts [cat, dog, animal, car, vehicle, fast]:")
    for title, counts in documents.items():
        print(f"  {title:<18}: {counts}")
    print()
    
    reference = "Short cat article"
    ref_vector = documents[reference]
    
    print(f"Similarity to '{reference}':")
    print(f"{'Document':<18} {'Cosine':<8} {'Euclidean':<10}")
    print("-" * 35)
    
    for title, vec in documents.items():
        if title != reference:
            cos_sim = cosine_similarity(ref_vector, vec)
            euc_dist = euclidean_distance(ref_vector, vec)
            
            print(f"{title:<18} {cos_sim:<8.3f} {euc_dist:<10.1f}")
    
    print("\n💡 Notice:")
    print("• Cosine: Long cat article = 1.0 (same topic, just longer)")
    print("• Euclidean: Long cat article has high distance (different magnitudes)")
    print("• For text similarity, cosine is usually better!")
    print()

def when_to_use_which_metric():
    """Guide on choosing the right similarity metric."""
    print("🎯 When to Use Which Metric")
    print("=" * 30)
    
    metrics_guide = {
        "Cosine Similarity": {
            "best_for": ["Text documents", "Recommendations", "When direction matters more than magnitude"],
            "range": "-1 to 1",
            "example": "Document similarity, user preferences"
        },
        "Euclidean Distance": {
            "best_for": ["Coordinates", "Measurements", "When actual distance matters"],
            "range": "0 to infinity",
            "example": "GPS coordinates, image pixels, sensor data"
        },
        "Dot Product": {
            "best_for": ["When both magnitude and direction matter"],
            "range": "-infinity to infinity", 
            "example": "Physics calculations, weighted similarities"
        },
        "Manhattan Distance": {
            "best_for": ["High-dimensional sparse data", "Grid-based problems"],
            "range": "0 to infinity",
            "example": "City block distances, feature differences"
        }
    }
    
    for metric_name, info in metrics_guide.items():
        print(f"🔸 {metric_name}")
        print(f"   Best for: {', '.join(info['best_for'])}")
        print(f"   Range: {info['range']}")
        print(f"   Example: {info['example']}")
        print()

def performance_comparison():
    """Compare performance of different metrics."""
    print("⚡ Performance Considerations")
    print("=" * 30)
    
    # Create larger vectors for timing
    import time
    
    # Generate random vectors
    np.random.seed(42)
    size = 1000
    num_vectors = 100
    
    vectors = [np.random.rand(size) for _ in range(num_vectors)]
    query_vector = np.random.rand(size)
    
    # Time each metric
    metrics = {
        "Cosine": cosine_similarity,
        "Euclidean": euclidean_distance,
        "Dot Product": dot_product_similarity,
        "Manhattan": manhattan_distance
    }
    
    print(f"Comparing {num_vectors} vectors of dimension {size}:")
    print(f"{'Metric':<12} {'Time (ms)':<10} {'Notes'}")
    print("-" * 35)
    
    for name, func in metrics.items():
        start_time = time.time()
        
        results = []
        for vec in vectors:
            result = func(query_vector, vec)
            results.append(result)
        
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        
        notes = ""
        if name == "Cosine":
            notes = "Most common for text"
        elif name == "Euclidean":
            notes = "Good for coordinates"
        elif name == "Dot Product":
            notes = "Fastest computation"
        elif name == "Manhattan":
            notes = "Good for sparse data"
        
        print(f"{name:<12} {elapsed_ms:<10.2f} {notes}")
    
    print("\n💡 In practice, choose based on your use case, not just speed!")

def main():
    """Run all similarity metric examples."""
    print("🎯 Similarity Metrics Deep Dive")
    print("=" * 35)
    print("Understanding how to measure similarity is crucial for")
    print("vector databases, search engines, and recommendation systems!")
    print()
    
    compare_metrics_simple_example()
    text_similarity_example()
    when_to_use_which_metric()
    performance_comparison()
    
    print("\n🎉 Great! You now understand the core similarity metrics.")
    print("🔜 Next: Run 'python simple_search.py' to build your first search engine!")

if __name__ == "__main__":
    main()
