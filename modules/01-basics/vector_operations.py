"""
Vector Operations - The Foundation of Vector Databases

This is your first hands-on lesson! We'll explore basic vector operations
that form the foundation of all vector database operations.

Learning Goals:
- Understand what vectors are
- Learn basic vector operations
- See how vectors represent real data
"""

import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

def create_sample_vectors() -> Tuple[np.ndarray, np.ndarray]:
    """Create some sample vectors to work with."""
    # Think of these as simplified representations of documents or products
    vector_a = np.array([1.0, 2.0, 3.0, 1.0])  # Document about "cats"
    vector_b = np.array([1.5, 2.2, 2.8, 1.2])  # Document about "cats and dogs"
    vector_c = np.array([0.1, 0.2, 4.0, 0.1])  # Document about "cars"
    
    print("🔢 Sample Vectors (think of these as document representations):")
    print(f"Document A (cats): {vector_a}")
    print(f"Document B (cats and dogs): {vector_b}")  
    print(f"Document C (cars): {vector_c}")
    print()
    
    return vector_a, vector_b, vector_c

def vector_magnitude(vector: np.ndarray) -> float:
    """Calculate the magnitude (length) of a vector."""
    magnitude = np.linalg.norm(vector)
    print(f"📏 Magnitude of {vector}: {magnitude:.3f}")
    return magnitude

def vector_addition_subtraction():
    """Demonstrate basic vector arithmetic."""
    print("➕ Vector Addition and Subtraction")
    print("=" * 40)
    
    v1 = np.array([1, 2, 3])
    v2 = np.array([4, 5, 6])
    
    addition = v1 + v2
    subtraction = v1 - v2
    
    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Addition: {v1} + {v2} = {addition}")
    print(f"Subtraction: {v1} - {v2} = {subtraction}")
    print()

def demonstrate_similarity():
    """Show how similar vectors have similar meanings."""
    print("🎯 Similarity Demonstration")
    print("=" * 30)
    
    # Create vectors representing different concepts
    cat_vector = np.array([0.8, 0.9, 0.1, 0.2])  # High on "animal" and "pet"
    dog_vector = np.array([0.9, 0.8, 0.1, 0.3])  # Similar to cat
    car_vector = np.array([0.1, 0.1, 0.9, 0.8])  # High on "vehicle" and "transport"
    
    print("Concept vectors (dimensions: [animal, pet, vehicle, transport]):")
    print(f"Cat:  {cat_vector}")
    print(f"Dog:  {dog_vector}")
    print(f"Car:  {car_vector}")
    
    # Calculate cosine similarity (we'll learn this properly in the next file)
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    cat_dog_sim = cosine_similarity(cat_vector, dog_vector)
    cat_car_sim = cosine_similarity(cat_vector, car_vector)
    
    print(f"\n📊 Similarities:")
    print(f"Cat ↔ Dog: {cat_dog_sim:.3f} (high = similar concepts)")
    print(f"Cat ↔ Car: {cat_car_sim:.3f} (low = different concepts)")
    print()

def normalize_vectors():
    """Demonstrate vector normalization - making all vectors the same length."""
    print("📐 Vector Normalization")
    print("=" * 25)
    
    original = np.array([3.0, 4.0, 0.0])
    normalized = original / np.linalg.norm(original)
    
    print(f"Original vector: {original}")
    print(f"Original magnitude: {np.linalg.norm(original):.3f}")
    print(f"Normalized vector: {normalized}")
    print(f"Normalized magnitude: {np.linalg.norm(normalized):.3f}")
    print("🔍 Notice: Normalized vectors always have magnitude = 1.0")
    print("💡 Why normalize? Makes similarity comparisons fair!")
    print()

def practical_example():
    """A practical example showing how vectors might represent products."""
    print("🛍️ Practical Example: Product Recommendations")
    print("=" * 45)
    
    # Imagine these are simplified product features: [price, quality, popularity, eco-friendly]
    products = {
        "iPhone": np.array([0.9, 0.9, 0.9, 0.3]),      # Expensive, high quality, popular
        "Samsung": np.array([0.8, 0.8, 0.8, 0.4]),     # Similar to iPhone
        "Budget Phone": np.array([0.2, 0.4, 0.3, 0.6]), # Cheap, lower quality
        "Eco Phone": np.array([0.6, 0.7, 0.2, 0.9]),   # Mid-range, very eco-friendly
    }
    
    print("Product vectors (dimensions: [price, quality, popularity, eco-friendly]):")
    for name, vector in products.items():
        print(f"{name:12}: {vector}")
    
    # Find products similar to iPhone
    iphone_vector = products["iPhone"]
    print(f"\n🔍 Products similar to iPhone:")
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    similarities = []
    for name, vector in products.items():
        if name != "iPhone":  # Don't compare iPhone to itself
            sim = cosine_similarity(iphone_vector, vector)
            similarities.append((name, sim))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    for name, sim in similarities:
        print(f"{name:12}: {sim:.3f} similarity")
    
    print("\n💡 This is the foundation of recommendation systems!")

def main():
    """Run all the vector operation examples."""
    print("🚀 Welcome to Vector Operations!")
    print("=" * 35)
    print("This is your introduction to the mathematical foundation")
    print("of vector databases, similarity search, and AI embeddings.")
    print()
    
    # Run each demonstration
    vector_addition_subtraction()
    
    vectors = create_sample_vectors()
    for i, vector in enumerate(vectors):
        vector_magnitude(vector)
    print()
    
    demonstrate_similarity()
    normalize_vectors()
    practical_example()
    
    print("\n🎉 Congratulations! You've learned the basics of vector operations.")
    print("🔜 Next: Run 'python similarity_metrics.py' to dive deeper into similarity!")

if __name__ == "__main__":
    main()
