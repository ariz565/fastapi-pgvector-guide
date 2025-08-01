"""
Vector Database Technology Comparison

Compare FAISS, pgvector, OpenSearch, and in-memory search across multiple dimensions.
This comprehensive analysis helps you choose the right tool for your specific use case!

Learning Goals:
- Understand when to use each vector database technology
- Compare performance, scalability, and features
- Make informed technology decisions
- See trade-offs between different approaches
"""

import time
import numpy as np
import psutil
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from enum import Enum

# Import our implementations
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01-basics'))

try:
    from simple_search import SimpleVectorSearch
except ImportError:
    print("⚠️  Could not import SimpleVectorSearch from Module 1")
    SimpleVectorSearch = None

# Try to import FAISS
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("⚠️  FAISS not available. Install with: pip install faiss-cpu")

# Try to import psycopg2 for PostgreSQL
try:
    import psycopg2
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("⚠️  PostgreSQL client not available. Install with: pip install psycopg2-binary")

# Try to import OpenSearch
try:
    from opensearchpy import OpenSearch
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    print("⚠️  OpenSearch client not available. Install with: pip install opensearch-py")

class VectorDBType(Enum):
    """Enumeration of vector database types."""
    IN_MEMORY = "In-Memory (Simple)"
    FAISS = "FAISS"
    PGVECTOR = "PostgreSQL + pgvector"
    OPENSEARCH = "OpenSearch"

@dataclass
class PerformanceMetrics:
    """Performance metrics for comparison."""
    setup_time: float
    insert_time: float
    search_time: float
    memory_usage_mb: float
    accuracy: float
    throughput_qps: float

@dataclass 
class FeatureComparison:
    """Feature comparison matrix."""
    db_type: VectorDBType
    scalability: str
    query_types: List[str]
    indexing: List[str]
    persistence: bool
    distributed: bool
    production_ready: bool
    learning_curve: str
    use_cases: List[str]

class VectorDBBenchmark:
    """Comprehensive vector database benchmark suite."""
    
    def __init__(self, vector_dimension: int = 128):
        self.dimension = vector_dimension
        self.results: Dict[VectorDBType, PerformanceMetrics] = {}
        
    def generate_test_data(self, n_vectors: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate test vectors and queries."""
        print(f"📊 Generating test data: {n_vectors} vectors, {self.dimension}D")
        
        np.random.seed(42)
        vectors = np.random.random((n_vectors, self.dimension)).astype('float32')
        queries = np.random.random((100, self.dimension)).astype('float32')
        
        return vectors, queries
    
    def benchmark_in_memory(self, vectors: np.ndarray, queries: np.ndarray) -> PerformanceMetrics:
        """Benchmark simple in-memory search."""
        print("🧪 Benchmarking In-Memory Search...")
        
        if SimpleVectorSearch is None:
            return PerformanceMetrics(0, 0, float('inf'), 0, 0, 0)
        
        # Setup
        start_time = time.time()
        engine = SimpleVectorSearch("cosine")
        setup_time = time.time() - start_time
        
        # Insert
        start_time = time.time()
        for i, vector in enumerate(vectors):
            metadata = {"id": i, "category": f"cat_{i % 5}"}
            engine.add_vector(vector, metadata)
        insert_time = time.time() - start_time
        
        # Memory usage
        memory_mb = sys.getsizeof(engine.vectors) / (1024 * 1024)
        
        # Search
        k = 10
        search_times = []
        
        for query in queries[:10]:  # Test with 10 queries
            start_time = time.time()
            results = engine.search(query, k)
            search_times.append(time.time() - start_time)
        
        avg_search_time = np.mean(search_times)
        throughput = 1.0 / avg_search_time if avg_search_time > 0 else 0
        
        return PerformanceMetrics(
            setup_time=setup_time,
            insert_time=insert_time,
            search_time=avg_search_time * 1000,  # Convert to ms
            memory_usage_mb=memory_mb,
            accuracy=1.0,  # Exact search
            throughput_qps=throughput
        )
    
    def benchmark_faiss(self, vectors: np.ndarray, queries: np.ndarray) -> PerformanceMetrics:
        """Benchmark FAISS performance."""
        print("🧪 Benchmarking FAISS...")
        
        if not FAISS_AVAILABLE:
            return PerformanceMetrics(0, 0, float('inf'), 0, 0, 0)
        
        # Setup
        start_time = time.time()
        index = faiss.IndexFlatL2(self.dimension)
        setup_time = time.time() - start_time
        
        # Insert
        start_time = time.time()
        index.add(vectors)
        insert_time = time.time() - start_time
        
        # Memory usage (approximate)
        memory_mb = vectors.nbytes / (1024 * 1024)
        
        # Search
        k = 10
        search_times = []
        
        for query in queries[:10]:
            start_time = time.time()
            distances, indices = index.search(query.reshape(1, -1), k)
            search_times.append(time.time() - start_time)
        
        avg_search_time = np.mean(search_times)
        throughput = 1.0 / avg_search_time if avg_search_time > 0 else 0
        
        return PerformanceMetrics(
            setup_time=setup_time,
            insert_time=insert_time,
            search_time=avg_search_time * 1000,
            memory_usage_mb=memory_mb,
            accuracy=1.0,  # Exact search with Flat index
            throughput_qps=throughput
        )
    
    def benchmark_pgvector(self, vectors: np.ndarray, queries: np.ndarray) -> PerformanceMetrics:
        """Benchmark PostgreSQL + pgvector (simulation)."""
        print("🧪 Benchmarking pgvector (simulated)...")
        
        # Since we can't assume PostgreSQL is set up, we'll simulate based on known characteristics
        # In practice, you would connect to actual PostgreSQL with pgvector
        
        n_vectors = len(vectors)
        
        # Simulated metrics based on pgvector characteristics
        setup_time = 0.1  # Index creation time
        insert_time = n_vectors * 0.001  # ~1ms per vector insert
        search_time = 10.0  # ~10ms for typical queries
        memory_mb = vectors.nbytes / (1024 * 1024) * 1.2  # Some overhead
        throughput = 100  # ~100 QPS typical
        
        return PerformanceMetrics(
            setup_time=setup_time,
            insert_time=insert_time,
            search_time=search_time,
            memory_usage_mb=memory_mb,
            accuracy=0.95,  # Approximate with IVF index
            throughput_qps=throughput
        )
    
    def benchmark_opensearch(self, vectors: np.ndarray, queries: np.ndarray) -> PerformanceMetrics:
        """Benchmark OpenSearch (simulation)."""
        print("🧪 Benchmarking OpenSearch (simulated)...")
        
        # Simulated metrics based on OpenSearch characteristics
        n_vectors = len(vectors)
        
        setup_time = 2.0  # Index creation and mapping
        insert_time = n_vectors * 0.002  # ~2ms per document
        search_time = 15.0  # ~15ms for vector queries
        memory_mb = vectors.nbytes / (1024 * 1024) * 1.5  # Index overhead
        throughput = 50  # ~50 QPS typical for vector queries
        
        return PerformanceMetrics(
            setup_time=setup_time,
            insert_time=insert_time,
            search_time=search_time,
            memory_usage_mb=memory_mb,
            accuracy=0.92,  # HNSW approximate search
            throughput_qps=throughput
        )
    
    def run_benchmark(self, n_vectors: int = 10000):
        """Run complete benchmark across all technologies."""
        print(f"🏁 Running Comprehensive Vector DB Benchmark")
        print(f"=" * 50)
        print(f"Dataset: {n_vectors} vectors, {self.dimension} dimensions")
        print()
        
        vectors, queries = self.generate_test_data(n_vectors)
        
        # Benchmark each technology
        self.results[VectorDBType.IN_MEMORY] = self.benchmark_in_memory(vectors, queries)
        self.results[VectorDBType.FAISS] = self.benchmark_faiss(vectors, queries)
        self.results[VectorDBType.PGVECTOR] = self.benchmark_pgvector(vectors, queries)
        self.results[VectorDBType.OPENSEARCH] = self.benchmark_opensearch(vectors, queries)
        
        return self.results
    
    def print_performance_comparison(self):
        """Print detailed performance comparison."""
        print(f"\n📊 Performance Comparison Results")
        print("=" * 50)
        
        # Table header
        print(f"{'Technology':<20} {'Setup(s)':<10} {'Insert(s)':<10} {'Search(ms)':<12} {'Memory(MB)':<12} {'QPS':<8}")
        print("-" * 80)
        
        # Results
        for db_type, metrics in self.results.items():
            print(f"{db_type.value:<20} "
                  f"{metrics.setup_time:<10.3f} "
                  f"{metrics.insert_time:<10.3f} "
                  f"{metrics.search_time:<12.2f} "
                  f"{metrics.memory_usage_mb:<12.1f} "
                  f"{metrics.throughput_qps:<8.1f}")
        
        print()
        
        # Winner analysis
        print("🏆 Performance Winners:")
        
        fastest_search = min(self.results.items(), key=lambda x: x[1].search_time)
        print(f"• Fastest Search: {fastest_search[0].value} ({fastest_search[1].search_time:.2f}ms)")
        
        lowest_memory = min(self.results.items(), key=lambda x: x[1].memory_usage_mb)
        print(f"• Lowest Memory: {lowest_memory[0].value} ({lowest_memory[1].memory_usage_mb:.1f}MB)")
        
        highest_throughput = max(self.results.items(), key=lambda x: x[1].throughput_qps)
        print(f"• Highest Throughput: {highest_throughput[0].value} ({highest_throughput[1].throughput_qps:.1f} QPS)")

def feature_comparison():
    """Compare features across different vector databases."""
    print(f"\n🔍 Feature Comparison Matrix")
    print("=" * 35)
    
    features = [
        FeatureComparison(
            db_type=VectorDBType.IN_MEMORY,
            scalability="Small datasets only",
            query_types=["Vector similarity"],
            indexing=["Linear search"],
            persistence=False,
            distributed=False,
            production_ready=False,
            learning_curve="Easy",
            use_cases=["Prototyping", "Small demos", "Learning"]
        ),
        FeatureComparison(
            db_type=VectorDBType.FAISS,
            scalability="Billions of vectors",
            query_types=["Vector similarity", "Range search"],
            indexing=["Flat", "IVF", "HNSW", "PQ", "LSH"],
            persistence=True,
            distributed=False,
            production_ready=True,
            learning_curve="Medium",
            use_cases=["High-speed search", "Offline processing", "Embeddings"]
        ),
        FeatureComparison(
            db_type=VectorDBType.PGVECTOR,
            scalability="Millions of vectors",
            query_types=["Vector similarity", "SQL queries", "Hybrid search"],
            indexing=["IVF", "HNSW"],
            persistence=True,
            distributed=True,
            production_ready=True,
            learning_curve="Medium",
            use_cases=["Full-stack apps", "Existing PostgreSQL", "ACID transactions"]
        ),
        FeatureComparison(
            db_type=VectorDBType.OPENSEARCH,
            scalability="Distributed scale",
            query_types=["Vector similarity", "Text search", "Hybrid", "Analytics"],
            indexing=["HNSW", "Full-text"],
            persistence=True,
            distributed=True,
            production_ready=True,
            learning_curve="Hard",
            use_cases=["Search engines", "Analytics", "Multi-modal search"]
        )
    ]
    
    for feature in features:
        print(f"\n🔸 {feature.db_type.value}")
        print(f"   Scalability: {feature.scalability}")
        print(f"   Query Types: {', '.join(feature.query_types)}")
        print(f"   Indexing: {', '.join(feature.indexing)}")
        print(f"   Persistence: {'✅' if feature.persistence else '❌'}")
        print(f"   Distributed: {'✅' if feature.distributed else '❌'}")
        print(f"   Production: {'✅' if feature.production_ready else '❌'}")
        print(f"   Learning Curve: {feature.learning_curve}")
        print(f"   Best For: {', '.join(feature.use_cases)}")

def decision_matrix():
    """Provide decision guidance for choosing vector databases."""
    print(f"\n🎯 Decision Matrix: Which Vector DB to Choose?")
    print("=" * 50)
    
    scenarios = [
        {
            "scenario": "Learning & Prototyping",
            "recommendation": "In-Memory Simple Search",
            "reason": "Easy to understand, no setup required"
        },
        {
            "scenario": "High-Performance ML Pipeline",
            "recommendation": "FAISS",
            "reason": "Fastest search, optimized for ML workloads"
        },
        {
            "scenario": "Full-Stack Web Application",
            "recommendation": "PostgreSQL + pgvector",
            "reason": "ACID transactions, familiar SQL, full database features"
        },
        {
            "scenario": "Search Engine / Content Discovery",
            "recommendation": "OpenSearch",
            "reason": "Hybrid search, analytics, distributed architecture"
        },
        {
            "scenario": "Real-time Recommendations",
            "recommendation": "FAISS or pgvector",
            "reason": "Low latency requirements, depends on data consistency needs"
        },
        {
            "scenario": "Large-Scale Analytics",
            "recommendation": "OpenSearch",
            "reason": "Built-in aggregations, distributed processing"
        },
        {
            "scenario": "Existing PostgreSQL Infrastructure",
            "recommendation": "pgvector",
            "reason": "Leverage existing database expertise and tooling"
        },
        {
            "scenario": "GPU-Accelerated Workloads",
            "recommendation": "FAISS",
            "reason": "Native GPU support for massive datasets"
        }
    ]
    
    print(f"{'Scenario':<35} {'Recommendation':<25} {'Key Reason'}")
    print("-" * 85)
    
    for scenario in scenarios:
        print(f"{scenario['scenario']:<35} {scenario['recommendation']:<25} {scenario['reason']}")

def cost_analysis():
    """Analyze cost implications of different choices."""
    print(f"\n💰 Cost Analysis")
    print("=" * 16)
    
    print("💻 Development Costs:")
    print("• In-Memory: Very Low (hours)")
    print("• FAISS: Low to Medium (days to weeks)")
    print("• pgvector: Medium (weeks)")
    print("• OpenSearch: High (weeks to months)")
    print()
    
    print("🏗️ Infrastructure Costs:")
    print("• In-Memory: None (development only)")
    print("• FAISS: Low (single server, optional GPU)")
    print("• pgvector: Medium (PostgreSQL server)")
    print("• OpenSearch: High (cluster required)")
    print()
    
    print("🔧 Maintenance Costs:")
    print("• In-Memory: None")
    print("• FAISS: Low (file-based storage)")
    print("• pgvector: Medium (database administration)")
    print("• OpenSearch: High (cluster management)")

def migration_strategies():
    """Discuss migration paths between technologies."""
    print(f"\n🔄 Migration Strategies")
    print("=" * 22)
    
    print("📈 Growth Path (recommended learning order):")
    print("1. In-Memory → FAISS: Learn performance optimization")
    print("2. FAISS → pgvector: Add ACID transactions and SQL")
    print("3. pgvector → OpenSearch: Scale to distributed systems")
    print()
    
    print("🔀 Common Migration Scenarios:")
    print("• Prototype → Production: In-Memory → FAISS/pgvector")
    print("• Single Server → Distributed: pgvector → OpenSearch")
    print("• ML Pipeline → Full App: FAISS → pgvector")
    print("• Search Enhancement: Traditional DB → OpenSearch")
    print()
    
    print("⚠️ Migration Considerations:")
    print("• Vector compatibility (dimensions, formats)")
    print("• Query API differences")
    print("• Performance characteristics")
    print("• Operational complexity")

def main():
    """Run the complete technology comparison."""
    print("⚖️ Vector Database Technology Comparison")
    print("=" * 45)
    print("Comprehensive analysis to help you choose the right")
    print("vector database technology for your specific needs!")
    print()
    
    # Run benchmark
    benchmark = VectorDBBenchmark(vector_dimension=128)
    benchmark.run_benchmark(n_vectors=5000)
    benchmark.print_performance_comparison()
    
    # Feature comparison
    feature_comparison()
    
    # Decision guidance
    decision_matrix()
    cost_analysis()
    migration_strategies()
    
    print(f"\n🎓 Key Takeaways:")
    print("✅ Each technology has specific strengths")
    print("✅ Choice depends on your requirements and constraints") 
    print("✅ Start simple, evolve as needs grow")
    print("✅ Consider total cost of ownership")
    
    print(f"\n🔜 Next Steps:")
    print("• Try building the same application with different technologies")
    print("• Measure performance with your actual data")
    print("• Consider hybrid approaches (e.g., FAISS + PostgreSQL)")
    print("• Plan your technology evolution path")

if __name__ == "__main__":
    main()
