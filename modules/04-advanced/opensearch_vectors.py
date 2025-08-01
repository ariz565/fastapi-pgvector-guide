"""
OpenSearch Vector Search - Distributed Vector Database

Learn OpenSearch for production-scale vector search with full database features.
OpenSearch combines the power of Elasticsearch with native vector similarity search!

Learning Goals:
- Set up OpenSearch with vector search capabilities
- Understand k-NN search and index configuration
- Implement hybrid search (text + vectors)
- Build production-ready search applications
"""

import json
import time
import numpy as np
from typing import List, Dict, Any, Optional
import requests

# Try to import OpenSearch client, provide fallback
try:
    from opensearchpy import OpenSearch, RequestsHttpConnection
    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False
    print("⚠️  opensearch-py not installed. Install with: pip install opensearch-py")

def check_opensearch_setup():
    """Check if OpenSearch is running and accessible."""
    print("🔍 Checking OpenSearch Connection")
    print("=" * 35)
    
    try:
        # Try to connect to local OpenSearch
        response = requests.get('http://localhost:9200', timeout=5)
        
        if response.status_code == 200:
            info = response.json()
            print(f"✅ OpenSearch is running!")
            print(f"   Version: {info.get('version', {}).get('number', 'Unknown')}")
            print(f"   Cluster: {info.get('cluster_name', 'Unknown')}")
            return True
        else:
            print(f"❌ OpenSearch responded with status {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to OpenSearch: {e}")
        print("\n🚀 Quick Setup Guide:")
        print("1. Using Docker (recommended):")
        print("   docker run -p 9200:9200 -p 9600:9600 -e 'discovery.type=single-node' opensearchproject/opensearch:latest")
        print("\n2. Or download from: https://opensearch.org/downloads.html")
        print("\n3. Make sure it's running on http://localhost:9200")
        return False

def create_opensearch_client():
    """Create OpenSearch client with proper configuration."""
    if not OPENSEARCH_AVAILABLE:
        print("❌ OpenSearch client not available. Install opensearch-py first.")
        return None
    
    try:
        client = OpenSearch(
            hosts=[{'host': 'localhost', 'port': 9200}],
            http_compress=True,
            http_auth=None,  # No auth for local development
            use_ssl=False,
            verify_certs=False,
            ssl_assert_hostname=False,
            ssl_show_warn=False,
            connection_class=RequestsHttpConnection,
            timeout=30
        )
        
        # Test connection
        info = client.info()
        print(f"✅ Connected to OpenSearch cluster: {info['cluster_name']}")
        return client
        
    except Exception as e:
        print(f"❌ Failed to create OpenSearch client: {e}")
        return None

def create_vector_index(client, index_name: str = "vector-demo"):
    """Create an index optimized for vector search."""
    print(f"\n🏗️  Creating Vector Index: {index_name}")
    print("=" * 40)
    
    # Delete index if it exists
    if client.indices.exists(index=index_name):
        print(f"🗑️  Deleting existing index: {index_name}")
        client.indices.delete(index=index_name)
    
    # Index configuration with k-NN settings
    index_config = {
        "settings": {
            "index": {
                "knn": True,  # Enable k-NN search
                "knn.algo_param.ef_search": 100,  # HNSW parameter
                "number_of_shards": 1,
                "number_of_replicas": 0
            }
        },
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "content": {
                    "type": "text",
                    "analyzer": "standard"
                },
                "category": {
                    "type": "keyword"
                },
                "vector": {
                    "type": "knn_vector",
                    "dimension": 128,  # Vector dimension
                    "method": {
                        "name": "hnsw",  # Hierarchical Navigable Small World
                        "space_type": "l2",  # Euclidean distance
                        "engine": "lucene",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 24
                        }
                    }
                },
                "timestamp": {
                    "type": "date"
                }
            }
        }
    }
    
    try:
        response = client.indices.create(index=index_name, body=index_config)
        print(f"✅ Index created successfully!")
        print(f"   Index: {index_name}")
        print(f"   Vector dimension: 128")
        print(f"   Algorithm: HNSW")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create index: {e}")
        return False

def generate_sample_documents():
    """Generate sample documents with vectors."""
    print("\n📚 Generating Sample Documents")
    print("=" * 32)
    
    # Sample documents
    documents = [
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "category": "technology"
        },
        {
            "title": "Deep Learning Fundamentals", 
            "content": "Deep learning uses neural networks with multiple layers to model complex patterns in data.",
            "category": "technology"
        },
        {
            "title": "Healthy Cooking Tips",
            "content": "Learn how to prepare nutritious meals with fresh ingredients and proper cooking techniques.",
            "category": "health"
        },
        {
            "title": "Travel Guide to Paris",
            "content": "Explore the beautiful city of Paris with its museums, cafes, and historic landmarks.",
            "category": "travel"
        },
        {
            "title": "Financial Planning Basics",
            "content": "Understanding budgeting, saving, and investing for a secure financial future.",
            "category": "finance"
        },
        {
            "title": "Python Programming Tutorial",
            "content": "Learn Python programming from basics to advanced concepts with practical examples.",
            "category": "technology"
        },
        {
            "title": "Yoga and Meditation",
            "content": "Discover the benefits of yoga and meditation for physical and mental wellness.",
            "category": "health"
        },
        {
            "title": "European Backpacking Adventure",
            "content": "Tips for budget-friendly backpacking across Europe with must-see destinations.",
            "category": "travel"
        }
    ]
    
    # Generate vectors for each document (in practice, use embeddings from text)
    np.random.seed(42)
    for i, doc in enumerate(documents):
        # Simulate text embeddings
        vector = np.random.random(128).astype('float32')
        
        # Add some category-based similarity
        if doc['category'] == 'technology':
            vector[0:20] += 0.5  # Higher values in first 20 dimensions
        elif doc['category'] == 'health':
            vector[20:40] += 0.5
        elif doc['category'] == 'travel':
            vector[40:60] += 0.5
        elif doc['category'] == 'finance':
            vector[60:80] += 0.5
        
        doc['vector'] = vector.tolist()
        doc['id'] = i + 1
        doc['timestamp'] = "2024-01-01T00:00:00Z"
    
    print(f"✅ Generated {len(documents)} sample documents")
    print("   Categories: technology, health, travel, finance")
    print("   Vector dimension: 128")
    
    return documents

def index_documents(client, documents: List[Dict], index_name: str = "vector-demo"):
    """Index documents in OpenSearch."""
    print(f"\n📥 Indexing Documents")
    print("=" * 20)
    
    try:
        for doc in documents:
            response = client.index(
                index=index_name,
                id=doc['id'],
                body=doc
            )
            print(f"✅ Indexed: {doc['title']}")
        
        # Refresh index to make documents searchable
        client.indices.refresh(index=index_name)
        print(f"\n🔄 Index refreshed - documents are now searchable")
        return True
        
    except Exception as e:
        print(f"❌ Failed to index documents: {e}")
        return False

def vector_search_example(client, index_name: str = "vector-demo"):
    """Demonstrate vector similarity search."""
    print(f"\n🔍 Vector Similarity Search")
    print("=" * 28)
    
    # Create a query vector (simulate searching for technology content)
    query_vector = np.random.random(128).astype('float32')
    query_vector[0:20] += 0.7  # High values in technology dimensions
    
    # k-NN search query
    search_query = {
        "size": 5,
        "query": {
            "knn": {
                "vector": {
                    "vector": query_vector.tolist(),
                    "k": 5
                }
            }
        },
        "_source": ["title", "content", "category"]
    }
    
    try:
        start_time = time.time()
        response = client.search(index=index_name, body=search_query)
        search_time = (time.time() - start_time) * 1000
        
        print(f"⚡ Search completed in {search_time:.2f}ms")
        print(f"📊 Found {len(response['hits']['hits'])} results")
        print()
        
        for i, hit in enumerate(response['hits']['hits'], 1):
            score = hit['_score']
            source = hit['_source']
            print(f"{i}. {source['title']}")
            print(f"   Category: {source['category']}")
            print(f"   Score: {score:.4f}")
            print(f"   Content: {source['content'][:80]}...")
            print()
        
        return True
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        return False

def hybrid_search_example(client, index_name: str = "vector-demo"):
    """Demonstrate hybrid search combining text and vector search."""
    print(f"\n🔍 Hybrid Search: Text + Vector")
    print("=" * 32)
    
    # Create query vector for technology content
    query_vector = np.random.random(128).astype('float32')
    query_vector[0:20] += 0.6
    
    # Hybrid search: text query + vector similarity
    hybrid_query = {
        "size": 5,
        "query": {
            "bool": {
                "should": [
                    {
                        "match": {
                            "content": "programming python machine learning"
                        }
                    },
                    {
                        "knn": {
                            "vector": {
                                "vector": query_vector.tolist(),
                                "k": 5,
                                "boost": 0.5  # Balance text vs vector importance
                            }
                        }
                    }
                ],
                "filter": [
                    {
                        "term": {
                            "category": "technology"
                        }
                    }
                ]
            }
        },
        "_source": ["title", "content", "category"],
        "explain": False
    }
    
    try:
        start_time = time.time()
        response = client.search(index=index_name, body=hybrid_query)
        search_time = (time.time() - start_time) * 1000
        
        print(f"⚡ Hybrid search completed in {search_time:.2f}ms")
        print("Query: Text='programming python machine learning' + Vector similarity")
        print("Filter: category='technology'")
        print()
        
        for i, hit in enumerate(response['hits']['hits'], 1):
            score = hit['_score']
            source = hit['_source']
            print(f"{i}. {source['title']}")
            print(f"   Category: {source['category']}")
            print(f"   Score: {score:.4f}")
            print(f"   Content: {source['content'][:80]}...")
            print()
        
        print("💡 Hybrid search combines:")
        print("• Text relevance (BM25 algorithm)")
        print("• Vector similarity (k-NN)")
        print("• Traditional filters")
        
        return True
        
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        return False

def aggregation_example(client, index_name: str = "vector-demo"):
    """Demonstrate aggregations with vector search."""
    print(f"\n📊 Vector Search with Aggregations")
    print("=" * 35)
    
    # Search with aggregations
    agg_query = {
        "size": 0,  # Don't return documents, just aggregations
        "aggs": {
            "categories": {
                "terms": {
                    "field": "category",
                    "size": 10
                }
            },
            "avg_vector_similarity": {
                "avg": {
                    "script": {
                        "source": "1.0 / (1.0 + doc['vector'].size())"  # Simple similarity metric
                    }
                }
            }
        }
    }
    
    try:
        response = client.search(index=index_name, body=agg_query)
        
        print("📈 Document categories:")
        for bucket in response['aggregations']['categories']['buckets']:
            print(f"   {bucket['key']}: {bucket['doc_count']} documents")
        
        print(f"\n📊 Total documents: {response['hits']['total']['value']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Aggregation query failed: {e}")
        return False

def performance_tuning_tips():
    """Show performance optimization techniques."""
    print(f"\n⚡ OpenSearch Vector Performance Tips")
    print("=" * 38)
    
    print("🎯 Index Configuration:")
    print("• Use HNSW algorithm for best speed/accuracy balance")
    print("• Adjust ef_construction (higher = better accuracy, slower indexing)")
    print("• Tune m parameter (connections per node)")
    print()
    
    print("🔍 Search Optimization:")
    print("• Use ef_search parameter to control search accuracy")
    print("• Filter before vector search when possible")
    print("• Use proper boost values in hybrid queries")
    print()
    
    print("💾 Memory and Storage:")
    print("• Vector dimensions directly impact memory usage")
    print("• Consider quantization for very large datasets")
    print("• Monitor heap usage and adjust JVM settings")
    print()
    
    print("🏗️ Cluster Configuration:")
    print("• Distribute vectors across multiple shards")
    print("• Use dedicated master nodes for large clusters")
    print("• Consider hot/warm architecture for time-series data")

def cleanup_demo(client, index_name: str = "vector-demo"):
    """Clean up the demo index."""
    print(f"\n🧹 Cleanup")
    print("=" * 10)
    
    try:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
            print(f"✅ Deleted demo index: {index_name}")
        else:
            print(f"ℹ️  Index {index_name} doesn't exist")
            
    except Exception as e:
        print(f"❌ Cleanup failed: {e}")

def main():
    """Run the complete OpenSearch vector search demonstration."""
    print("🔍 OpenSearch Vector Search - Production Scale")
    print("=" * 50)
    print("Learn to build distributed vector search systems")
    print("with the power of Elasticsearch + native vector support!")
    print()
    
    # Check if OpenSearch is available
    if not check_opensearch_setup():
        print("\n⚠️  Please set up OpenSearch and try again.")
        return
    
    if not OPENSEARCH_AVAILABLE:
        print("\n⚠️  Please install opensearch-py: pip install opensearch-py")
        return
    
    # Create client
    client = create_opensearch_client()
    if not client:
        return
    
    index_name = "vector-learning-demo"
    
    try:
        # Run examples
        if create_vector_index(client, index_name):
            documents = generate_sample_documents()
            
            if index_documents(client, documents, index_name):
                vector_search_example(client, index_name)
                hybrid_search_example(client, index_name)
                aggregation_example(client, index_name)
                performance_tuning_tips()
        
        # Cleanup
        cleanup_demo(client, index_name)
        
        print("\n🎉 OpenSearch vector search demo completed!")
        print("🔜 Next: Try faiss_vs_opensearch.py to compare technologies!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        cleanup_demo(client, index_name)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        cleanup_demo(client, index_name)

if __name__ == "__main__":
    main()
