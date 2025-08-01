# Vector Database Interview Questions

## From Beginner to Advanced Level

_Comprehensive interview preparation covering FastAPI, PostgreSQL+pgvector, FAISS, OpenSearch, and vector search concepts_

---

## 📚 Table of Contents

1. [Vector Database Fundamentals](#vector-database-fundamentals)
2. [Similarity Search & Metrics](#similarity-search--metrics)
3. [Text Embeddings & NLP](#text-embeddings--nlp)
4. [FastAPI & Web Development](#fastapi--web-development)
5. [PostgreSQL & pgvector](#postgresql--pgvector)
6. [FAISS (Facebook AI Similarity Search)](#faiss-facebook-ai-similarity-search)
7. [OpenSearch & Elasticsearch](#opensearch--elasticsearch)
8. [System Design & Architecture](#system-design--architecture)
9. [Performance & Optimization](#performance--optimization)
10. [Real-world Applications](#real-world-applications)

---

## Vector Database Fundamentals

### Beginner Level

**Q1: What is a vector database and how does it differ from traditional databases?**

- **Answer**: A vector database stores and queries high-dimensional vectors (numerical arrays) representing data like text, images, or audio. Unlike traditional databases that use exact matches or range queries, vector databases perform similarity searches using mathematical distance calculations.

**Q2: What are vector embeddings?**

- **Answer**: Vector embeddings are numerical representations of data (text, images, etc.) in high-dimensional space where semantically similar items are close together. They capture meaning and context in a format machines can process.

**Q3: Name three common use cases for vector databases.**

- **Answer**:
  1. Semantic search (finding similar documents by meaning)
  2. Recommendation systems (finding similar products/content)
  3. Image/video search and retrieval

**Q4: What is the difference between sparse and dense vectors?**

- **Answer**:
  - **Sparse vectors**: Mostly zeros with few non-zero values (e.g., TF-IDF)
  - **Dense vectors**: Most dimensions have non-zero values (e.g., word2vec, BERT embeddings)

### Intermediate Level

**Q5: Explain the concept of vector dimensionality and its impact on performance.**

- **Answer**: Vector dimensionality is the number of features/dimensions in a vector. Higher dimensions can capture more nuanced relationships but increase storage, computation costs, and suffer from the "curse of dimensionality" where distance metrics become less discriminative.

**Q6: What is the curse of dimensionality in vector search?**

- **Answer**: As vector dimensions increase, the relative difference between nearest and farthest neighbors becomes smaller, making similarity search less effective. Distance metrics lose discriminative power in very high-dimensional spaces.

**Q7: Compare different approaches to generate embeddings.**

- **Answer**:
  - **Statistical**: TF-IDF, Count vectors (interpretable but limited semantic understanding)
  - **Neural**: Word2Vec, GloVe (capture semantic relationships)
  - **Transformer-based**: BERT, Sentence-BERT (context-aware, state-of-the-art)

### Advanced Level

**Q8: How would you handle embedding drift in production systems?**

- **Answer**:
  - Monitor embedding quality metrics over time
  - Implement A/B testing for new embedding models
  - Use versioning for embeddings
  - Gradual migration strategies
  - Regular retraining and validation

**Q9: Explain vector quantization and its trade-offs.**

- **Answer**: Vector quantization reduces vector precision (e.g., float32 to int8) to save memory and increase speed. Trade-offs include reduced accuracy, potential loss of fine-grained similarities, but significant performance gains and storage savings.

---

## Similarity Search & Metrics

### Beginner Level

**Q10: What are the main distance metrics used in vector search?**

- **Answer**:
  1. **Cosine similarity**: Measures angle between vectors (good for text)
  2. **Euclidean distance**: Straight-line distance (sensitive to magnitude)
  3. **Dot product**: Measures alignment and magnitude
  4. **Manhattan distance**: Sum of absolute differences

**Q11: When would you use cosine similarity vs Euclidean distance?**

- **Answer**:
  - **Cosine similarity**: When magnitude doesn't matter (text similarity, document comparison)
  - **Euclidean distance**: When magnitude is important (spatial data, image features)

**Q12: What is k-nearest neighbors (k-NN) search?**

- **Answer**: k-NN finds the k most similar vectors to a query vector based on a distance metric. It's the fundamental operation in vector search.

### Intermediate Level

**Q13: Explain the difference between exact and approximate similarity search.**

- **Answer**:
  - **Exact search**: Guarantees finding the true k nearest neighbors (slow for large datasets)
  - **Approximate search**: Trades accuracy for speed using indexing structures (HNSW, IVF)

**Q14: What is the recall metric in vector search and why is it important?**

- **Answer**: Recall measures the percentage of true nearest neighbors found by an approximate search algorithm. It's crucial for balancing speed vs accuracy in production systems.

**Q15: How do you handle multi-vector search (searching with multiple query vectors)?**

- **Answer**:
  - Average the query vectors
  - Use weighted combinations
  - Perform separate searches and merge results
  - Use advanced aggregation methods

### Advanced Level

**Q16: Describe the HNSW (Hierarchical Navigable Small World) algorithm.**

- **Answer**: HNSW creates a multi-layer graph where each layer contains connections between similar vectors. Search starts at the top layer (sparse, long-range connections) and moves down to denser layers, providing efficient approximate search with good recall.

**Q17: Explain Product Quantization (PQ) and its role in large-scale vector search.**

- **Answer**: PQ divides vectors into subvectors and quantizes each independently using k-means clustering. This dramatically reduces memory usage while maintaining reasonable search quality, essential for billion-scale vector search.

---

## Text Embeddings & NLP

### Beginner Level

**Q18: What is TF-IDF and how does it create vector representations?**

- **Answer**: TF-IDF (Term Frequency-Inverse Document Frequency) weighs words by their frequency in a document vs their rarity across the corpus. It creates sparse vectors where each dimension represents a word's importance.

**Q19: How do word embeddings like Word2Vec capture semantic meaning?**

- **Answer**: Word2Vec learns dense representations by predicting context words (CBOW) or target words from context (Skip-gram). Words used in similar contexts get similar embeddings, capturing semantic relationships.

**Q20: What's the difference between word-level and sentence-level embeddings?**

- **Answer**:
  - **Word-level**: Each word gets a vector (Word2Vec, GloVe)
  - **Sentence-level**: Entire sentences/documents get vectors (Doc2Vec, Sentence-BERT)

### Intermediate Level

**Q21: Explain the attention mechanism in transformer models and its impact on embeddings.**

- **Answer**: Attention allows models to focus on relevant parts of input when generating embeddings. It creates context-aware representations where the same word can have different embeddings based on surrounding context.

**Q22: How do you handle out-of-vocabulary (OOV) words in embeddings?**

- **Answer**:
  - Use subword tokenization (BPE, WordPiece)
  - Character-level embeddings
  - FastText (character n-grams)
  - Replace with special [UNK] tokens

**Q23: What are the advantages of Sentence-BERT over averaging BERT embeddings?**

- **Answer**: Sentence-BERT is fine-tuned on sentence pairs using siamese networks, creating embeddings optimized for semantic similarity. Simple BERT averaging loses the semantic structure that BERT captures.

### Advanced Level

**Q24: How would you implement multilingual vector search?**

- **Answer**:
  - Use multilingual embedding models (mBERT, XLM-R)
  - Translate queries to a common language
  - Cross-lingual alignment techniques
  - Language-specific embedding spaces with alignment

**Q25: Describe techniques for embedding fine-tuning for domain-specific applications.**

- **Answer**:
  - Continue pre-training on domain data
  - Task-specific fine-tuning (classification, similarity)
  - Few-shot learning with prompts
  - Adapter modules for efficiency
  - Metric learning approaches

---

## FastAPI & Web Development

### Beginner Level

**Q26: Why would you choose FastAPI for building vector search APIs?**

- **Answer**: FastAPI provides automatic OpenAPI documentation, built-in validation with Pydantic, async support for high concurrency, type hints, and excellent performance for API development.

**Q27: How do you handle large vector uploads in FastAPI?**

- **Answer**:
  - Use streaming uploads for large files
  - Implement request size limits
  - Process vectors asynchronously
  - Use background tasks for heavy operations

**Q28: What are Pydantic models and how do they help with vector data validation?**

- **Answer**: Pydantic models provide automatic data validation and serialization. For vectors, they can validate dimensions, data types, and ranges, ensuring API consistency.

### Intermediate Level

**Q29: How would you implement rate limiting for a vector search API?**

- **Answer**:
  - Use middleware like `slowapi` (Redis-backed)
  - Implement token bucket algorithms
  - Different limits for different endpoints
  - User-based rate limiting

**Q30: Describe async/await patterns for vector database operations in FastAPI.**

- **Answer**: Use async database clients, implement connection pooling, handle concurrent searches efficiently, and use background tasks for indexing operations to avoid blocking the API.

**Q31: How do you handle errors and exceptions in vector search APIs?**

- **Answer**:
  - Custom exception handlers for vector-specific errors
  - Proper HTTP status codes
  - Meaningful error messages
  - Logging and monitoring
  - Graceful degradation

### Advanced Level

**Q32: Design a caching strategy for vector search results.**

- **Answer**:
  - Cache frequently accessed vectors
  - Use Redis for search result caching
  - Implement cache invalidation strategies
  - Consider embedding fingerprints for cache keys
  - Balance memory usage vs hit rates

**Q33: How would you implement A/B testing for different embedding models in production?**

- **Answer**:
  - Feature flags for model selection
  - Dual-write to multiple indices
  - Traffic splitting middleware
  - Metrics collection for comparison
  - Gradual rollout strategies

---

## PostgreSQL & pgvector

### Beginner Level

**Q34: What is pgvector and what capabilities does it add to PostgreSQL?**

- **Answer**: pgvector is a PostgreSQL extension that adds vector data types and similarity search functions. It enables storing vectors and performing efficient k-NN searches using various distance metrics.

**Q35: How do you create a table with vector columns in PostgreSQL?**

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding VECTOR(768)
);
```

**Q36: What are the supported distance operators in pgvector?**

- **Answer**:
  - `<->` : L2 distance (Euclidean)
  - `<#>` : Negative inner product
  - `<=>` : Cosine distance

### Intermediate Level

**Q37: Explain different index types available in pgvector and when to use them.**

- **Answer**:
  - **IVF (Inverted File)**: Good for large datasets, faster search but approximate
  - **HNSW**: Better recall, good for real-time applications
  - **No index**: Exact search, suitable for small datasets

**Q38: How do you optimize pgvector performance for large datasets?**

- **Answer**:
  - Choose appropriate index type and parameters
  - Tune `work_mem` and `shared_buffers`
  - Use partial indexes for filtered searches
  - Consider vector dimensionality reduction
  - Partition large tables

**Q39: How would you implement hybrid search (text + vector) in PostgreSQL?**

```sql
SELECT *,
       ts_rank(to_tsvector(content), query) as text_score,
       1 - (embedding <=> $1) as vector_score
FROM documents
WHERE to_tsvector(content) @@ query
ORDER BY (text_score * 0.3 + vector_score * 0.7) DESC;
```

### Advanced Level

**Q40: Describe strategies for handling pgvector index updates in high-write environments.**

- **Answer**:
  - Batch updates to reduce index rebuilding
  - Use separate read/write replicas
  - Consider delayed indexing strategies
  - Monitor index bloat and maintenance
  - Implement write queuing for heavy loads

**Q41: How would you implement vector versioning and rollback in PostgreSQL?**

- **Answer**:
  - Use temporal tables for version history
  - Implement soft deletes with version columns
  - Store embedding model metadata
  - Create migration scripts for schema changes
  - Use logical replication for zero-downtime updates

---

## FAISS (Facebook AI Similarity Search)

### Beginner Level

**Q42: What is FAISS and what problems does it solve?**

- **Answer**: FAISS is a library for efficient similarity search of dense vectors. It solves the problem of fast k-NN search in large-scale datasets (millions to billions of vectors) using various indexing strategies.

**Q43: What's the difference between CPU and GPU FAISS implementations?**

- **Answer**:
  - **CPU FAISS**: Uses multi-threading, good for moderate datasets
  - **GPU FAISS**: Leverages GPU parallelism, excellent for large datasets and real-time applications

**Q44: Name three basic FAISS index types.**

- **Answer**:
  1. **IndexFlatL2**: Exact search using L2 distance (brute force)
  2. **IndexIVFFlat**: Clustering-based approximate search
  3. **IndexHNSW**: Hierarchical Navigable Small World graphs

### Intermediate Level

**Q45: Explain the IVF (Inverted File) indexing strategy in FAISS.**

- **Answer**: IVF clusters vectors using k-means, creating an inverted index where each cluster (centroid) points to vectors in that cluster. Search involves finding nearest centroids and searching only those clusters.

**Q46: How do you choose the right FAISS index for your use case?**

- **Answer**: Consider:
  - Dataset size (millions vs billions)
  - Memory constraints
  - Search speed requirements
  - Accuracy needs (exact vs approximate)
  - Update frequency

**Q47: What is quantization in FAISS and why is it useful?**

- **Answer**: Quantization reduces vector precision (e.g., Product Quantization) to save memory. It's useful for handling large datasets that don't fit in memory while maintaining reasonable search quality.

### Advanced Level

**Q48: Describe the training process for FAISS indices and its importance.**

- **Answer**: Training involves learning index parameters (cluster centroids, quantization codebooks) from a representative sample of data. Good training is crucial for index performance and requires representative, sufficient training data.

**Q49: How would you implement real-time vector updates in a FAISS-based system?**

- **Answer**:
  - Use mutable indices (like HNSW) for real-time updates
  - Implement index reconstruction strategies
  - Use hybrid approaches (immediate + batch updates)
  - Consider index sharding for distributed updates
  - Implement read-while-write patterns

**Q50: Explain FAISS's approach to billion-scale vector search.**

- **Answer**:
  - Multi-level quantization (OPQ + PQ)
  - GPU acceleration for parallel processing
  - Index sharding across multiple machines
  - Memory-mapped indices for large datasets
  - Sophisticated pruning strategies

---

## OpenSearch & Elasticsearch

### Beginner Level

**Q51: How does OpenSearch handle vector search differently from traditional text search?**

- **Answer**: OpenSearch uses k-NN algorithms (HNSW, IVF) for vector similarity search instead of inverted indices for text. It supports native vector fields and distance-based queries rather than term matching.

**Q52: What is the k-NN field type in OpenSearch?**

- **Answer**: The `knn_vector` field type stores dense vectors and enables similarity search. It requires specifying dimension and method (algorithm) parameters.

**Q53: How do you perform a basic vector search in OpenSearch?**

```json
{
  "query": {
    "knn": {
      "vector_field": {
        "vector": [0.1, 0.2, 0.3, ...],
        "k": 10
      }
    }
  }
}
```

### Intermediate Level

**Q54: Explain hybrid search in OpenSearch combining text and vector queries.**

- **Answer**: Hybrid search uses boolean queries to combine text matching (BM25) with vector similarity (k-NN), allowing search based on both semantic meaning and keyword relevance with configurable boosting.

**Q55: How do you optimize OpenSearch for vector workloads?**

- **Answer**:
  - Tune HNSW parameters (ef_construction, m)
  - Adjust JVM heap for vector memory usage
  - Use proper shard sizing
  - Configure circuit breakers for memory protection
  - Monitor vector index performance

**Q56: What are the key differences between Lucene and Faiss engines in OpenSearch?**

- **Answer**:
  - **Lucene**: Native integration, supports filtering, HNSW algorithm
  - **Faiss**: More algorithms available, potentially faster for some use cases, less integration

### Advanced Level

**Q57: How would you implement vector search with real-time filtering in OpenSearch?**

- **Answer**:
  - Use post-filtering with bool queries
  - Implement pre-filtering for better performance
  - Consider filter caching strategies
  - Use script scoring for complex filtering
  - Optimize for filter cardinality

**Q58: Describe strategies for managing vector index lifecycle in OpenSearch clusters.**

- **Answer**:
  - Index templates for consistent configuration
  - Hot-warm-cold architecture for time-series vectors
  - Automated rollover and deletion policies
  - Index shrinking for read-only data
  - Cross-cluster replication for DR

---

## System Design & Architecture

### Intermediate Level

**Q59: Design a scalable vector search system for a recommendation engine.**

- **Answer**:
  - Microservices architecture with embedding service
  - Distributed vector storage (sharded indices)
  - Caching layer for frequent queries
  - Async processing for batch updates
  - Load balancing and auto-scaling
  - Monitoring and alerting

**Q60: How would you handle vector index updates in a distributed system?**

- **Answer**:
  - Event-driven architecture with message queues
  - Eventual consistency models
  - Versioned indices for zero-downtime updates
  - Read replicas for high availability
  - Conflict resolution strategies

**Q61: Describe the trade-offs between different vector database architectures.**

- **Answer**:
  - **Embedded**: Simple, fast, limited scalability
  - **Client-server**: Network overhead, better scalability
  - **Distributed**: High scalability, complexity, consistency challenges
  - **Cloud-native**: Managed services, vendor lock-in

### Advanced Level

**Q62: Design a multi-tenant vector search system with isolation guarantees.**

- **Answer**:
  - Tenant-specific indices or index aliases
  - Resource quotas and rate limiting
  - Data encryption and access controls
  - Performance isolation strategies
  - Cost allocation and monitoring

**Q63: How would you implement cross-datacenter vector search replication?**

- **Answer**:
  - Async replication with conflict resolution
  - Multi-master setup with vector clocks
  - Network-aware routing
  - Consistency monitoring
  - Disaster recovery procedures

**Q64: Design a system for A/B testing different embedding models in production.**

- **Answer**:
  - Feature flags for model selection
  - Parallel index maintenance
  - Traffic splitting middleware
  - Metrics collection and analysis
  - Automated rollback mechanisms

---

## Performance & Optimization

### Intermediate Level

**Q65: What factors affect vector search performance and how do you optimize them?**

- **Answer**:
  - **Index type**: Choose based on use case
  - **Vector dimensions**: Higher = slower, consider PCA
  - **Memory usage**: Keep working set in memory
  - **Batch processing**: Group operations
  - **Hardware**: Use GPUs for large datasets

**Q66: How do you monitor and troubleshoot vector database performance?**

- **Answer**:
  - Query latency percentiles
  - Index build times
  - Memory usage patterns
  - Cache hit rates
  - Error rates and timeouts

**Q67: Explain the concept of recall vs latency trade-offs in vector search.**

- **Answer**: Higher recall (finding more true neighbors) typically requires more computation, increasing latency. Systems must balance these based on application requirements using techniques like parameter tuning and approximate algorithms.

### Advanced Level

**Q68: How would you implement intelligent query routing in a multi-index vector system?**

- **Answer**:
  - Query analysis for optimal index selection
  - Load-based routing algorithms
  - Index capability matching
  - Performance prediction models
  - Dynamic routing table updates

**Q69: Describe advanced techniques for reducing vector storage requirements.**

- **Answer**:
  - Dimensionality reduction (PCA, t-SNE)
  - Vector quantization (scalar, product)
  - Sparse encoding techniques
  - Compression algorithms
  - Hierarchical storage management

---

## Real-world Applications

### Intermediate Level

**Q70: How would you build a semantic search engine for a documentation website?**

- **Answer**:
  - Document chunking and preprocessing
  - Sentence-BERT embeddings
  - Vector storage with metadata
  - Query expansion techniques
  - Result ranking and filtering
  - User feedback incorporation

**Q71: Describe implementing a real-time recommendation system using vector search.**

- **Answer**:
  - User/item embedding generation
  - Real-time vector updates
  - Collaborative filtering with vectors
  - Cold start problem handling
  - A/B testing framework
  - Performance optimization

**Q72: How would you implement image similarity search for an e-commerce platform?**

- **Answer**:
  - CNN feature extraction (ResNet, VGG)
  - Image preprocessing pipeline
  - Vector normalization
  - Multi-scale search
  - Category-based filtering
  - Visual explanation of results

### Advanced Level

**Q73: Design a fraud detection system using vector embeddings.**

- **Answer**:
  - Transaction embedding creation
  - Anomaly detection algorithms
  - Real-time scoring pipeline
  - Model drift monitoring
  - Explainable AI components
  - Compliance and auditability

**Q74: How would you implement a voice search system using vector databases?**

- **Answer**:
  - Audio preprocessing and feature extraction
  - Speaker diarization
  - Speech-to-text integration
  - Audio embedding models
  - Temporal sequence handling
  - Noise robustness techniques

**Q75: Describe building a knowledge graph with vector embeddings for question answering.**

- **Answer**:
  - Entity and relation embedding
  - Graph neural networks
  - Multi-hop reasoning
  - Knowledge base completion
  - Vector-enhanced graph traversal
  - Explanation generation

---

## Bonus: Scenario-Based Questions

**Q76: Your vector search system is experiencing high latency during peak hours. How do you diagnose and fix this?**

**Q77: A client wants to search across 100 million product descriptions. Which technology stack would you recommend and why?**

**Q78: How would you implement versioning for embedding models in a production system with zero downtime?**

**Q79: Design a system that can switch between different similarity metrics based on user preferences.**

**Q80: You need to implement federated search across multiple vector databases. How would you approach this?**

---

## 📋 Interview Preparation Tips

### Technical Preparation

1. **Hands-on Practice**: Build projects using the technologies mentioned
2. **Performance Analysis**: Understand trade-offs between different approaches
3. **System Design**: Practice designing end-to-end vector search systems
4. **Code Implementation**: Be ready to write code for basic operations

### Key Areas to Focus On

- **Fundamentals**: Vector math, similarity metrics, embedding generation
- **Practical Experience**: Real-world problem solving and optimization
- **Technology Comparison**: When to use which tool and why
- **Scalability**: Handling large datasets and high query volumes

### Common Interview Formats

- **Coding Questions**: Implement similarity search algorithms
- **System Design**: Design scalable vector search systems
- **Problem Solving**: Optimize performance for specific use cases
- **Technology Deep-dive**: Explain internals of specific tools

---

## 🎯 Study Checklist

- [ ] Understand vector fundamentals and math
- [ ] Practice implementing similarity metrics
- [ ] Build end-to-end applications with FastAPI
- [ ] Experiment with pgvector, FAISS, and OpenSearch
- [ ] Study performance optimization techniques
- [ ] Practice system design scenarios
- [ ] Review real-world use cases
- [ ] Understand deployment and monitoring

---

_Good luck with your interviews! Remember to relate theoretical knowledge to practical experience and be ready to discuss trade-offs and design decisions._
