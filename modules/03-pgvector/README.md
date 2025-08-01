# Module 3: PostgreSQL + pgvector - Semantic Search Engine

Build a production-ready semantic search system using PostgreSQL 17 with pgvector extension. This comprehensive implementation demonstrates real-world vector database usage with FastAPI.

## 🎯 What You'll Build

A complete semantic search system featuring:

- **Document indexing** with automatic embedding generation
- **Semantic similarity search** using vector operations
- **Hybrid search** combining text and vector similarity
- **REST API** with FastAPI for web integration
- **Performance optimization** with vector indexing
- **Analytics and monitoring** capabilities

## �️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Semantic       │    │   PostgreSQL    │
│   Web API       │◄──►│   Search Engine  │◄──►│   + pgvector    │
│   (REST)        │    │   (Python)       │    │   (Database)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                        ▲                       ▲
         │                        │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Web Frontend  │    │   Document       │    │   Vector        │
│   (Browser)     │    │   Indexer        │    │   Embeddings    │
│                 │    │   (Batch)        │    │   (384-dim)     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
modules/03-pgvector/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── setup_pgvector.py          # Automated setup script
├── database_config.py         # Database connection & setup
├── document_indexer.py        # Document indexing with embeddings
├── semantic_search.py         # Search engine implementation
├── semantic_search_api.py     # FastAPI REST API
└── setup.md                   # Detailed setup instructions
```

## 🚀 Quick Start (15 minutes)

### Step 1: Prerequisites

```bash
# Ensure you have PostgreSQL 17 installed and running
# Install pgvector extension (see setup.md for details)

# Verify PostgreSQL is running
psql -U postgres -c "SELECT version();"
```

### Step 2: Install Dependencies

```bash
# Navigate to the pgvector module
cd modules/03-pgvector

# Install Python packages
pip install -r requirements.txt
```

### Step 3: Automated Setup

```bash
# Run the setup script - it will guide you through everything
python setup_pgvector.py

# Or for quick testing
python setup_pgvector.py test
```

### Step 4: Start the API Server

```bash
# Start the FastAPI server
python semantic_search_api.py

# Open in browser: http://localhost:8000/docs
```

## � Manual Setup (Alternative)

If you prefer step-by-step setup:

### 1. Database Setup

```python
# Setup database and tables
python database_config.py
```

### 2. Index Sample Documents

```python
# Load sample documents with embeddings
python document_indexer.py
```

### 3. Test Search Functionality

```python
# Test semantic search
python semantic_search.py

# Try interactive demo
python -c "from semantic_search import interactive_search_demo; interactive_search_demo()"
```

## 📊 Features Demonstrated

### Core Functionality

- ✅ **Vector Storage**: Efficient storage of 384-dimensional embeddings
- ✅ **Semantic Search**: Similarity search using cosine distance
- ✅ **Hybrid Search**: Combined text + vector search with weighted scoring
- ✅ **Batch Processing**: Efficient bulk document indexing
- ✅ **Performance Indexing**: IVFFlat and HNSW vector indexes

### Advanced Features

- ✅ **REST API**: Complete FastAPI implementation with OpenAPI docs
- ✅ **Category Filtering**: Search within specific document categories
- ✅ **Similar Documents**: Find documents similar to a given document
- ✅ **Search Analytics**: Performance monitoring and query analytics
- ✅ **Background Tasks**: Async processing for heavy operations

### Production Ready

- ✅ **Error Handling**: Comprehensive error handling and logging
- ✅ **Data Validation**: Pydantic models for request/response validation
- ✅ **Health Checks**: System health monitoring endpoints
- ✅ **CORS Support**: Web application integration
- ✅ **Environment Config**: Configurable database connections

## 🎯 Learning Objectives

### Database Skills

- Understanding pgvector extension capabilities
- Designing schemas for vector data
- Creating and optimizing vector indexes
- Managing large-scale vector operations

### Search Engine Skills

- Implementing semantic similarity algorithms
- Combining multiple ranking signals (hybrid search)
- Performance optimization techniques
- Search analytics and monitoring

### API Development Skills

- Building REST APIs with FastAPI
- Handling async operations
- Request/response validation
- Error handling and logging

### Production Skills

- Database connection management
- Background task processing
- Health monitoring
- Configuration management

## 📈 Performance Characteristics

### Search Performance

- **Semantic Search**: ~10-50ms for 10k documents
- **Hybrid Search**: ~20-100ms (depends on text complexity)
- **Batch Indexing**: ~1000 documents/minute
- **Memory Usage**: ~500MB for 10k documents with embeddings

### Scalability Features

- **Indexes**: Automatic index creation for 1000+ documents
- **Connection Pooling**: Efficient database connections
- **Batch Operations**: Optimized for bulk operations
- **Background Tasks**: Non-blocking heavy operations

## 🧪 Example Usage

### Basic Semantic Search

```python
from semantic_search import SemanticSearchEngine

search_engine = SemanticSearchEngine()

# Search for documents
results = search_engine.search("machine learning algorithms", limit=5)

for result in results['results']:
    print(f"{result['title']} (score: {result['similarity_score']})")
```

### REST API Usage

```bash
# Search via API
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "python programming", "limit": 5}'

# Add new document
curl -X POST "http://localhost:8000/documents" \
  -H "Content-Type: application/json" \
  -d '{"title": "My Document", "content": "Document content here", "category": "tech"}'
```

### Hybrid Search

```python
# Combine text and vector search
results = search_engine.hybrid_search(
    "python programming",
    text_weight=0.4,    # 40% text relevance
    vector_weight=0.6,  # 60% semantic similarity
    limit=5
)
```

## � Code Highlights

### Database Configuration

```python
# Automatic pgvector setup with error handling
class VectorDatabase:
    def setup_pgvector_extension(self):
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        # Comprehensive error handling and verification
```

### Embedding Generation

```python
# Efficient batch embedding generation
def generate_batch_embeddings(self, texts):
    embeddings = self.model.encode(texts, show_progress_bar=True)
    return [emb.astype(np.float32) for emb in embeddings]
```

### Vector Search Query

```sql
-- Semantic similarity with filtering
SELECT title, content,
       1 - (embedding <=> %s) as similarity_score
FROM documents
WHERE embedding IS NOT NULL
  AND category = %s
ORDER BY embedding <=> %s
LIMIT %s;
```

### Hybrid Search Implementation

```sql
-- Combined text + vector scoring
SELECT title, content,
  -- Text search score
  ts_rank(to_tsvector('english', title || ' ' || content),
          plainto_tsquery('english', %s)) as text_score,
  -- Vector similarity score
  1 - (embedding <=> %s) as vector_score,
  -- Weighted combination
  (%s * ts_rank(...)) + (%s * (1 - (embedding <=> %s))) as combined_score
FROM documents
ORDER BY combined_score DESC;
```

## �️ Troubleshooting

### Common Issues

**pgvector Extension Not Found**

```bash
# Install pgvector (Ubuntu/Debian)
sudo apt install postgresql-17-pgvector

# Or compile from source
git clone https://github.com/pgvector/pgvector.git
```

**Connection Refused**

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection parameters in code
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectordb
```

**Slow Search Performance**

```python
# Create vector indexes for better performance
db.create_vector_indexes()  # Automatic for 1000+ documents
```

**Memory Issues**

```python
# Reduce batch size for large documents
batch_size = 10  # Instead of default 50
indexer.add_documents_batch(documents[:batch_size])
```

## 🔄 What's Next?

After completing this module:

1. **Explore Advanced Features**:

   - Experiment with different embedding models
   - Try different similarity metrics
   - Implement custom ranking algorithms

2. **Scale Up**:

   - Test with larger document collections
   - Optimize for your specific use case
   - Add custom metadata fields

3. **Integration**:

   - Connect to your existing applications
   - Build custom frontends
   - Integrate with other data sources

4. **Production Deployment**:
   - Set up monitoring and alerting
   - Configure backups and disaster recovery
   - Implement rate limiting and security

## 📚 Additional Resources

- **pgvector Documentation**: https://github.com/pgvector/pgvector
- **PostgreSQL Performance**: https://www.postgresql.org/docs/17/performance-tips.html
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Sentence Transformers**: https://www.sbert.net/

## 🤝 Contributing

Found improvements or bugs? Feel free to:

- Open an issue for bugs or questions
- Submit pull requests for improvements
- Share your use cases and examples

---

**Ready to build your semantic search engine?** Start with `python setup_pgvector.py` and you'll have a working system in minutes! 🚀
