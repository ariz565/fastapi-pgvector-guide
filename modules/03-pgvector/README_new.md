# Module 3: Clean Function-Based Semantic Search Engine

**Professional semantic search system using PostgreSQL 17 + pgvector with clean, modular architecture**

This refactored implementation demonstrates production-ready semantic search using function-based design principles, proper separation of concerns, and comprehensive documentation.

## 🎯 What You'll Build

A complete semantic search system featuring:

- **Clean Architecture**: Function-based design with proper separation of concerns
- **Semantic Search**: Vector-based similarity search using embeddings
- **Hybrid Search**: Combines text and vector similarity with configurable weights
- **REST API**: FastAPI with automatic OpenAPI documentation
- **Production Ready**: Error handling, logging, monitoring, and analytics
- **Modular Design**: Easy to extend and maintain

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   main.py       │    │   routes.py      │    │   search.py     │
│   FastAPI App   │◄──►│   API Routes     │◄──►│   Search Logic  │
│   & Lifecycle   │    │   & Handlers     │    │   Functions     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                        ▲                       ▲
         │                        │                       │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   models.py     │    │   database.py    │    │   embeddings.py │
│   Pydantic      │    │   Database       │    │   Embedding     │
│   Models        │    │   Operations     │    │   Functions     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                ▲
                                │
                    ┌──────────────────┐
                    │   config.py      │
                    │   Configuration  │
                    │   & DB Setup     │
                    └──────────────────┘
```

## 📁 Project Structure

```
modules/03-pgvector/
├── README.md                   # This comprehensive guide
├── requirements.txt            # Python dependencies
├── setup.py                   # Automated setup script
│
├── main.py                    # FastAPI application & lifecycle
├── routes.py                  # API route handlers
├── models.py                  # Pydantic data models
├── search.py                  # Search functions
├── database.py                # Database operations
├── embeddings.py              # Embedding functions
└── config.py                  # Configuration & setup
```

## 🚀 Quick Start (10 minutes)

### Step 1: Prerequisites

```bash
# Ensure you have PostgreSQL 17 installed and running
# Python 3.8+ is required

# Clone or navigate to the project directory
cd modules/03-pgvector
```

### Step 2: Automated Setup

```bash
# Run the automated setup script
python setup.py

# This will:
# ✅ Check Python version
# ✅ Install dependencies
# ✅ Setup database and pgvector
# ✅ Initialize embedding model
# ✅ Create sample data
# ✅ Run functionality tests
# ✅ Optionally start the server
```

### Step 3: Manual Setup (Alternative)

```bash
# Install dependencies
pip install -r requirements.txt

# Setup database
python -c "from config import initialize_database; initialize_database()"

# Start the server
python main.py
```

### Step 4: Verify Installation

Open your browser to:

- **API Documentation**: http://127.0.0.1:8000/docs
- **Health Check**: http://127.0.0.1:8000/api/v1/health
- **Alternative Docs**: http://127.0.0.1:8000/redoc

## 🔧 Clean Architecture Principles

### Function-Based Design

Each module contains pure functions with single responsibilities:

```python
# search.py - Clean search functions
def semantic_search(query: str, limit: int = 10,
                   category_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform semantic search using vector similarity

    Args:
        query: Search query text
        limit: Maximum number of results
        category_filter: Optional category filter

    Returns:
        Dictionary containing search results and metadata
    """
    # Implementation with comprehensive error handling and logging

# database.py - Pure database operations
def search_documents_by_vector(query_embedding: np.ndarray,
                              limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search documents using vector similarity

    Args:
        query_embedding: Query vector for similarity search
        limit: Maximum number of results

    Returns:
        List of search result dictionaries
    """
    # Clean database operation with proper connection handling
```

### Separation of Concerns

- **`config.py`**: Database configuration and connection management
- **`models.py`**: Data validation and serialization with Pydantic
- **`database.py`**: Pure database operations and queries
- **`embeddings.py`**: Embedding generation and model management
- **`search.py`**: Search logic and algorithms
- **`routes.py`**: API route handlers and HTTP concerns
- **`main.py`**: Application initialization and lifecycle

### Comprehensive Documentation

Every function includes:

- Clear docstrings with Args and Returns
- Type hints for all parameters
- Detailed # comments explaining logic
- Error handling and logging
- Usage examples

## 📊 API Endpoints

### Core Search Endpoints

```bash
# Semantic search using vector similarity
POST /api/v1/search
{
    "query": "machine learning algorithms",
    "limit": 10,
    "similarity_threshold": 0.5
}

# Hybrid search (text + vector)
POST /api/v1/search/hybrid
{
    "query": "python programming",
    "text_weight": 0.3,
    "vector_weight": 0.7,
    "limit": 10
}

# Find similar documents
POST /api/v1/search/similar
{
    "document_id": 5,
    "limit": 10,
    "exclude_same_category": false
}

# Search by category
GET /api/v1/search/category/technology?limit=20

# Multi-query search with different strategies
POST /api/v1/search/multi
{
    "queries": ["machine learning", "artificial intelligence"],
    "strategy": "average",  # or "union", "intersection"
    "limit": 10
}
```

### Document Management

```bash
# Add single document
POST /api/v1/documents
{
    "title": "Document Title",
    "content": "Document content...",
    "category": "technology",
    "url": "https://example.com"
}

# Batch add documents
POST /api/v1/documents/batch
{
    "documents": [
        {"title": "Doc 1", "content": "Content 1"},
        {"title": "Doc 2", "content": "Content 2"}
    ]
}

# Get document by ID
GET /api/v1/documents/123

# Delete document
DELETE /api/v1/documents/123
```

### Analytics and Monitoring

```bash
# System health check
GET /api/v1/health

# Database statistics
GET /api/v1/stats

# Search analytics
GET /api/v1/analytics?days=7

# List all categories
GET /api/v1/categories

# Embedding model information
GET /api/v1/embeddings/info

# Performance benchmark
POST /api/v1/embeddings/benchmark?num_samples=100
```

## 🔍 Key Features

### 1. Semantic Search Engine

```python
# Pure function-based semantic search
from search import semantic_search

results = semantic_search(
    query="artificial intelligence applications",
    limit=5,
    category_filter="technology",
    similarity_threshold=0.6
)

print(f"Found {results['total_results']} results in {results['search_time_ms']}ms")
```

### 2. Hybrid Search Capabilities

```python
# Combine text and vector search with configurable weights
from search import hybrid_search

results = hybrid_search(
    query="machine learning python",
    text_weight=0.4,      # 40% text search weight
    vector_weight=0.6,    # 60% vector search weight
    limit=10
)
```

### 3. Document Management

```python
# Clean database operations
from database import insert_document, get_documents_by_category
from embeddings import generate_embedding

# Add document with embedding
embedding = generate_embedding("Document content here")
doc_id = insert_document(
    title="My Document",
    content="Document content...",
    category="tech",
    embedding=embedding
)
```

### 4. Comprehensive Error Handling

```python
# All functions include proper error handling
def semantic_search(query: str, limit: int = 10) -> Dict[str, Any]:
    try:
        # Search logic with detailed logging
        logger.info(f"🔍 Searching for: '{query[:50]}...'")

        # Generate embedding with fallback
        query_embedding = generate_embedding(query)
        if query_embedding is None:
            return {
                'query': query,
                'results': [],
                'error': 'Failed to generate query embedding'
            }

        # Database search with error recovery
        results = search_documents_by_vector(query_embedding, limit)

        logger.info(f"✅ Found {len(results)} results")
        return format_search_response(results)

    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return create_error_response(str(e))
```

## 🎯 Code Quality Features

### Type Safety

- Type hints on all function parameters and return values
- Pydantic models for data validation
- Optional types for nullable values

### Logging and Monitoring

- Comprehensive logging with emojis for easy scanning
- Request tracking with unique IDs
- Performance timing for all operations
- Health checks and system monitoring

### Error Handling

- Graceful error recovery at all levels
- Standardized error response format
- Detailed error logging with context
- User-friendly error messages

### Documentation

- Docstrings for every function
- Clear # comments explaining logic
- Type annotations for better IDE support
- Comprehensive README with examples

## 🧪 Testing and Development

### Run Tests

```bash
# Basic functionality tests
python -c "
from search import semantic_search
from database import get_document_count

print(f'Documents: {get_document_count()}')
results = semantic_search('test query')
print(f'Search results: {results[\"total_results\"]}')
"

# Performance benchmark
curl -X POST "http://127.0.0.1:8000/api/v1/embeddings/benchmark?num_samples=100"
```

### Development Mode

```bash
# Start with auto-reload for development
python main.py --reload

# Or use uvicorn directly
uvicorn main:app --reload --port 8000
```

### Interactive Testing

```python
# Test search functions directly
from search import semantic_search, hybrid_search, find_similar_documents

# Semantic search
results = semantic_search("machine learning")
print(f"Results: {results['total_results']}")

# Hybrid search
hybrid_results = hybrid_search("python programming", text_weight=0.3, vector_weight=0.7)

# Similar documents
similar = find_similar_documents(document_id=1, limit=5)
```

## 📈 Performance Optimization

### Vector Indexing

```sql
-- Automatic HNSW index creation for fast similarity search
CREATE INDEX idx_documents_embedding_hnsw
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

### Batch Processing

```python
# Efficient batch embedding generation
from embeddings import generate_embeddings_batch

texts = ["Text 1", "Text 2", "Text 3"]
embeddings = generate_embeddings_batch(texts, batch_size=32)
```

### Connection Pooling

```python
# Efficient database connection management
from config import get_db_connection

with get_db_connection() as conn:
    # Database operations with automatic cleanup
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM documents LIMIT 10")
    results = cursor.fetchall()
```

## 🔧 Configuration

### Environment Variables

```bash
# Database configuration (optional - defaults provided)
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=semantic_search
export DB_USER=postgres
export DB_PASSWORD=postgres
```

### Model Configuration

```python
# Configure embedding model in embeddings.py
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dimensional embeddings
BATCH_SIZE = 32                       # Batch processing size
MAX_TEXT_LENGTH = 8000               # Maximum text length
```

## 🚀 Production Deployment

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

### Production Server

```bash
# Install production server
pip install gunicorn

# Run with gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

## 📚 Learning Outcomes

After completing this module, you'll understand:

1. **Clean Architecture**: Function-based design with proper separation
2. **Vector Databases**: PostgreSQL + pgvector for production use
3. **Semantic Search**: Embedding-based similarity search
4. **Hybrid Search**: Combining text and vector search effectively
5. **FastAPI**: Production-ready REST API development
6. **Error Handling**: Comprehensive error management strategies
7. **Performance**: Optimization techniques for large-scale search
8. **Monitoring**: Health checks, analytics, and system monitoring

## 🎉 Next Steps

1. **Extend Search**: Add faceted search, filters, and advanced ranking
2. **Scale Up**: Implement distributed search across multiple databases
3. **Add Features**: Document clustering, recommendation systems
4. **Optimize**: Fine-tune embedding models for your domain
5. **Monitor**: Add metrics, alerting, and performance monitoring
6. **Deploy**: Production deployment with CI/CD pipelines

## 🤝 Contributing

This is a learning project demonstrating clean architecture principles. Feel free to:

- Extend functionality with new search algorithms
- Add more comprehensive error handling
- Implement additional API endpoints
- Optimize performance for your use case
- Add more detailed monitoring and analytics

---

**Happy Learning! 🚀**

_This clean, function-based implementation demonstrates production-ready semantic search with proper architecture, comprehensive documentation, and extensive error handling._
