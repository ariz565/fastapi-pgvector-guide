# Module 3: PostgreSQL + pgvector

Learn to build production-grade vector databases using PostgreSQL with the pgvector extension. This is where vector search becomes truly scalable!

## 🎯 Learning Goals

- Setup PostgreSQL with pgvector extension
- Design vector database schemas
- Implement efficient vector storage and retrieval
- Master vector indexing for performance
- Build production-ready vector applications

## 📚 What You'll Learn

### 1. Database Setup

- Install PostgreSQL and pgvector
- Configure vector extensions
- Design optimal schemas

### 2. Vector Operations

- Store and retrieve vectors efficiently
- Implement similarity search with SQL
- Handle different vector types and sizes

### 3. Performance Optimization

- Vector indexing strategies (IVFFlat, HNSW)
- Query optimization techniques
- Batch operations and bulk loading

### 4. Real Applications

- Document search with persistent storage
- User recommendation systems
- Image similarity search

## 🚀 Prerequisites

### Windows Setup:

```bash
# Install PostgreSQL (download from postgresql.org)
# Or use Docker:
docker run --name postgres-vector -e POSTGRES_PASSWORD=password -p 5432:5432 -d ankane/pgvector
```

### Python Dependencies:

```bash
pip install psycopg2-binary sqlalchemy asyncpg
```

## 📁 Module Structure

- `setup.md` - Complete setup instructions
- `basic_operations.py` - Vector CRUD operations
- `indexing_performance.py` - Performance optimization
- `document_search_db.py` - Full application example
- `migrations/` - Database schema files

## 💡 Key Concepts

- **Vector Extensions**: pgvector adds vector data types to PostgreSQL
- **Similarity Operators**: `<->`, `<#>`, `<=>` for different distance metrics
- **Vector Indexes**: IVFFlat and HNSW for fast approximate search
- **Hybrid Search**: Combining vector and traditional filters

## 🔧 Quick Start

1. **Setup Database**: Follow `setup.md`
2. **Run Basic Example**: `python basic_operations.py`
3. **Test Performance**: `python indexing_performance.py`

Ready to build scalable vector databases? Start with the setup guide!
