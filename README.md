# Vector Database Learning Journey

A comprehensive hands-on learning project for mastering vector databases, pgvector, FastAPI, and similarity search through building practical applications.

## 🎯 Learning Objectives

- Understand vector embeddings and similarity search concepts
- Master FastAPI for building vector-powered APIs
- Learn PostgreSQL with pgvector extension
- Build real-world similarity search applications
- Implement efficient vector indexing and querying

## 📚 Learning Path

### Phase 1: Foundations (modules/01-basics/)

- Vector operations and similarity metrics
- Basic similarity search implementation
- Understanding embeddings

### Phase 2: Web APIs (modules/02-fastapi/)

- FastAPI fundamentals with vectors
- REST API for similarity search
- Request/response handling

### Phase 3: Database Integration (modules/03-pgvector/)

- PostgreSQL setup with pgvector
- Vector storage and retrieval
- Database-backed similarity search

### Phase 4: Advanced Technologies (modules/04-advanced/)

- FAISS for high-performance vector search
- OpenSearch for distributed vector databases
- Performance comparison and optimization
- Technology selection guidance

### Phase 5: Real-world Projects (projects/)

- Complete applications combining all concepts
- Production-ready implementations

## 🚀 Quick Start

1. **Setup Environment**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Start with Basics**:

   ```bash
   python modules/01-basics/vector_operations.py
   ```

3. **Run FastAPI Examples**:

   ```bash
   cd modules/02-fastapi
   uvicorn main:app --reload
   ```

4. **Setup PostgreSQL with pgvector** (see modules/03-pgvector/setup.md)

## 📖 Learning Resources

Each module includes:

- 📝 Detailed explanations and theory
- 💻 Hands-on code examples
- 🔬 Exercises to practice
- 🎯 Real-world applications
- 📊 Performance comparisons

## 🛠 Technology Stack

- **Python 3.8+**: Core programming language
- **FastAPI**: Modern web framework for APIs
- **PostgreSQL**: Database with vector support
- **pgvector**: Vector similarity search extension
- **NumPy**: Numerical operations
- **scikit-learn**: Machine learning utilities
- **sentence-transformers**: Text embeddings

## 📁 Project Structure

```
vector-learning/
├── modules/                 # Learning modules
│   ├── 01-basics/          # Vector fundamentals
│   ├── 02-fastapi/         # Web API development
│   ├── 03-pgvector/        # Database integration
│   └── 04-advanced/        # Advanced techniques
├── projects/               # Complete applications
├── data/                   # Sample datasets
├── docs/                   # Additional documentation
└── tests/                  # Unit tests
```

## 🎓 Learning Tips

1. **Hands-on Approach**: Run every code example
2. **Experiment**: Modify parameters and observe results
3. **Build Projects**: Apply concepts in real applications
4. **Performance Focus**: Understand trade-offs and optimizations
5. **Real Data**: Use actual datasets for meaningful results

## 🤝 Next Steps

Start with `modules/01-basics/README.md` for your first lesson!
