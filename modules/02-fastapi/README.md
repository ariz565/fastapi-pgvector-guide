# Module 2: FastAPI + Vector Search

Build modern web APIs that power vector search applications! This module combines your vector knowledge with web development.

## 🎯 Learning Goals

- Build REST APIs with FastAPI
- Create endpoints for vector operations
- Handle file uploads and text processing
- Implement real-time similarity search
- Learn API best practices for vector applications

## 📚 What You'll Build

### 1. Basic Vector API

Simple endpoints for vector operations and similarity search.

### 2. Document Search API

Upload documents, convert to vectors, and search semantically.

### 3. Real-time Search Service

Interactive search with live results and filtering.

### 4. Advanced Features

- Batch processing
- Async operations
- Error handling
- API documentation

## 🚀 Getting Started

1. **Install Dependencies**:

   ```bash
   pip install fastapi uvicorn python-multipart
   ```

2. **Run Basic API**:

   ```bash
   cd modules/02-fastapi
   uvicorn main:app --reload
   ```

3. **Open Interactive Docs**:
   Visit `http://localhost:8000/docs`

## 📁 Files Overview

- `main.py` - Basic vector API
- `document_search.py` - Document upload and search
- `advanced_api.py` - Production-ready features
- `models.py` - Pydantic models for requests/responses
- `vector_service.py` - Business logic separation

## 💡 Key Concepts

- **REST API Design**: Clean, intuitive endpoints
- **Request Validation**: Automatic with Pydantic
- **Response Models**: Structured JSON responses
- **Error Handling**: Graceful failure management
- **Interactive Docs**: Automatic API documentation

Ready to build web APIs? Start with `python main.py`!
