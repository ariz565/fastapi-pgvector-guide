"""
FastAPI Vector Search API - Basic Implementation

Your first vector-powered web API! This demonstrates how to build
REST endpoints for vector operations and similarity search.

Learning Goals:
- Create FastAPI applications with vector operations
- Handle HTTP requests and responses
- Implement basic search endpoints
- Use interactive API documentation
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import uvicorn

# Import our vector search from the basics module
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '01-basics'))

try:
    from simple_search import SimpleVectorSearch
except ImportError:
    print("⚠️  Could not import SimpleVectorSearch. Make sure you've completed module 01.")
    # Create a simple fallback
    class SimpleVectorSearch:
        def __init__(self, metric="cosine"):
            self.vectors = []
            self.metadata = []
        
        def add_vector(self, vector, metadata):
            self.vectors.append(vector)
            self.metadata.append(metadata)
        
        def search(self, query_vector, top_k=5):
            # Simple fallback implementation
            if not self.vectors:
                return []
            similarities = []
            for i, vec in enumerate(self.vectors):
                # Simple dot product similarity
                sim = np.dot(query_vector, vec)
                similarities.append((self.metadata[i], sim))
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]

# Initialize FastAPI app
app = FastAPI(
    title="Vector Search API",
    description="A REST API for vector similarity search operations",
    version="1.0.0",
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc"  # ReDoc alternative
)

# Global search engine instance
search_engine = SimpleVectorSearch("cosine")

# Pydantic models for request/response validation
class VectorModel(BaseModel):
    """Model for vector data."""
    vector: List[float] = Field(..., description="Vector values", min_items=1)
    metadata: Dict = Field(default_factory=dict, description="Associated metadata")

class SearchRequest(BaseModel):
    """Model for search requests."""
    query_vector: List[float] = Field(..., description="Query vector for similarity search")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results to return")

class SearchResult(BaseModel):
    """Model for individual search results."""
    metadata: Dict
    similarity_score: float
    rank: int

class SearchResponse(BaseModel):
    """Model for search response."""
    query_vector: List[float]
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    timestamp: datetime

class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str
    vector_count: int
    api_version: str
    timestamp: datetime

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Welcome endpoint with API information."""
    return {
        "message": "Welcome to the Vector Search API!",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        vector_count=len(search_engine.vectors),
        api_version="1.0.0",
        timestamp=datetime.now()
    )

@app.post("/vectors", response_model=Dict[str, str])
async def add_vector(vector_data: VectorModel):
    """
    Add a vector to the search index.
    
    - **vector**: List of float values representing the vector
    - **metadata**: Dictionary of associated metadata
    """
    try:
        vector_array = np.array(vector_data.vector)
        search_engine.add_vector(vector_array, vector_data.metadata)
        
        return {
            "message": "Vector added successfully",
            "vector_id": len(search_engine.vectors) - 1,
            "dimensions": len(vector_data.vector)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error adding vector: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_vectors(search_request: SearchRequest):
    """
    Search for similar vectors.
    
    - **query_vector**: Vector to search for
    - **top_k**: Number of results to return (1-100)
    """
    start_time = datetime.now()
    
    try:
        query_array = np.array(search_request.query_vector)
        raw_results = search_engine.search(query_array, search_request.top_k)
        
        # Format results
        formatted_results = []
        for rank, (metadata, score) in enumerate(raw_results, 1):
            formatted_results.append(SearchResult(
                metadata=metadata,
                similarity_score=score,
                rank=rank
            ))
        
        end_time = datetime.now()
        search_time_ms = (end_time - start_time).total_seconds() * 1000
        
        return SearchResponse(
            query_vector=search_request.query_vector,
            results=formatted_results,
            total_results=len(formatted_results),
            search_time_ms=search_time_ms,
            timestamp=end_time
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Search error: {str(e)}")

@app.get("/vectors", response_model=Dict[str, any])
async def get_vectors(
    limit: int = Query(default=10, ge=1, le=100, description="Number of vectors to return"),
    offset: int = Query(default=0, ge=0, description="Number of vectors to skip")
):
    """
    Get stored vectors with pagination.
    
    - **limit**: Maximum number of vectors to return
    - **offset**: Number of vectors to skip (for pagination)
    """
    total_vectors = len(search_engine.vectors)
    
    if offset >= total_vectors:
        return {
            "vectors": [],
            "total": total_vectors,
            "limit": limit,
            "offset": offset,
            "has_more": False
        }
    
    end_idx = min(offset + limit, total_vectors)
    vectors_slice = search_engine.vectors[offset:end_idx]
    metadata_slice = search_engine.metadata[offset:end_idx]
    
    # Format vectors for response
    formatted_vectors = []
    for i, (vector, metadata) in enumerate(zip(vectors_slice, metadata_slice)):
        formatted_vectors.append({
            "id": offset + i,
            "vector": vector.tolist(),
            "metadata": metadata,
            "dimensions": len(vector)
        })
    
    return {
        "vectors": formatted_vectors,
        "total": total_vectors,
        "limit": limit,
        "offset": offset,
        "has_more": end_idx < total_vectors
    }

@app.delete("/vectors", response_model=Dict[str, str])
async def clear_vectors():
    """Clear all vectors from the search index."""
    global search_engine
    search_engine = SimpleVectorSearch("cosine")
    
    return {
        "message": "All vectors cleared successfully",
        "vector_count": 0
    }

@app.get("/stats", response_model=Dict[str, any])
async def get_statistics():
    """Get statistics about the vector search index."""
    if len(search_engine.vectors) == 0:
        return {
            "vector_count": 0,
            "dimensions": 0,
            "similarity_metric": "cosine",
            "memory_usage_mb": 0
        }
    
    vectors_array = np.array(search_engine.vectors)
    memory_bytes = vectors_array.nbytes + sum(len(str(m)) for m in search_engine.metadata)
    
    return {
        "vector_count": len(search_engine.vectors),
        "dimensions": vectors_array.shape[1] if len(vectors_array.shape) > 1 else 0,
        "similarity_metric": getattr(search_engine, 'metric', 'cosine'),
        "memory_usage_mb": round(memory_bytes / (1024 * 1024), 2),
        "vector_shape": vectors_array.shape
    }

# Sample data initialization
@app.on_event("startup")
async def startup_event():
    """Initialize the API with sample data."""
    print("🚀 Starting Vector Search API...")
    
    # Add some sample vectors for demonstration
    sample_vectors = [
        {
            "vector": [0.8, 0.1, 0.2, 0.3, 0.1],
            "metadata": {"title": "Technology Article", "category": "tech", "id": 1}
        },
        {
            "vector": [0.1, 0.9, 0.1, 0.1, 0.2],
            "metadata": {"title": "Pet Care Guide", "category": "animals", "id": 2}
        },
        {
            "vector": [0.2, 0.1, 0.8, 0.2, 0.1],
            "metadata": {"title": "Cooking Recipe", "category": "food", "id": 3}
        },
        {
            "vector": [0.1, 0.2, 0.1, 0.8, 0.1],
            "metadata": {"title": "Sports News", "category": "sports", "id": 4}
        }
    ]
    
    for sample in sample_vectors:
        vector_array = np.array(sample["vector"])
        search_engine.add_vector(vector_array, sample["metadata"])
    
    print(f"✅ API initialized with {len(sample_vectors)} sample vectors")
    print("📚 Visit http://localhost:8000/docs for interactive documentation")

def main():
    """Run the FastAPI application."""
    print("🌐 Starting Vector Search API Server")
    print("=" * 35)
    print("This API provides endpoints for vector similarity search.")
    print()
    print("Available endpoints:")
    print("• GET  /          - API information")
    print("• GET  /health    - Health check")
    print("• POST /vectors   - Add vector")
    print("• POST /search    - Search vectors")
    print("• GET  /vectors   - List vectors")
    print("• GET  /stats     - Index statistics")
    print()
    print("Interactive documentation:")
    print("• Swagger UI: http://localhost:8000/docs")
    print("• ReDoc:      http://localhost:8000/redoc")
    print()
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
