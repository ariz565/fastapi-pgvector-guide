# FastAPI Semantic Search Application
# Production-ready REST API for semantic document search using PostgreSQL + pgvector

from fastapi import FastAPI, HTTPException, Query, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
import logging
from datetime import datetime
import time
import asyncio
from contextlib import asynccontextmanager

# Import our modules
from database_config import VectorDatabase, DatabaseConfig
from document_indexer import DocumentIndexer
from semantic_search import SemanticSearchEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
db_instance = None
indexer_instance = None
search_engine_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting Semantic Search API...")
    
    global db_instance, indexer_instance, search_engine_instance
    
    # Initialize database connection
    try:
        db_instance = VectorDatabase()
        
        # Test database connection
        if not db_instance.test_connection():
            logger.error("❌ Database connection failed!")
            raise Exception("Database connection failed")
        
        # Initialize indexer and search engine
        indexer_instance = DocumentIndexer()
        search_engine_instance = SemanticSearchEngine()
        
        logger.info("✅ Semantic Search API started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Semantic Search API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Semantic Search API",
    description="High-performance semantic document search using PostgreSQL + pgvector",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware for web applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models using Pydantic

class DocumentCreate(BaseModel):
    """Model for creating new documents"""
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(..., min_length=1, description="Document content/text")
    category: Optional[str] = Field(None, max_length=100, description="Document category")
    url: Optional[str] = Field(None, max_length=1000, description="Source URL")
    
    @validator('title', 'content')
    def validate_text_fields(cls, v):
        # Remove excessive whitespace and validate content
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()

class DocumentResponse(BaseModel):
    """Model for document responses"""
    id: int
    title: str
    content_preview: str
    category: Optional[str]
    url: Optional[str]
    created_at: Optional[datetime]
    similarity_score: Optional[float] = None

class SearchRequest(BaseModel):
    """Model for search requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results")
    category_filter: Optional[str] = Field(None, description="Filter by category")
    similarity_threshold: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")

class HybridSearchRequest(BaseModel):
    """Model for hybrid search requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    text_weight: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Weight for text search")
    vector_weight: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Weight for vector search")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum number of results")
    category_filter: Optional[str] = Field(None, description="Filter by category")
    
    @validator('text_weight', 'vector_weight')
    def validate_weights(cls, v, values):
        # Ensure weights sum to approximately 1.0
        text_w = values.get('text_weight', 0.3)
        vector_w = v if 'text_weight' in values else values.get('vector_weight', 0.7)
        total = text_w + vector_w
        if abs(total - 1.0) > 0.1:
            raise ValueError('text_weight + vector_weight should sum to approximately 1.0')
        return v

class SearchResponse(BaseModel):
    """Model for search responses"""
    query: str
    search_type: str = "semantic"
    results: List[DocumentResponse]
    total_results: int
    search_time_ms: float
    filters: Optional[Dict] = None
    timestamp: datetime

class HealthResponse(BaseModel):
    """Model for health check responses"""
    status: str
    database_connected: bool
    total_documents: int
    documents_with_embeddings: int
    api_version: str
    timestamp: datetime

class BatchDocumentCreate(BaseModel):
    """Model for batch document creation"""
    documents: List[DocumentCreate] = Field(..., min_items=1, max_items=100, description="List of documents to create")

# Dependency injection functions

def get_database():
    """Get database instance"""
    if db_instance is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    return db_instance

def get_indexer():
    """Get document indexer instance"""
    if indexer_instance is None:
        raise HTTPException(status_code=503, detail="Document indexer not initialized")
    return indexer_instance

def get_search_engine():
    """Get search engine instance"""
    if search_engine_instance is None:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    return search_engine_instance

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Welcome endpoint with API information
    """
    return {
        "message": "Welcome to the Semantic Search API!",
        "description": "High-performance semantic document search using PostgreSQL + pgvector",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check(db: VectorDatabase = Depends(get_database)):
    """
    Health check endpoint with detailed system status
    """
    try:
        # Test database connection
        db_connected = db.test_connection()
        
        # Get database statistics
        stats = db.get_database_stats()
        total_docs = stats['documents']['total_documents'] if stats else 0
        docs_with_embeddings = stats['documents']['documents_with_embeddings'] if stats else 0
        
        status = "healthy" if db_connected else "unhealthy"
        
        return HealthResponse(
            status=status,
            database_connected=db_connected,
            total_documents=total_docs,
            documents_with_embeddings=docs_with_embeddings,
            api_version="1.0.0",
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/documents", response_model=Dict[str, Union[str, int]])
async def create_document(
    document: DocumentCreate,
    background_tasks: BackgroundTasks,
    indexer: DocumentIndexer = Depends(get_indexer)
):
    """
    Create a new document with automatic embedding generation
    """
    try:
        # Add document to database with embedding
        doc_id = indexer.add_document(
            title=document.title,
            content=document.content,
            category=document.category,
            url=document.url
        )
        
        if doc_id:
            # Schedule index optimization in background if needed
            background_tasks.add_task(check_and_optimize_indexes, indexer)
            
            return {
                "message": "Document created successfully",
                "document_id": doc_id,
                "title": document.title,
                "category": document.category or "uncategorized"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create document")
            
    except Exception as e:
        logger.error(f"Error creating document: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating document: {str(e)}")

@app.post("/documents/batch", response_model=Dict[str, Union[str, int, List[int]]])
async def create_documents_batch(
    batch: BatchDocumentCreate,
    background_tasks: BackgroundTasks,
    indexer: DocumentIndexer = Depends(get_indexer)
):
    """
    Create multiple documents efficiently using batch processing
    """
    try:
        # Convert Pydantic models to dictionaries
        documents = []
        for doc in batch.documents:
            documents.append({
                'title': doc.title,
                'content': doc.content,
                'category': doc.category,
                'url': doc.url
            })
        
        # Add documents in batch
        doc_ids = indexer.add_documents_batch(documents)
        
        if doc_ids:
            # Schedule index optimization in background
            background_tasks.add_task(check_and_optimize_indexes, indexer)
            
            return {
                "message": f"Successfully created {len(doc_ids)} documents",
                "document_ids": doc_ids,
                "total_created": len(doc_ids),
                "total_requested": len(documents)
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to create documents")
            
    except Exception as e:
        logger.error(f"Error creating documents batch: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating documents: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def semantic_search(
    search_request: SearchRequest,
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """
    Perform semantic similarity search across documents
    """
    try:
        # Execute semantic search
        results = search_engine.search(
            query=search_request.query,
            limit=search_request.limit,
            category_filter=search_request.category_filter,
            similarity_threshold=search_request.similarity_threshold
        )
        
        if 'error' in results:
            raise HTTPException(status_code=500, detail=results['error'])
        
        # Convert to response format
        document_results = []
        for result in results['results']:
            doc_response = DocumentResponse(
                id=result['id'],
                title=result['title'],
                content_preview=result['content_preview'],
                category=result['category'],
                url=result['url'],
                created_at=datetime.fromisoformat(result['created_at']) if result['created_at'] else None,
                similarity_score=result['similarity_score']
            )
            document_results.append(doc_response)
        
        return SearchResponse(
            query=results['query'],
            search_type="semantic",
            results=document_results,
            total_results=results['total_results'],
            search_time_ms=results['search_time_ms'],
            filters=results['filters'],
            timestamp=datetime.fromisoformat(results['timestamp'])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during semantic search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")

@app.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(
    search_request: HybridSearchRequest,
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """
    Perform hybrid search combining text search and vector similarity
    """
    try:
        # Execute hybrid search
        results = search_engine.hybrid_search(
            query=search_request.query,
            text_weight=search_request.text_weight,
            vector_weight=search_request.vector_weight,
            limit=search_request.limit,
            category_filter=search_request.category_filter
        )
        
        if 'error' in results:
            raise HTTPException(status_code=500, detail=results['error'])
        
        # Convert to response format
        document_results = []
        for result in results['results']:
            doc_response = DocumentResponse(
                id=result['id'],
                title=result['title'],
                content_preview=result['content_preview'],
                category=result['category'],
                url=result['url'],
                created_at=datetime.fromisoformat(result['created_at']) if result['created_at'] else None,
                similarity_score=result['combined_score']  # Use combined score for hybrid
            )
            document_results.append(doc_response)
        
        return SearchResponse(
            query=results['query'],
            search_type="hybrid",
            results=document_results,
            total_results=results['total_results'],
            search_time_ms=results['search_time_ms'],
            filters={'weights': results['weights']},
            timestamp=datetime.fromisoformat(results['timestamp'])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during hybrid search: {e}")
        raise HTTPException(status_code=500, detail=f"Hybrid search error: {str(e)}")

@app.get("/documents/{document_id}/similar")
async def find_similar_documents(
    document_id: int,
    limit: int = Query(10, ge=1, le=50, description="Maximum number of similar documents"),
    exclude_same_category: bool = Query(False, description="Exclude documents from same category"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """
    Find documents similar to a given document
    """
    try:
        results = search_engine.find_similar_documents(
            document_id=document_id,
            limit=limit,
            exclude_same_category=exclude_same_category
        )
        
        if 'error' in results:
            raise HTTPException(status_code=404, detail=results['error'])
        
        # Convert to response format
        similar_docs = []
        for result in results['similar_documents']:
            doc_response = DocumentResponse(
                id=result['id'],
                title=result['title'],
                content_preview=result['content_preview'],
                category=result['category'],
                url=result['url'],
                created_at=datetime.fromisoformat(result['created_at']) if result['created_at'] else None,
                similarity_score=result['similarity_score']
            )
            similar_docs.append(doc_response)
        
        return {
            "reference_document": results['reference_document'],
            "similar_documents": similar_docs,
            "total_results": results['total_results'],
            "filters": results['filters']
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error finding similar documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error finding similar documents: {str(e)}")

@app.get("/categories")
async def list_categories(
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """
    Get list of available document categories with counts
    """
    try:
        db = search_engine.db
        stats = db.get_database_stats()
        
        if stats and stats['categories']:
            return {
                "categories": stats['categories'],
                "total_categories": len(stats['categories'])
            }
        else:
            return {"categories": [], "total_categories": 0}
            
    except Exception as e:
        logger.error(f"Error listing categories: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing categories: {str(e)}")

@app.get("/categories/{category}/documents")
async def get_documents_by_category(
    category: str,
    limit: int = Query(20, ge=1, le=100, description="Maximum number of documents"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """
    Get documents from a specific category
    """
    try:
        documents = search_engine.search_by_category(category, limit)
        
        # Convert to response format
        document_results = []
        for doc in documents:
            doc_response = DocumentResponse(
                id=doc['id'],
                title=doc['title'],
                content_preview=doc['content_preview'],
                category=doc['category'],
                url=doc['url'],
                created_at=datetime.fromisoformat(doc['created_at']) if doc['created_at'] else None
            )
            document_results.append(doc_response)
        
        return {
            "category": category,
            "documents": document_results,
            "total_results": len(document_results)
        }
        
    except Exception as e:
        logger.error(f"Error getting documents by category: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting documents: {str(e)}")

@app.get("/analytics/search")
async def get_search_analytics(
    days: int = Query(7, ge=1, le=90, description="Number of days to analyze"),
    search_engine: SemanticSearchEngine = Depends(get_search_engine)
):
    """
    Get search analytics and performance metrics
    """
    try:
        analytics = search_engine.get_search_analytics(days)
        
        return {
            "analytics": analytics,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting search analytics: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting analytics: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    indexer: DocumentIndexer = Depends(get_indexer)
):
    """
    Delete a specific document
    """
    try:
        success = indexer.delete_document(document_id)
        
        if success:
            return {"message": f"Document {document_id} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@app.post("/admin/reindex")
async def reindex_all_documents(
    background_tasks: BackgroundTasks,
    indexer: DocumentIndexer = Depends(get_indexer)
):
    """
    Reindex all documents (regenerate embeddings)
    This is an admin operation that runs in the background
    """
    try:
        # Run reindexing in background
        background_tasks.add_task(reindex_documents_task, indexer)
        
        return {
            "message": "Reindexing started in background",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Error starting reindex: {e}")
        raise HTTPException(status_code=500, detail=f"Error starting reindex: {str(e)}")

# Background Tasks

async def check_and_optimize_indexes(indexer: DocumentIndexer):
    """
    Background task to check and optimize vector indexes when needed
    """
    try:
        # Get current document count
        stats = indexer.db.get_database_stats()
        if stats:
            doc_count = stats['documents']['documents_with_embeddings']
            
            # Create indexes if we have enough documents and none exist
            if doc_count >= 1000:
                logger.info(f"📈 {doc_count} documents detected. Checking vector indexes...")
                indexer.db.create_vector_indexes()
                
    except Exception as e:
        logger.error(f"Error in background index optimization: {e}")

async def reindex_documents_task(indexer: DocumentIndexer):
    """
    Background task for reindexing all documents
    """
    try:
        logger.info("🔄 Starting background reindexing...")
        success = indexer.reindex_all_documents()
        
        if success:
            logger.info("✅ Background reindexing completed successfully")
        else:
            logger.error("❌ Background reindexing failed")
            
    except Exception as e:
        logger.error(f"Error in background reindexing: {e}")

# Error Handlers

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    return JSONResponse(
        status_code=400,
        content={"detail": f"Validation error: {str(exc)}"}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Main function for running the application

def main():
    """
    Main function to run the FastAPI application
    """
    import uvicorn
    
    print("🚀 Starting Semantic Search API Server")
    print("=" * 45)
    print("This API provides semantic document search capabilities using PostgreSQL + pgvector")
    print()
    print("Available endpoints:")
    print("• GET  /              - API information")
    print("• GET  /health        - Health check with detailed status")
    print("• POST /documents     - Create single document")
    print("• POST /documents/batch - Create multiple documents")
    print("• POST /search        - Semantic similarity search")
    print("• POST /search/hybrid - Hybrid text + vector search")
    print("• GET  /documents/{id}/similar - Find similar documents")
    print("• GET  /categories    - List available categories")
    print("• GET  /analytics/search - Search performance analytics")
    print()
    print("Interactive documentation:")
    print("• Swagger UI: http://localhost:8000/docs")
    print("• ReDoc:      http://localhost:8000/redoc")
    print()
    print("Prerequisites:")
    print("• PostgreSQL 17 with pgvector extension")
    print("• Database tables created (run database_config.py first)")
    print("• Sample documents indexed (run document_indexer.py)")
    print()
    
    # Run the server
    uvicorn.run(
        "semantic_search_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
