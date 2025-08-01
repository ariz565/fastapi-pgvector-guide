# API Route Handlers for Semantic Search
# Clean function-based FastAPI route handlers with proper separation

from fastapi import APIRouter, HTTPException, BackgroundTasks, Query, Depends
from fastapi.responses import JSONResponse
from typing import List, Dict, Optional, Union
import logging
import time
from datetime import datetime
import uuid

from models import (
    DocumentCreate, DocumentBatch, SearchRequest, HybridSearchRequest,
    SimilarDocumentsRequest, SearchResponse, HybridSearchResponse,
    SimilarDocumentsResponse, CategoryResponse, HealthResponse,
    DatabaseStats, AnalyticsResponse, ErrorResponse
)
from search import (
    semantic_search, hybrid_search, find_similar_documents,
    search_by_category, multi_query_search, faceted_search,
    search_suggestions
)
from database import (
    insert_document, insert_documents_batch, get_document_by_id,
    get_documents_without_embeddings, update_document_embedding,
    get_all_categories, get_search_analytics, delete_document,
    get_document_count, get_embedding_coverage
)
from embeddings import (
    generate_embedding, generate_embeddings_batch,
    get_embedding_model_info, calculate_text_similarity,
    benchmark_embedding_generation
)
from config import test_database_connection, get_database_stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Global variables for tracking
app_start_time = time.time()

def generate_request_id() -> str:
    """Generate a unique request ID for tracking"""
    return str(uuid.uuid4())[:8]

def create_error_response(error_message: str, detail: Optional[str] = None, 
                         request_id: Optional[str] = None) -> Dict:
    """
    Create a standardized error response
    
    Args:
        error_message: Main error message
        detail: Optional detailed error information
        request_id: Optional request ID for tracking
        
    Returns:
        Error response dictionary
    """
    return {
        "error": error_message,
        "detail": detail,
        "timestamp": datetime.now().isoformat(),
        "request_id": request_id
    }

# Health and Status Endpoints

@router.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """
    Check the health status of the semantic search system
    
    Returns comprehensive health information including:
    - Database connectivity
    - Embedding model status
    - Document statistics
    - System uptime
    """
    try:
        request_id = generate_request_id()
        logger.info(f"🏥 Health check requested [ID: {request_id}]")
        
        # Test database connection
        db_connected = test_database_connection()
        
        # Get embedding model info
        model_info = get_embedding_model_info()
        embedding_model_loaded = model_info.get('loaded', False)
        
        # Get document statistics
        total_docs, docs_with_embeddings, coverage = get_embedding_coverage()
        
        # Calculate uptime
        uptime_seconds = time.time() - app_start_time
        
        health_data = {
            "status": "healthy" if db_connected and embedding_model_loaded else "degraded",
            "database_connected": db_connected,
            "embedding_model_loaded": embedding_model_loaded,
            "total_documents": total_docs,
            "documents_with_embeddings": docs_with_embeddings,
            "embedding_coverage_percent": round(coverage, 2),
            "uptime_seconds": round(uptime_seconds, 2),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"✅ Health check completed [ID: {request_id}]")
        return health_data
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Health check failed", str(e), request_id)
        )

@router.get("/stats", response_model=DatabaseStats, summary="Database Statistics")
async def get_stats():
    """
    Get comprehensive database and system statistics
    
    Returns detailed statistics about:
    - Document counts and embedding coverage
    - Search analytics
    - Category distribution
    - Performance metrics
    """
    try:
        request_id = generate_request_id()
        logger.info(f"📊 Statistics requested [ID: {request_id}]")
        
        # Get database statistics
        db_stats = get_database_stats()
        
        if db_stats is None:
            raise HTTPException(
                status_code=500,
                detail=create_error_response("Failed to retrieve database statistics", None, request_id)
            )
        
        # Get category information
        categories = get_all_categories()
        
        # Add categories to response
        db_stats['categories'] = categories
        
        logger.info(f"✅ Statistics retrieved [ID: {request_id}]")
        return db_stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Error retrieving statistics", str(e), request_id)
        )

# Search Endpoints

@router.post("/search", response_model=SearchResponse, summary="Semantic Search")
async def search_documents(request: SearchRequest):
    """
    Perform semantic search using vector similarity
    
    This endpoint finds documents that are semantically similar to the query text
    using vector embeddings and cosine similarity.
    
    Features:
    - Vector-based semantic similarity search
    - Optional category filtering
    - Configurable similarity threshold
    - Detailed timing information
    """
    try:
        request_id = generate_request_id()
        logger.info(f"🔍 Search request: '{request.query[:50]}...' [ID: {request_id}]")
        
        # Perform semantic search
        results = semantic_search(
            query=request.query,
            limit=request.limit,
            category_filter=request.category_filter,
            similarity_threshold=request.similarity_threshold
        )
        
        # Add request ID to response
        results['request_id'] = request_id
        
        logger.info(f"✅ Search completed: {results['total_results']} results [ID: {request_id}]")
        return results
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Search failed", str(e), generate_request_id())
        )

@router.post("/search/hybrid", response_model=HybridSearchResponse, summary="Hybrid Search")
async def hybrid_search_documents(request: HybridSearchRequest):
    """
    Perform hybrid search combining text and vector similarity
    
    This endpoint combines traditional full-text search with vector similarity
    to provide more comprehensive results that match both keywords and semantic meaning.
    
    Features:
    - Combines PostgreSQL full-text search with vector similarity
    - Configurable weights for text vs vector components
    - Detailed scoring breakdown
    - Category filtering support
    """
    try:
        request_id = generate_request_id()
        logger.info(f"🔍 Hybrid search: '{request.query[:50]}...' [ID: {request_id}]")
        
        # Perform hybrid search
        results = hybrid_search(
            query=request.query,
            text_weight=request.text_weight,
            vector_weight=request.vector_weight,
            limit=request.limit,
            category_filter=request.category_filter
        )
        
        # Add request ID to response
        results['request_id'] = request_id
        
        logger.info(f"✅ Hybrid search completed: {results['total_results']} results [ID: {request_id}]")
        return results
        
    except Exception as e:
        logger.error(f"❌ Hybrid search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Hybrid search failed", str(e), generate_request_id())
        )

@router.post("/search/similar", response_model=SimilarDocumentsResponse, summary="Find Similar Documents")
async def find_similar(request: SimilarDocumentsRequest):
    """
    Find documents similar to a specific document
    
    This endpoint finds documents that are semantically similar to a reference document
    using vector similarity comparison.
    
    Features:
    - Document-to-document similarity search
    - Optional category exclusion filtering
    - Similarity score ranking
    - Reference document information
    """
    try:
        request_id = generate_request_id()
        logger.info(f"🔗 Similar documents for ID {request.document_id} [ID: {request_id}]")
        
        # Find similar documents
        results = find_similar_documents(
            document_id=request.document_id,
            limit=request.limit,
            exclude_same_category=request.exclude_same_category
        )
        
        # Check for errors
        if 'error' in results:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(results['error'], None, request_id)
            )
        
        # Add request ID to response
        results['request_id'] = request_id
        
        logger.info(f"✅ Found {results['total_results']} similar documents [ID: {request_id}]")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Similar documents error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Similar documents search failed", str(e), generate_request_id())
        )

@router.get("/search/category/{category}", response_model=CategoryResponse, summary="Search by Category")
async def search_category(
    category: str,
    limit: int = Query(default=50, ge=1, le=100, description="Maximum number of results")
):
    """
    Get documents from a specific category
    
    This endpoint retrieves documents from a specified category,
    ordered by creation date (most recent first).
    
    Features:
    - Category-based document filtering
    - Chronological ordering
    - Configurable result limits
    """
    try:
        request_id = generate_request_id()
        logger.info(f"📁 Category search: '{category}' [ID: {request_id}]")
        
        # Search by category
        results = search_by_category(category=category, limit=limit)
        
        # Check for errors
        if 'error' in results:
            raise HTTPException(
                status_code=500,
                detail=create_error_response(results['error'], None, request_id)
            )
        
        # Add request ID to response
        results['request_id'] = request_id
        
        logger.info(f"✅ Category search completed: {results['total_results']} results [ID: {request_id}]")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Category search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Category search failed", str(e), generate_request_id())
        )

@router.post("/search/multi", summary="Multi-Query Search")
async def multi_query_search_endpoint(
    queries: List[str],
    strategy: str = Query(default="average", description="Combination strategy: average, union, intersection"),
    limit: int = Query(default=10, ge=1, le=100, description="Maximum number of results")
):
    """
    Perform search with multiple queries using different combination strategies
    
    This endpoint allows searching with multiple queries and combining results
    using different strategies for more comprehensive search capabilities.
    
    Strategies:
    - average: Average query embeddings and search once
    - union: Combine results from all queries
    - intersection: Find documents that match all queries
    """
    try:
        request_id = generate_request_id()
        logger.info(f"🔍 Multi-query search: {len(queries)} queries, strategy: {strategy} [ID: {request_id}]")
        
        # Perform multi-query search
        results = multi_query_search(
            queries=queries,
            limit=limit,
            strategy=strategy
        )
        
        # Check for errors
        if 'error' in results:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(results['error'], None, request_id)
            )
        
        # Add request ID to response
        results['request_id'] = request_id
        
        logger.info(f"✅ Multi-query search completed: {results['total_results']} results [ID: {request_id}]")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Multi-query search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Multi-query search failed", str(e), generate_request_id())
        )

@router.get("/search/suggestions", summary="Search Suggestions")
async def get_search_suggestions(
    q: str = Query(..., min_length=1, description="Partial query for suggestions"),
    limit: int = Query(default=5, ge=1, le=10, description="Maximum number of suggestions")
):
    """
    Get search suggestions based on partial query
    
    This endpoint provides search suggestions to help users formulate better queries.
    Useful for implementing autocomplete functionality.
    """
    try:
        request_id = generate_request_id()
        logger.info(f"💡 Search suggestions for: '{q}' [ID: {request_id}]")
        
        # Get search suggestions
        suggestions = search_suggestions(partial_query=q, limit=limit)
        
        response = {
            'partial_query': q,
            'suggestions': suggestions,
            'count': len(suggestions),
            'request_id': request_id
        }
        
        logger.info(f"✅ Generated {len(suggestions)} suggestions [ID: {request_id}]")
        return response
        
    except Exception as e:
        logger.error(f"❌ Search suggestions error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Search suggestions failed", str(e), generate_request_id())
        )

# Document Management Endpoints

@router.post("/documents", summary="Add Document")
async def add_document(document: DocumentCreate, background_tasks: BackgroundTasks):
    """
    Add a new document to the search index
    
    This endpoint adds a new document and generates its embedding in the background.
    The document becomes searchable once the embedding is generated.
    
    Features:
    - Immediate document storage
    - Background embedding generation
    - Input validation and sanitization
    """
    try:
        request_id = generate_request_id()
        logger.info(f"📄 Adding document: '{document.title[:50]}...' [ID: {request_id}]")
        
        # Insert document without embedding first
        doc_id = insert_document(
            title=document.title,
            content=document.content,
            category=document.category,
            url=document.url
        )
        
        if doc_id is None:
            raise HTTPException(
                status_code=500,
                detail=create_error_response("Failed to insert document", None, request_id)
            )
        
        # Generate embedding in background
        background_tasks.add_task(generate_document_embedding, doc_id, document.content)
        
        response = {
            'id': doc_id,
            'title': document.title,
            'status': 'created',
            'embedding_status': 'generating',
            'message': 'Document created successfully. Embedding generation in progress.',
            'request_id': request_id
        }
        
        logger.info(f"✅ Document created with ID {doc_id} [ID: {request_id}]")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding document: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Failed to add document", str(e), generate_request_id())
        )

@router.post("/documents/batch", summary="Add Documents in Batch")
async def add_documents_batch(batch: DocumentBatch, background_tasks: BackgroundTasks):
    """
    Add multiple documents in a batch operation
    
    This endpoint efficiently adds multiple documents and generates embeddings
    in the background for better performance.
    
    Features:
    - Batch document insertion
    - Efficient embedding generation
    - Progress tracking
    - Transaction safety
    """
    try:
        request_id = generate_request_id()
        logger.info(f"📄 Adding document batch: {len(batch.documents)} documents [ID: {request_id}]")
        
        # Prepare documents for batch insertion
        documents_data = []
        for doc in batch.documents:
            doc_data = {
                'title': doc.title,
                'content': doc.content,
                'category': doc.category,
                'url': doc.url,
                'embedding': None  # Will be generated in background
            }
            documents_data.append(doc_data)
        
        # Insert documents in batch
        doc_ids = insert_documents_batch(documents_data)
        
        if not doc_ids:
            raise HTTPException(
                status_code=500,
                detail=create_error_response("Failed to insert document batch", None, request_id)
            )
        
        # Generate embeddings in background
        background_tasks.add_task(generate_batch_embeddings, doc_ids, [doc.content for doc in batch.documents])
        
        response = {
            'inserted_count': len(doc_ids),
            'document_ids': doc_ids,
            'status': 'created',
            'embedding_status': 'generating',
            'message': f'Batch of {len(doc_ids)} documents created successfully. Embedding generation in progress.',
            'request_id': request_id
        }
        
        logger.info(f"✅ Document batch created: {len(doc_ids)} documents [ID: {request_id}]")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error adding document batch: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Failed to add document batch", str(e), generate_request_id())
        )

@router.get("/documents/{document_id}", summary="Get Document")
async def get_document(document_id: int):
    """
    Retrieve a specific document by ID
    
    Returns detailed information about a document including metadata.
    """
    try:
        request_id = generate_request_id()
        logger.info(f"📄 Getting document {document_id} [ID: {request_id}]")
        
        # Get document from database
        document = get_document_by_id(document_id)
        
        if document is None:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(f"Document {document_id} not found", None, request_id)
            )
        
        # Add request ID to response
        document['request_id'] = request_id
        
        logger.info(f"✅ Document {document_id} retrieved [ID: {request_id}]")
        return document
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Failed to retrieve document", str(e), generate_request_id())
        )

@router.delete("/documents/{document_id}", summary="Delete Document")
async def delete_document_endpoint(document_id: int):
    """
    Delete a document by ID
    
    Permanently removes a document from the search index.
    """
    try:
        request_id = generate_request_id()
        logger.info(f"🗑️ Deleting document {document_id} [ID: {request_id}]")
        
        # Delete document from database
        success = delete_document(document_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(f"Document {document_id} not found", None, request_id)
            )
        
        response = {
            'document_id': document_id,
            'status': 'deleted',
            'message': 'Document deleted successfully',
            'request_id': request_id
        }
        
        logger.info(f"✅ Document {document_id} deleted [ID: {request_id}]")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting document: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Failed to delete document", str(e), generate_request_id())
        )

# Analytics Endpoints

@router.get("/analytics", response_model=AnalyticsResponse, summary="Search Analytics")
async def get_analytics(
    days: int = Query(default=7, ge=1, le=30, description="Number of days to analyze")
):
    """
    Get search analytics and usage statistics
    
    This endpoint provides insights into search patterns, performance,
    and usage trends over a specified time period.
    
    Features:
    - Search volume and performance metrics
    - Popular search terms
    - Daily performance trends
    - Configurable time windows
    """
    try:
        request_id = generate_request_id()
        logger.info(f"📊 Analytics requested for {days} days [ID: {request_id}]")
        
        # Get search analytics
        analytics = get_search_analytics(days)
        
        if not analytics:
            # Return empty analytics if no data
            analytics = {
                'total_searches': 0,
                'avg_search_time_ms': 0,
                'avg_results_count': 0,
                'top_queries': [],
                'period_days': days
            }
        
        # Format response
        response = {
            'period_days': days,
            'total_searches': analytics.get('total_searches', 0),
            'avg_search_time_ms': analytics.get('avg_search_time_ms', 0),
            'avg_results_count': analytics.get('avg_results_count', 0),
            'top_queries': analytics.get('top_queries', []),
            'daily_performance': analytics.get('daily_performance', []),
            'generated_at': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Analytics generated [ID: {request_id}]")
        return response
        
    except Exception as e:
        logger.error(f"❌ Analytics error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Failed to generate analytics", str(e), generate_request_id())
        )

# Utility Endpoints

@router.get("/categories", summary="List Categories")
async def list_categories():
    """
    Get all document categories with statistics
    
    Returns a list of all categories in the system along with document counts
    and embedding coverage information.
    """
    try:
        request_id = generate_request_id()
        logger.info(f"📁 Categories list requested [ID: {request_id}]")
        
        # Get all categories
        categories = get_all_categories()
        
        response = {
            'categories': categories,
            'total_categories': len(categories),
            'request_id': request_id
        }
        
        logger.info(f"✅ Categories list returned: {len(categories)} categories [ID: {request_id}]")
        return response
        
    except Exception as e:
        logger.error(f"❌ Categories list error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Failed to list categories", str(e), generate_request_id())
        )

@router.get("/embeddings/info", summary="Embedding Model Information")
async def embedding_model_info():
    """
    Get information about the current embedding model
    
    Returns details about the embedding model being used,
    including model name, dimensions, and performance characteristics.
    """
    try:
        request_id = generate_request_id()
        logger.info(f"🔍 Embedding model info requested [ID: {request_id}]")
        
        # Get embedding model information
        model_info = get_embedding_model_info()
        
        # Add request ID to response
        model_info['request_id'] = request_id
        
        logger.info(f"✅ Embedding model info returned [ID: {request_id}]")
        return model_info
        
    except Exception as e:
        logger.error(f"❌ Embedding model info error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Failed to get embedding model info", str(e), generate_request_id())
        )

@router.post("/embeddings/benchmark", summary="Benchmark Embedding Generation")
async def benchmark_embeddings(num_samples: int = Query(default=100, ge=10, le=1000)):
    """
    Benchmark embedding generation performance
    
    This endpoint runs performance tests on the embedding generation system
    to measure throughput and latency characteristics.
    """
    try:
        request_id = generate_request_id()
        logger.info(f"⚡ Embedding benchmark: {num_samples} samples [ID: {request_id}]")
        
        # Run benchmark
        benchmark_results = benchmark_embedding_generation(num_samples)
        
        # Add request ID to response
        benchmark_results['request_id'] = request_id
        
        logger.info(f"✅ Embedding benchmark completed [ID: {request_id}]")
        return benchmark_results
        
    except Exception as e:
        logger.error(f"❌ Embedding benchmark error: {e}")
        raise HTTPException(
            status_code=500,
            detail=create_error_response("Embedding benchmark failed", str(e), generate_request_id())
        )

# Background Task Functions

async def generate_document_embedding(doc_id: int, content: str):
    """
    Background task to generate embedding for a single document
    
    Args:
        doc_id: Document ID
        content: Document content
    """
    try:
        logger.info(f"🔄 Generating embedding for document {doc_id}")
        
        # Generate embedding
        embedding = generate_embedding(content)
        
        if embedding is not None:
            # Update document with embedding
            success = update_document_embedding(doc_id, embedding)
            if success:
                logger.info(f"✅ Embedding generated for document {doc_id}")
            else:
                logger.error(f"❌ Failed to update embedding for document {doc_id}")
        else:
            logger.error(f"❌ Failed to generate embedding for document {doc_id}")
            
    except Exception as e:
        logger.error(f"❌ Error in background embedding generation: {e}")

async def generate_batch_embeddings(doc_ids: List[int], contents: List[str]):
    """
    Background task to generate embeddings for multiple documents
    
    Args:
        doc_ids: List of document IDs
        contents: List of document contents
    """
    try:
        logger.info(f"🔄 Generating embeddings for {len(doc_ids)} documents")
        
        # Generate embeddings in batch
        embeddings = generate_embeddings_batch(contents)
        
        # Update documents with embeddings
        success_count = 0
        for doc_id, embedding in zip(doc_ids, embeddings):
            if embedding is not None:
                success = update_document_embedding(doc_id, embedding)
                if success:
                    success_count += 1
        
        logger.info(f"✅ Generated embeddings for {success_count}/{len(doc_ids)} documents")
        
    except Exception as e:
        logger.error(f"❌ Error in background batch embedding generation: {e}")
