# Main FastAPI Application for Semantic Search
# Clean function-based FastAPI app with proper initialization and lifecycle management

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
import time
import uvicorn
from datetime import datetime

from routes import router
from config import initialize_database, test_database_connection
from embeddings import initialize_embedding_model, get_embedding_model_info

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application startup time tracking
app_start_time = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager - handles startup and shutdown events
    
    This function manages the complete lifecycle of the application,
    ensuring proper initialization and cleanup of resources.
    """
    global app_start_time
    
    # Startup
    logger.info("🚀 Starting Semantic Search API...")
    app_start_time = time.time()
    
    try:
        # Initialize database
        logger.info("🔧 Initializing database connection...")
        if not initialize_database():
            logger.error("❌ Database initialization failed!")
            raise Exception("Database initialization failed")
        
        # Test database connection
        logger.info("🔍 Testing database connection...")
        if not test_database_connection():
            logger.error("❌ Database connection test failed!")
            raise Exception("Database connection test failed")
        
        # Initialize embedding model
        logger.info("🤖 Initializing embedding model...")
        if not initialize_embedding_model():
            logger.warning("⚠️ Embedding model initialization failed, will retry on first request")
        else:
            model_info = get_embedding_model_info()
            logger.info(f"✅ Embedding model loaded: {model_info.get('model_name')} ({model_info.get('embedding_dimension')}D)")
        
        logger.info("✅ Semantic Search API started successfully")
        logger.info(f"📊 API Documentation available at: http://localhost:8000/docs")
        logger.info(f"🔍 Health check available at: http://localhost:8000/health")
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info("🛑 Shutting down Semantic Search API...")
    
    # Calculate uptime
    if app_start_time:
        uptime = time.time() - app_start_time
        logger.info(f"⏱️ Total uptime: {uptime:.2f} seconds")
    
    logger.info("✅ Semantic Search API shutdown complete")

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application
    
    Returns:
        Configured FastAPI application instance
    """
    # Create FastAPI app with metadata
    app = FastAPI(
        title="Semantic Search API",
        description="""
        **High-performance semantic document search using PostgreSQL + pgvector**
        
        This API provides comprehensive semantic search capabilities including:
        
        - **Semantic Search**: Vector-based similarity search using embeddings
        - **Hybrid Search**: Combines text search with vector similarity 
        - **Document Management**: Add, update, and delete documents
        - **Analytics**: Search performance and usage analytics
        - **Real-time**: Fast search with optimized vector indexes
        
        ## Key Features
        
        - 🔍 **Semantic Understanding**: Finds documents by meaning, not just keywords
        - ⚡ **High Performance**: Optimized PostgreSQL queries with vector indexes
        - 🔄 **Hybrid Search**: Best of both text and semantic search
        - 📊 **Analytics**: Comprehensive search analytics and monitoring
        - 🛡️ **Production Ready**: Error handling, logging, and health checks
        
        ## Getting Started
        
        1. Check system health: `GET /health`
        2. Add documents: `POST /documents` or `POST /documents/batch`
        3. Search documents: `POST /search` or `POST /search/hybrid`
        4. View analytics: `GET /analytics`
        
        ## Search Types
        
        - **Semantic**: Pure vector similarity search
        - **Hybrid**: Combined text + vector search with configurable weights
        - **Category**: Filter by document category
        - **Similar**: Find documents similar to a reference document
        
        """,
        version="1.0.0",
        contact={
            "name": "Semantic Search API",
            "url": "https://github.com/your-repo/semantic-search",
        },
        license_info={
            "name": "MIT",
            "url": "https://opensource.org/licenses/MIT",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
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
    
    # Include the API routes
    app.include_router(router, prefix="/api/v1")
    
    # Add global exception handlers
    add_exception_handlers(app)
    
    # Add custom middleware
    add_custom_middleware(app)
    
    return app

def add_exception_handlers(app: FastAPI):
    """
    Add global exception handlers for better error responses
    
    Args:
        app: FastAPI application instance
    """
    
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle request validation errors with detailed messages"""
        logger.warning(f"Validation error for {request.url}: {exc}")
        
        return JSONResponse(
            status_code=422,
            content={
                "error": "Request validation failed",
                "detail": "Invalid request data provided",
                "validation_errors": exc.errors(),
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected error for {request.url}: {exc}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": "An unexpected error occurred while processing your request",
                "timestamp": datetime.now().isoformat(),
                "path": str(request.url)
            }
        )

def add_custom_middleware(app: FastAPI):
    """
    Add custom middleware for logging and monitoring
    
    Args:
        app: FastAPI application instance
    """
    
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all incoming requests with timing information"""
        start_time = time.time()
        
        # Log request start
        logger.info(f"➡️ {request.method} {request.url.path} - Client: {request.client.host if request.client else 'unknown'}")
        
        # Process request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000
        
        # Log response
        logger.info(f"⬅️ {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.2f}ms")
        
        # Add timing header
        response.headers["X-Process-Time"] = str(process_time)
        
        return response

# Create the application instance
app = create_app()

# Root endpoint for basic API information
@app.get("/", summary="API Root", tags=["Root"])
async def root():
    """
    API root endpoint providing basic information
    
    Returns general information about the Semantic Search API
    including version, status, and available endpoints.
    """
    return {
        "name": "Semantic Search API",
        "version": "1.0.0",
        "description": "High-performance semantic document search using PostgreSQL + pgvector",
        "status": "running",
        "docs_url": "/docs",
        "health_check": "/api/v1/health",
        "search_endpoint": "/api/v1/search",
        "timestamp": datetime.now().isoformat(),
        "features": [
            "Semantic vector search",
            "Hybrid text + vector search", 
            "Document management",
            "Search analytics",
            "Real-time performance monitoring"
        ]
    }

# Additional utility endpoints
@app.get("/version", summary="API Version", tags=["Utility"])
async def get_version():
    """Get API version information"""
    return {
        "version": "1.0.0",
        "api_name": "Semantic Search API",
        "build_date": "2024-01-01",
        "python_version": "3.8+",
        "framework": "FastAPI",
        "database": "PostgreSQL + pgvector"
    }

@app.get("/ping", summary="Ping", tags=["Utility"])
async def ping():
    """Simple ping endpoint for basic connectivity testing"""
    return {
        "status": "ok",
        "message": "pong",
        "timestamp": datetime.now().isoformat()
    }

def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server with uvicorn
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        reload: Enable auto-reload for development
    """
    logger.info(f"🚀 Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
        access_log=True
    )

if __name__ == "__main__":
    # Run the server if this file is executed directly
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Semantic Search API")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, reload=args.reload)
