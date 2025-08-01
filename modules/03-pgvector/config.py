# Configuration and Database Connection Setup
# Clean function-based approach for PostgreSQL + pgvector configuration

import os
import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration constants
DEFAULT_DB_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'semantic_search',
    'user': 'postgres',
    'password': 'postgres'
}

def get_database_config() -> Dict[str, Any]:
    """
    Get database configuration from environment variables or defaults
    
    Returns:
        Dict containing database connection parameters
    """
    config = {
        'host': os.getenv('DB_HOST', DEFAULT_DB_CONFIG['host']),
        'port': int(os.getenv('DB_PORT', DEFAULT_DB_CONFIG['port'])),
        'database': os.getenv('DB_NAME', DEFAULT_DB_CONFIG['database']),
        'user': os.getenv('DB_USER', DEFAULT_DB_CONFIG['user']),
        'password': os.getenv('DB_PASSWORD', DEFAULT_DB_CONFIG['password'])
    }
    
    logger.info(f"Database config: {config['host']}:{config['port']}/{config['database']}")
    return config

@contextmanager
def get_db_connection():
    """
    Context manager for database connections
    
    Yields:
        psycopg2 connection object
    """
    config = get_database_config()
    connection = None
    
    try:
        # Create database connection
        connection = psycopg2.connect(
            cursor_factory=RealDictCursor,
            **config
        )
        connection.autocommit = True
        yield connection
        
    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        if connection:
            connection.rollback()
        raise
        
    finally:
        if connection:
            connection.close()

def test_database_connection() -> bool:
    """
    Test database connectivity and pgvector extension
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Test basic connection
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                logger.info(f"PostgreSQL version: {version['version']}")
                
                # Test pgvector extension
                cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                result = cursor.fetchone()
                
                if result:
                    logger.info("✅ pgvector extension is available")
                    return True
                else:
                    logger.warning("⚠️ pgvector extension not found")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Database connection test failed: {e}")
        return False

def setup_pgvector_extension() -> bool:
    """
    Install and configure pgvector extension
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Create extension if it doesn't exist
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Verify installation
                cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
                result = cursor.fetchone()
                
                if result:
                    logger.info("✅ pgvector extension installed successfully")
                    return True
                else:
                    logger.error("❌ Failed to install pgvector extension")
                    return False
                    
    except psycopg2.Error as e:
        logger.error(f"❌ Error setting up pgvector: {e}")
        return False

def create_database_schema() -> bool:
    """
    Create all necessary tables and indexes for semantic search
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Create documents table with vector column
                create_documents_table = """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    content TEXT NOT NULL,
                    category VARCHAR(100),
                    url VARCHAR(1000),
                    embedding VECTOR(384),  -- Sentence-BERT embeddings are 384-dimensional
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(create_documents_table)
                
                # Create search analytics table
                create_analytics_table = """
                CREATE TABLE IF NOT EXISTS search_analytics (
                    id SERIAL PRIMARY KEY,
                    query TEXT NOT NULL,
                    search_type VARCHAR(50) NOT NULL,
                    results_count INTEGER,
                    search_time_ms REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(create_analytics_table)
                
                # Create full-text search index for hybrid search
                create_text_index = """
                CREATE INDEX IF NOT EXISTS idx_documents_content_fts 
                ON documents USING gin(to_tsvector('english', content));
                """
                cursor.execute(create_text_index)
                
                # Create updated_at trigger function
                create_trigger_function = """
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql';
                """
                cursor.execute(create_trigger_function)
                
                # Create trigger for documents table
                create_trigger = """
                DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
                CREATE TRIGGER update_documents_updated_at
                    BEFORE UPDATE ON documents
                    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
                """
                cursor.execute(create_trigger)
                
                logger.info("✅ Database schema created successfully")
                return True
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error creating database schema: {e}")
        return False

def create_vector_indexes() -> bool:
    """
    Create optimized vector indexes for similarity search
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Check if we have enough documents for indexing
                cursor.execute("SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL;")
                doc_count = cursor.fetchone()['count']
                
                if doc_count < 100:
                    logger.info(f"Only {doc_count} documents found. Skipping vector index creation.")
                    return True
                
                # Create HNSW index for fast approximate search
                create_hnsw_index = """
                CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw 
                ON documents USING hnsw (embedding vector_cosine_ops) 
                WITH (m = 16, ef_construction = 64);
                """
                cursor.execute(create_hnsw_index)
                
                # Create IVF index for large datasets (alternative to HNSW)
                # Uncomment if you have very large datasets (>1M documents)
                # create_ivf_index = """
                # CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivf 
                # ON documents USING ivfflat (embedding vector_cosine_ops) 
                # WITH (lists = 100);
                # """
                # cursor.execute(create_ivf_index)
                
                logger.info("✅ Vector indexes created successfully")
                return True
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error creating vector indexes: {e}")
        return False

def get_database_stats() -> Optional[Dict[str, Any]]:
    """
    Get database statistics for monitoring
    
    Returns:
        Dictionary containing database statistics
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get document count
                cursor.execute("SELECT COUNT(*) as total_docs FROM documents;")
                total_docs = cursor.fetchone()['total_docs']
                
                # Get documents with embeddings
                cursor.execute("SELECT COUNT(*) as docs_with_embeddings FROM documents WHERE embedding IS NOT NULL;")
                docs_with_embeddings = cursor.fetchone()['docs_with_embeddings']
                
                # Get recent searches
                cursor.execute("""
                    SELECT COUNT(*) as recent_searches 
                    FROM search_analytics 
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours';
                """)
                recent_searches = cursor.fetchone()['recent_searches']
                
                # Get average search time
                cursor.execute("""
                    SELECT AVG(search_time_ms) as avg_search_time 
                    FROM search_analytics 
                    WHERE timestamp > CURRENT_TIMESTAMP - INTERVAL '24 hours';
                """)
                avg_search_time = cursor.fetchone()['avg_search_time']
                
                return {
                    'total_documents': total_docs,
                    'documents_with_embeddings': docs_with_embeddings,
                    'recent_searches_24h': recent_searches,
                    'avg_search_time_ms': float(avg_search_time) if avg_search_time else 0,
                    'embedding_coverage': round((docs_with_embeddings / total_docs * 100), 2) if total_docs > 0 else 0
                }
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error getting database stats: {e}")
        return None

def initialize_database() -> bool:
    """
    Complete database initialization process
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("🔧 Initializing database for semantic search...")
    
    # Test connection
    if not test_database_connection():
        logger.error("❌ Database connection failed")
        return False
    
    # Setup pgvector extension
    if not setup_pgvector_extension():
        logger.error("❌ Failed to setup pgvector extension")
        return False
    
    # Create schema
    if not create_database_schema():
        logger.error("❌ Failed to create database schema")
        return False
    
    logger.info("✅ Database initialized successfully")
    return True
