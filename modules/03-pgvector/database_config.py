# PostgreSQL Database Configuration and Connection
# This module handles database connections, table creation, and pgvector setup

import os
import logging
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.extensions import register_adapter, AsIs
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """
    Database configuration class for PostgreSQL with pgvector
    Handles connection parameters and environment variables
    """
    
    def __init__(self):
        # Database connection parameters
        # Using environment variables for security (can be overridden)
        self.host = os.getenv('DB_HOST', 'localhost')
        self.port = os.getenv('DB_PORT', '5432')
        self.database = os.getenv('DB_NAME', 'vectordb')
        self.user = os.getenv('DB_USER', 'postgres')
        self.password = os.getenv('DB_PASSWORD', 'password')
        
        # Connection pool settings
        self.min_connections = 1
        self.max_connections = 10
        
        logger.info(f"Database config initialized for {self.host}:{self.port}/{self.database}")
    
    def get_connection_string(self):
        """
        Generate PostgreSQL connection string
        Returns formatted connection string for psycopg2
        """
        return f"host={self.host} port={self.port} dbname={self.database} user={self.user} password={self.password}"

class VectorDatabase:
    """
    Main database class for vector operations with PostgreSQL + pgvector
    Handles connections, table creation, and vector operations
    """
    
    def __init__(self, config=None):
        # Use provided config or create default
        self.config = config or DatabaseConfig()
        self.connection_string = self.config.get_connection_string()
        
        # Register numpy array adapter for psycopg2
        # This allows numpy arrays to be stored as PostgreSQL arrays
        register_adapter(np.ndarray, self._adapt_numpy_array)
        
        logger.info("VectorDatabase initialized")
    
    def _adapt_numpy_array(self, numpy_array):
        """
        Adapter function to convert numpy arrays to PostgreSQL array format
        This is required for storing vectors in the database
        """
        return AsIs(','.join([str(x) for x in numpy_array]))
    
    @contextmanager
    def get_connection(self):
        """
        Context manager for database connections
        Ensures proper connection cleanup and error handling
        """
        connection = None
        try:
            # Create connection with dictionary cursor for easier data access
            connection = psycopg2.connect(
                self.connection_string,
                cursor_factory=RealDictCursor
            )
            yield connection
            connection.commit()
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if connection:
                connection.close()
    
    def test_connection(self):
        """
        Test database connection and pgvector extension
        Returns True if everything is working correctly
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Test basic connection
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                logger.info(f"PostgreSQL version: {version['version']}")
                
                # Test pgvector extension
                cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                vector_ext = cursor.fetchone()
                
                if vector_ext:
                    logger.info("✅ pgvector extension is installed and active")
                    return True
                else:
                    logger.warning("⚠️ pgvector extension not found")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Connection test failed: {e}")
            return False
    
    def setup_pgvector_extension(self):
        """
        Install and enable pgvector extension
        Must be run with superuser privileges
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create extension if it doesn't exist
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                logger.info("✅ pgvector extension created/verified")
                
                # Verify installation
                cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
                result = cursor.fetchone()
                
                if result:
                    logger.info(f"pgvector version: {result.get('extversion', 'unknown')}")
                    return True
                else:
                    logger.error("Failed to create pgvector extension")
                    return False
                    
        except Exception as e:
            logger.error(f"Error setting up pgvector: {e}")
            logger.info("💡 Tip: Make sure you have superuser privileges and pgvector is installed")
            return False
    
    def create_semantic_search_tables(self):
        """
        Create tables for semantic search application
        Includes documents table with vector embeddings
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create documents table with vector column
                documents_table = """
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title VARCHAR(500) NOT NULL,
                    content TEXT NOT NULL,
                    category VARCHAR(100),
                    url VARCHAR(1000),
                    embedding VECTOR(384),  -- 384 dimensions for sentence-transformers
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(documents_table)
                
                # Create searches table to track user queries
                searches_table = """
                CREATE TABLE IF NOT EXISTS searches (
                    id SERIAL PRIMARY KEY,
                    query_text TEXT NOT NULL,
                    query_embedding VECTOR(384),
                    results_count INTEGER DEFAULT 0,
                    search_time_ms FLOAT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
                cursor.execute(searches_table)
                
                # Create indexes for better performance
                # Index on category for filtering
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);")
                
                # Index on created_at for time-based queries
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);")
                
                # Text search index on title and content
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_documents_text_search ON documents USING gin(to_tsvector('english', title || ' ' || content));")
                
                logger.info("✅ Semantic search tables created successfully")
                
                # Show table information
                cursor.execute("""
                    SELECT table_name, column_name, data_type 
                    FROM information_schema.columns 
                    WHERE table_name IN ('documents', 'searches')
                    ORDER BY table_name, ordinal_position;
                """)
                
                columns = cursor.fetchall()
                logger.info("📋 Created tables and columns:")
                current_table = None
                for col in columns:
                    if col['table_name'] != current_table:
                        current_table = col['table_name']
                        logger.info(f"  {current_table}:")
                    logger.info(f"    {col['column_name']}: {col['data_type']}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            return False
    
    def create_vector_indexes(self):
        """
        Create vector similarity indexes for fast search
        Uses both IVFFlat and HNSW indexes for different use cases
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                logger.info("🔧 Creating vector indexes (this may take a while)...")
                
                # Check if we have enough data for indexing
                cursor.execute("SELECT COUNT(*) as count FROM documents WHERE embedding IS NOT NULL;")
                doc_count = cursor.fetchone()['count']
                
                if doc_count < 1000:
                    logger.info(f"📊 Only {doc_count} documents with embeddings. Skipping indexing for now.")
                    logger.info("💡 Tip: Vector indexes are most beneficial with 1000+ documents")
                    return True
                
                # Create IVFFlat index (good for large datasets)
                # Number of lists should be approximately sqrt(number of rows)
                lists = max(10, int(doc_count ** 0.5))
                
                ivfflat_index = f"""
                CREATE INDEX IF NOT EXISTS idx_documents_embedding_ivfflat 
                ON documents USING ivfflat (embedding vector_cosine_ops) 
                WITH (lists = {lists});
                """
                cursor.execute(ivfflat_index)
                logger.info(f"✅ IVFFlat index created with {lists} lists")
                
                # Create HNSW index (better for real-time queries)
                # Only create if PostgreSQL version supports it
                try:
                    hnsw_index = """
                    CREATE INDEX IF NOT EXISTS idx_documents_embedding_hnsw 
                    ON documents USING hnsw (embedding vector_cosine_ops) 
                    WITH (m = 16, ef_construction = 64);
                    """
                    cursor.execute(hnsw_index)
                    logger.info("✅ HNSW index created (m=16, ef_construction=64)")
                except Exception as hnsw_error:
                    logger.warning(f"HNSW index creation failed: {hnsw_error}")
                    logger.info("💡 HNSW may not be available in your pgvector version")
                
                # Create index on searches table
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_searches_embedding ON searches USING ivfflat (query_embedding vector_cosine_ops);")
                
                logger.info("🎉 Vector indexing completed!")
                return True
                
        except Exception as e:
            logger.error(f"Error creating vector indexes: {e}")
            return False
    
    def get_database_stats(self):
        """
        Get statistics about the database and tables
        Useful for monitoring and performance analysis
        """
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                stats = {}
                
                # Document count and statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_documents,
                        COUNT(embedding) as documents_with_embeddings,
                        COUNT(DISTINCT category) as unique_categories
                    FROM documents;
                """)
                doc_stats = cursor.fetchone()
                stats['documents'] = dict(doc_stats)
                
                # Search statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_searches,
                        AVG(search_time_ms) as avg_search_time_ms,
                        AVG(results_count) as avg_results_count
                    FROM searches;
                """)
                search_stats = cursor.fetchone()
                stats['searches'] = dict(search_stats)
                
                # Category breakdown
                cursor.execute("""
                    SELECT category, COUNT(*) as count 
                    FROM documents 
                    WHERE category IS NOT NULL
                    GROUP BY category 
                    ORDER BY count DESC;
                """)
                categories = cursor.fetchall()
                stats['categories'] = [dict(cat) for cat in categories]
                
                # Index information
                cursor.execute("""
                    SELECT indexname, tablename 
                    FROM pg_indexes 
                    WHERE tablename IN ('documents', 'searches')
                    ORDER BY tablename, indexname;
                """)
                indexes = cursor.fetchall()
                stats['indexes'] = [dict(idx) for idx in indexes]
                
                return stats
                
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return None
    
    def cleanup_database(self, confirm=False):
        """
        Clean up all semantic search tables and data
        Use with caution - this will delete all data!
        """
        if not confirm:
            logger.warning("⚠️ cleanup_database called without confirmation")
            logger.info("To actually clean up, call with confirm=True")
            return False
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                # Drop tables in correct order (handle foreign keys if any)
                tables = ['searches', 'documents']
                
                for table in tables:
                    cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                    logger.info(f"🗑️ Dropped table: {table}")
                
                logger.info("✅ Database cleanup completed")
                return True
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False

def main():
    """
    Main function to test database setup and configuration
    Run this to verify your PostgreSQL + pgvector installation
    """
    print("🔍 Testing PostgreSQL + pgvector Setup")
    print("=" * 45)
    
    # Initialize database
    db = VectorDatabase()
    
    # Test connection
    print("\n1. Testing database connection...")
    if not db.test_connection():
        print("❌ Database connection failed!")
        print("💡 Check your PostgreSQL installation and credentials")
        return False
    
    # Setup pgvector extension
    print("\n2. Setting up pgvector extension...")
    if not db.setup_pgvector_extension():
        print("❌ pgvector setup failed!")
        print("💡 Make sure pgvector is installed and you have superuser privileges")
        return False
    
    # Create tables
    print("\n3. Creating semantic search tables...")
    if not db.create_semantic_search_tables():
        print("❌ Table creation failed!")
        return False
    
    # Show database stats
    print("\n4. Database statistics:")
    stats = db.get_database_stats()
    if stats:
        print(f"   📄 Documents: {stats['documents']['total_documents']}")
        print(f"   🔍 Searches: {stats['searches']['total_searches']}")
        print(f"   📑 Categories: {stats['documents']['unique_categories']}")
        print(f"   🗂️ Indexes: {len(stats['indexes'])}")
    
    print("\n✅ Database setup completed successfully!")
    print("\n🚀 Next steps:")
    print("   1. Run document_indexer.py to add sample documents")
    print("   2. Run semantic_search.py to test search functionality")
    
    return True

if __name__ == "__main__":
    main()
