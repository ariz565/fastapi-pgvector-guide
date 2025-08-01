# Setup Script for PostgreSQL + pgvector Semantic Search
# This script helps you set up and run the complete semantic search system

import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """
    Check if Python version is compatible
    """
    if sys.version_info < (3, 8):
        logger.error("❌ Python 3.8 or higher is required")
        logger.info("💡 Please upgrade your Python installation")
        return False
    
    logger.info(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_requirements():
    """
    Install required Python packages
    """
    logger.info("📦 Installing Python packages...")
    
    try:
        # Install packages from requirements.txt
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--upgrade"
        ])
        
        logger.info("✅ All packages installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install packages: {e}")
        logger.info("💡 Try running: pip install -r requirements.txt")
        return False

def check_postgresql_connection():
    """
    Check if PostgreSQL is running and accessible
    """
    logger.info("🔍 Checking PostgreSQL connection...")
    
    try:
        # Import our database module
        from database_config import VectorDatabase
        
        # Test connection
        db = VectorDatabase()
        if db.test_connection():
            logger.info("✅ PostgreSQL connection successful")
            return True
        else:
            logger.error("❌ PostgreSQL connection failed")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Failed to import database module: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        logger.info("💡 Make sure PostgreSQL 17 is running on localhost:5432")
        logger.info("💡 Check your database credentials in the code")
        return False

def setup_database():
    """
    Setup database tables and pgvector extension
    """
    logger.info("🏗️ Setting up database...")
    
    try:
        from database_config import VectorDatabase
        
        db = VectorDatabase()
        
        # Setup pgvector extension
        if not db.setup_pgvector_extension():
            logger.error("❌ Failed to setup pgvector extension")
            logger.info("💡 Make sure you have superuser privileges")
            logger.info("💡 Run: CREATE EXTENSION vector; in your PostgreSQL database")
            return False
        
        # Create tables
        if not db.create_semantic_search_tables():
            logger.error("❌ Failed to create database tables")
            return False
        
        logger.info("✅ Database setup completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database setup failed: {e}")
        return False

def load_sample_data():
    """
    Load sample documents for testing
    """
    logger.info("📚 Loading sample documents...")
    
    try:
        from document_indexer import DocumentIndexer, create_sample_documents
        
        # Initialize indexer
        indexer = DocumentIndexer()
        
        # Check if we already have documents
        stats = indexer.db.get_database_stats()
        if stats and stats['documents']['total_documents'] > 0:
            logger.info(f"📄 Found {stats['documents']['total_documents']} existing documents")
            user_input = input("Do you want to add more sample documents? (y/n): ").strip().lower()
            if user_input not in ['y', 'yes']:
                logger.info("⏭️ Skipping sample data loading")
                return True
        
        # Create and load sample documents
        sample_docs = create_sample_documents()
        doc_ids = indexer.add_documents_batch(sample_docs)
        
        if doc_ids:
            logger.info(f"✅ Loaded {len(doc_ids)} sample documents")
            return True
        else:
            logger.error("❌ Failed to load sample documents")
            return False
            
    except Exception as e:
        logger.error(f"❌ Sample data loading failed: {e}")
        return False

def test_search_functionality():
    """
    Test semantic search functionality
    """
    logger.info("🔍 Testing search functionality...")
    
    try:
        from semantic_search import SemanticSearchEngine
        
        # Initialize search engine
        search_engine = SemanticSearchEngine()
        
        # Test search queries
        test_queries = [
            "machine learning",
            "healthy food",
            "programming"
        ]
        
        for query in test_queries:
            results = search_engine.search(query, limit=3)
            
            if results['total_results'] > 0:
                logger.info(f"  ✅ '{query}' → {results['total_results']} results in {results['search_time_ms']:.1f}ms")
            else:
                logger.warning(f"  ⚠️ '{query}' → No results found")
        
        logger.info("✅ Search functionality test completed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Search test failed: {e}")
        return False

def create_env_file():
    """
    Create a .env file with database configuration
    """
    env_file = Path('.env')
    
    if env_file.exists():
        logger.info("📄 .env file already exists")
        return True
    
    logger.info("📝 Creating .env configuration file...")
    
    env_content = """# PostgreSQL Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=password

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# Embedding Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Search Configuration
DEFAULT_SEARCH_LIMIT=10
MAX_SEARCH_LIMIT=100
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        logger.info("✅ .env file created")
        logger.info("💡 Edit .env file to customize your database connection")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create .env file: {e}")
        return False

def run_setup():
    """
    Run the complete setup process
    """
    print("🚀 PostgreSQL + pgvector Semantic Search Setup")
    print("=" * 50)
    print("This script will set up your semantic search system")
    print()
    
    # Check prerequisites
    if not check_python_version():
        return False
    
    # Create environment file
    create_env_file()
    
    # Install packages
    if not install_requirements():
        return False
    
    # Check database connection
    if not check_postgresql_connection():
        print("\n💡 Setup Instructions:")
        print("1. Make sure PostgreSQL 17 is installed and running")
        print("2. Create a database named 'vectordb'")
        print("3. Install pgvector extension: https://github.com/pgvector/pgvector#installation")
        print("4. Update database credentials in the code if needed")
        return False
    
    # Setup database
    if not setup_database():
        return False
    
    # Load sample data
    if not load_sample_data():
        return False
    
    # Test functionality
    if not test_search_functionality():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print()
    print("🔜 Next steps:")
    print("1. Run search tests:")
    print("   python semantic_search.py")
    print()
    print("2. Start the FastAPI server:")
    print("   python semantic_search_api.py")
    print()
    print("3. Open your browser:")
    print("   http://localhost:8000/docs")
    print()
    print("4. Try the interactive demo:")
    print("   python -c \"from semantic_search import interactive_search_demo; interactive_search_demo()\"")
    
    return True

def quick_test():
    """
    Quick test to verify everything is working
    """
    print("🔬 Quick Functionality Test")
    print("=" * 30)
    
    try:
        # Test imports
        logger.info("Testing imports...")
        from database_config import VectorDatabase
        from document_indexer import DocumentIndexer
        from semantic_search import SemanticSearchEngine
        logger.info("✅ All imports successful")
        
        # Test database
        logger.info("Testing database connection...")
        db = VectorDatabase()
        if not db.test_connection():
            logger.error("❌ Database connection failed")
            return False
        logger.info("✅ Database connection successful")
        
        # Test search
        logger.info("Testing search...")
        search_engine = SemanticSearchEngine()
        results = search_engine.search("test query", limit=1)
        logger.info(f"✅ Search test successful ({results['total_results']} results)")
        
        print("\n🎉 All tests passed! System is ready to use.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """
    Main function - setup wizard
    """
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick test mode
        return quick_test()
    
    # Full setup mode
    try:
        success = run_setup()
        if success:
            print("\n🚀 Your semantic search system is ready!")
        else:
            print("\n❌ Setup failed. Please check the errors above.")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Setup interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during setup: {e}")
        return False

if __name__ == "__main__":
    main()
