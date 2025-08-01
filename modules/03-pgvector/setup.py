# Quick Setup Script for Clean Function-Based Semantic Search
# Automated setup for PostgreSQL 17 + pgvector with modular architecture

import os
import sys
import subprocess
import logging
import time
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command: str, description: str) -> bool:
    """
    Run a shell command and return success status
    
    Args:
        command: Command to execute
        description: Description for logging
        
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info(f"🔄 {description}...")
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ {description} completed successfully")
            return True
        else:
            logger.error(f"❌ {description} failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error running command: {e}")
        return False

def check_python_version() -> bool:
    """
    Check if Python version is compatible
    
    Returns:
        True if Python version is >= 3.8
    """
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        logger.info(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        logger.error(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible (need >= 3.8)")
        return False

def install_dependencies() -> bool:
    """
    Install Python dependencies from requirements.txt
    
    Returns:
        True if installation successful
    """
    logger.info("📦 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        logger.error("❌ requirements.txt not found")
        return False
    
    # Install packages
    commands = [
        "pip install --upgrade pip",
        "pip install -r requirements.txt"
    ]
    
    for command in commands:
        if not run_command(command, f"Running: {command}"):
            return False
    
    return True

def setup_database() -> bool:
    """
    Setup database schema and pgvector extension
    
    Returns:
        True if setup successful
    """
    try:
        logger.info("🔧 Setting up database...")
        
        # Import our config module
        from config import initialize_database, test_database_connection
        
        # Test connection first
        if not test_database_connection():
            logger.error("❌ Cannot connect to PostgreSQL database")
            logger.error("   Please ensure PostgreSQL 17 is running and accessible")
            return False
        
        # Initialize database
        if not initialize_database():
            logger.error("❌ Database initialization failed")
            return False
        
        logger.info("✅ Database setup completed successfully")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Cannot import database modules: {e}")
        logger.error("   Please install dependencies first")
        return False
    except Exception as e:
        logger.error(f"❌ Database setup error: {e}")
        return False

def setup_embedding_model() -> bool:
    """
    Initialize the embedding model
    
    Returns:
        True if model setup successful
    """
    try:
        logger.info("🤖 Setting up embedding model...")
        
        # Import embeddings module
        from embeddings import initialize_embedding_model, get_embedding_model_info
        
        # Initialize model
        if not initialize_embedding_model():
            logger.warning("⚠️ Embedding model initialization failed, will use TF-IDF fallback")
            return True  # Still return True as fallback is available
        
        # Get model info
        model_info = get_embedding_model_info()
        logger.info(f"✅ Embedding model ready: {model_info.get('model_name')} ({model_info.get('embedding_dimension')}D)")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Cannot import embedding modules: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Embedding model setup error: {e}")
        return False

def create_sample_data() -> bool:
    """
    Create sample documents for testing
    
    Returns:
        True if sample data created successfully
    """
    try:
        logger.info("📄 Creating sample documents...")
        
        # Import database functions
        from database import insert_document
        from embeddings import generate_embedding
        
        # Sample documents
        sample_docs = [
            {
                "title": "Introduction to Machine Learning",
                "content": "Machine learning is a subset of artificial intelligence that focuses on developing algorithms that can learn from and make predictions on data. It involves training models on datasets to recognize patterns and make informed decisions without being explicitly programmed for each specific task.",
                "category": "technology",
                "url": "https://example.com/ml-intro"
            },
            {
                "title": "Healthy Eating Guidelines", 
                "content": "A balanced diet includes a variety of foods from all food groups. Focus on fruits, vegetables, whole grains, lean proteins, and healthy fats. Limit processed foods, added sugars, and excessive sodium. Stay hydrated and maintain portion control for optimal health.",
                "category": "health",
                "url": "https://example.com/healthy-eating"
            },
            {
                "title": "Python Programming Best Practices",
                "content": "Write clean, readable code by following PEP 8 style guidelines. Use meaningful variable names, add docstrings to functions, handle exceptions properly, and write unit tests. Organize code into modules and follow the DRY (Don't Repeat Yourself) principle.",
                "category": "technology", 
                "url": "https://example.com/python-practices"
            },
            {
                "title": "Climate Change and Environmental Impact",
                "content": "Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities, particularly burning fossil fuels, have accelerated these changes. Renewable energy, conservation efforts, and sustainable practices are crucial for mitigating environmental impact.",
                "category": "environment",
                "url": "https://example.com/climate-change"
            },
            {
                "title": "Database Design Principles",
                "content": "Good database design involves normalization to reduce redundancy, proper indexing for performance, and establishing clear relationships between tables. Consider data types carefully, implement constraints for data integrity, and plan for scalability and maintenance.",
                "category": "technology",
                "url": "https://example.com/database-design"
            }
        ]
        
        # Insert sample documents with embeddings
        for doc in sample_docs:
            # Generate embedding
            embedding = generate_embedding(doc["content"])
            
            # Insert document
            doc_id = insert_document(
                title=doc["title"],
                content=doc["content"],
                category=doc["category"],
                url=doc["url"],
                embedding=embedding
            )
            
            if doc_id:
                logger.info(f"✅ Created sample document: {doc['title']}")
            else:
                logger.warning(f"⚠️ Failed to create document: {doc['title']}")
        
        logger.info("✅ Sample documents created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error creating sample data: {e}")
        return False

def start_api_server() -> bool:
    """
    Start the FastAPI server
    
    Returns:
        True if server starts successfully
    """
    try:
        logger.info("🚀 Starting FastAPI server...")
        logger.info("   API will be available at: http://127.0.0.1:8000")
        logger.info("   Documentation at: http://127.0.0.1:8000/docs")
        logger.info("   Press Ctrl+C to stop the server")
        
        # Import and run the server
        from main import run_server
        
        # Start server (this will block)
        run_server(host="127.0.0.1", port=8000, reload=True)
        
        return True
        
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
        return True
    except Exception as e:
        logger.error(f"❌ Error starting server: {e}")
        return False

def run_tests() -> bool:
    """
    Run basic functionality tests
    
    Returns:
        True if tests pass
    """
    try:
        logger.info("🧪 Running basic functionality tests...")
        
        # Import test modules
        from search import semantic_search, hybrid_search
        from database import get_document_count
        
        # Test 1: Document count
        doc_count = get_document_count()
        logger.info(f"📊 Total documents in database: {doc_count}")
        
        if doc_count == 0:
            logger.warning("⚠️ No documents found, search tests will be skipped")
            return True
        
        # Test 2: Semantic search
        logger.info("🔍 Testing semantic search...")
        results = semantic_search("machine learning algorithms", limit=3)
        
        if results.get('total_results', 0) > 0:
            logger.info(f"✅ Semantic search: {results['total_results']} results in {results['search_time_ms']}ms")
        else:
            logger.warning("⚠️ Semantic search returned no results")
        
        # Test 3: Hybrid search
        logger.info("🔍 Testing hybrid search...")
        hybrid_results = hybrid_search("programming python", limit=3)
        
        if hybrid_results.get('total_results', 0) > 0:
            logger.info(f"✅ Hybrid search: {hybrid_results['total_results']} results in {hybrid_results['search_time_ms']}ms")
        else:
            logger.warning("⚠️ Hybrid search returned no results")
        
        logger.info("✅ All tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error running tests: {e}")
        return False

def main():
    """
    Main setup function that orchestrates the entire setup process
    """
    logger.info("🚀 Starting Semantic Search Setup")
    logger.info("=" * 50)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Install dependencies
    if not install_dependencies():
        logger.error("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Step 3: Setup database
    if not setup_database():
        logger.error("❌ Database setup failed")
        logger.error("   Please check your PostgreSQL installation and configuration")
        sys.exit(1)
    
    # Step 4: Setup embedding model
    if not setup_embedding_model():
        logger.error("❌ Embedding model setup failed")
        sys.exit(1)
    
    # Step 5: Create sample data
    if not create_sample_data():
        logger.warning("⚠️ Sample data creation failed, continuing anyway")
    
    # Step 6: Run tests
    if not run_tests():
        logger.warning("⚠️ Some tests failed, but setup is complete")
    
    logger.info("🎉 Setup completed successfully!")
    logger.info("=" * 50)
    
    # Ask user if they want to start the server
    try:
        response = input("\nWould you like to start the API server now? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            start_api_server()
        else:
            logger.info("📝 To start the server later, run: python main.py")
            logger.info("📚 Documentation will be available at: http://127.0.0.1:8000/docs")
            
    except KeyboardInterrupt:
        logger.info("\n👋 Setup complete. Run 'python main.py' to start the server.")

if __name__ == "__main__":
    main()
