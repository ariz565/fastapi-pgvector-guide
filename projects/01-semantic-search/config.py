# Configuration file for Semantic Document Search Project
# This file contains all the settings and configurations

import os

# Database Configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'semantic_search',
    'username': 'postgres',
    'password': 'postgres',  # Change this to your actual password
}

# Text Embedding Model Configuration
# We use sentence-transformers for converting text to vectors
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Small, fast model good for beginners
EMBEDDING_SIZE = 384  # Size of vectors this model produces

# File Upload Configuration
UPLOAD_FOLDER = 'data/uploads'
ALLOWED_EXTENSIONS = ['txt', 'pdf', 'docx']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB max file size

# Search Configuration
SEARCH_RESULTS_LIMIT = 10  # Number of results to return
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score to include in results

# API Configuration
API_HOST = '127.0.0.1'
API_PORT = 8000

# Create upload directory if it doesn't exist
def create_directories():
    """Creates necessary directories for the application"""
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
