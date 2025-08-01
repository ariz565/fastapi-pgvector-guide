# Database connection and management module
# This module handles connecting to PostgreSQL with pgvector extension

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from config import DATABASE_CONFIG

class DatabaseManager:
    """Manages database connections and operations for semantic search"""
    
    def __init__(self):
        """Initialize database manager"""
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            # Create connection string
            connection_string = f"postgresql://{DATABASE_CONFIG['username']}:{DATABASE_CONFIG['password']}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"
            
            # Connect to database
            self.connection = psycopg2.connect(
                host=DATABASE_CONFIG['host'],
                port=DATABASE_CONFIG['port'],
                database=DATABASE_CONFIG['database'],
                user=DATABASE_CONFIG['username'],
                password=DATABASE_CONFIG['password'],
                cursor_factory=RealDictCursor
            )
            
            self.cursor = self.connection.cursor()
            print("Successfully connected to PostgreSQL database")
            return True
            
        except Exception as e:
            print(f"Error connecting to database: {e}")
            return False
    
    def setup_database(self):
        """Create database tables and enable pgvector extension"""
        try:
            # Enable pgvector extension
            self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # Create documents table to store document metadata
            create_documents_table = """
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                title VARCHAR(255) NOT NULL,
                content TEXT NOT NULL,
                file_path VARCHAR(500),
                file_type VARCHAR(50),
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                word_count INTEGER,
                char_count INTEGER
            );
            """
            
            # Create embeddings table to store vector embeddings
            create_embeddings_table = """
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            
            # Execute table creation
            self.cursor.execute(create_documents_table)
            self.cursor.execute(create_embeddings_table)
            
            # Create index for faster vector similarity search
            create_index = """
            CREATE INDEX IF NOT EXISTS embeddings_vector_idx 
            ON embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
            self.cursor.execute(create_index)
            
            # Commit changes
            self.connection.commit()
            print("Database tables and indexes created successfully")
            return True
            
        except Exception as e:
            print(f"Error setting up database: {e}")
            self.connection.rollback()
            return False
    
    def insert_document(self, title, content, file_path, file_type):
        """Insert a new document into the database"""
        try:
            # Calculate document statistics
            word_count = len(content.split())
            char_count = len(content)
            
            # Insert document
            insert_query = """
            INSERT INTO documents (title, content, file_path, file_type, word_count, char_count)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id;
            """
            
            self.cursor.execute(insert_query, (title, content, file_path, file_type, word_count, char_count))
            document_id = self.cursor.fetchone()['id']
            self.connection.commit()
            
            print(f"Document '{title}' inserted with ID: {document_id}")
            return document_id
            
        except Exception as e:
            print(f"Error inserting document: {e}")
            self.connection.rollback()
            return None
    
    def insert_embedding(self, document_id, chunk_text, chunk_index, embedding):
        """Insert an embedding vector for a document chunk"""
        try:
            # Convert numpy array to list for PostgreSQL
            embedding_list = embedding.tolist()
            
            insert_query = """
            INSERT INTO embeddings (document_id, chunk_text, chunk_index, embedding)
            VALUES (%s, %s, %s, %s);
            """
            
            self.cursor.execute(insert_query, (document_id, chunk_text, chunk_index, embedding_list))
            self.connection.commit()
            return True
            
        except Exception as e:
            print(f"Error inserting embedding: {e}")
            self.connection.rollback()
            return False
    
    def search_similar_embeddings(self, query_embedding, limit=10):
        """Search for similar embeddings using cosine similarity"""
        try:
            # Convert numpy array to list for PostgreSQL
            query_embedding_list = query_embedding.tolist()
            
            # Search query using cosine similarity
            search_query = """
            SELECT 
                e.chunk_text,
                d.title,
                d.file_path,
                d.file_type,
                e.embedding <=> %s as distance,
                1 - (e.embedding <=> %s) as similarity_score
            FROM embeddings e
            JOIN documents d ON e.document_id = d.id
            ORDER BY e.embedding <=> %s
            LIMIT %s;
            """
            
            self.cursor.execute(search_query, (query_embedding_list, query_embedding_list, query_embedding_list, limit))
            results = self.cursor.fetchall()
            
            return results
            
        except Exception as e:
            print(f"Error searching embeddings: {e}")
            return []
    
    def get_all_documents(self):
        """Retrieve all documents from database"""
        try:
            query = "SELECT * FROM documents ORDER BY upload_date DESC;"
            self.cursor.execute(query)
            return self.cursor.fetchall()
            
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    def delete_document(self, document_id):
        """Delete a document and its embeddings"""
        try:
            # Delete document (embeddings will be deleted automatically due to CASCADE)
            delete_query = "DELETE FROM documents WHERE id = %s;"
            self.cursor.execute(delete_query, (document_id,))
            self.connection.commit()
            
            print(f"Document with ID {document_id} deleted successfully")
            return True
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            self.connection.rollback()
            return False
    
    def close(self):
        """Close database connection"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("Database connection closed")

# Helper function to create database manager
def get_database_manager():
    """Create and return a database manager instance"""
    db_manager = DatabaseManager()
    if db_manager.connect():
        return db_manager
    else:
        return None
