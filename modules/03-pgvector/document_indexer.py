# Document Indexer and Embedding Generator
# This module handles text embedding generation and document storage in PostgreSQL

import logging
import time
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional, Union
import re
import json

# Import sentence transformers for generating embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# Import our database configuration
from database_config import VectorDatabase, DatabaseConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextEmbeddingGenerator:
    """
    Handles text embedding generation using various models
    Supports both sentence-transformers and fallback methods
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize embedding generator with specified model
        
        Args:
            model_name: Name of the sentence-transformer model to use
                       'all-MiniLM-L6-v2' is fast and produces 384-dim embeddings
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dimension = 384  # Default for MiniLM model
        
        # Try to load sentence-transformers model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                logger.info(f"Loading sentence-transformer model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.embedding_dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"✅ Model loaded successfully. Embedding dimension: {self.embedding_dimension}")
            except Exception as e:
                logger.error(f"Failed to load sentence-transformer model: {e}")
                self.model = None
        
        # If sentence-transformers failed, use fallback
        if self.model is None:
            logger.warning("Using fallback TF-IDF embedding method")
            self._setup_tfidf_fallback()
    
    def _setup_tfidf_fallback(self):
        """
        Setup TF-IDF based fallback embedding method
        Used when sentence-transformers is not available
        """
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            
            # Create TF-IDF vectorizer with reasonable parameters
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # SVD to reduce dimensionality to 384 (matching sentence-transformers)
            self.svd = TruncatedSVD(n_components=384, random_state=42)
            
            self.embedding_dimension = 384
            self.use_tfidf = True
            logger.info("✅ TF-IDF fallback method initialized")
            
        except ImportError:
            logger.error("Neither sentence-transformers nor scikit-learn available!")
            logger.error("Please install one of them: pip install sentence-transformers OR pip install scikit-learn")
            raise
    
    def generate_embedding(self, text):
        """
        Generate embedding vector for given text
        
        Args:
            text: Input text string
            
        Returns:
            numpy array representing the text embedding
        """
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(text)
        
        if self.model is not None:
            # Use sentence-transformers model
            try:
                embedding = self.model.encode(cleaned_text, convert_to_numpy=True)
                return embedding.astype(np.float32)
            except Exception as e:
                logger.error(f"Error generating embedding with sentence-transformers: {e}")
                return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        elif hasattr(self, 'use_tfidf'):
            # Use TF-IDF fallback (requires fitting on corpus first)
            logger.warning("TF-IDF fallback requires corpus fitting - returning zero vector")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
        
        else:
            # No method available
            logger.error("No embedding method available")
            return np.zeros(self.embedding_dimension, dtype=np.float32)
    
    def generate_batch_embeddings(self, texts):
        """
        Generate embeddings for multiple texts efficiently
        
        Args:
            texts: List of text strings
            
        Returns:
            List of numpy arrays representing embeddings
        """
        if not texts:
            return []
        
        # Clean all texts
        cleaned_texts = [self._preprocess_text(text) for text in texts]
        
        if self.model is not None:
            try:
                # Batch processing with sentence-transformers is much faster
                embeddings = self.model.encode(cleaned_texts, convert_to_numpy=True, show_progress_bar=True)
                return [emb.astype(np.float32) for emb in embeddings]
            except Exception as e:
                logger.error(f"Error in batch embedding generation: {e}")
                # Fallback to individual processing
                return [self.generate_embedding(text) for text in texts]
        else:
            # Process individually if no batch method available
            return [self.generate_embedding(text) for text in texts]
    
    def _preprocess_text(self, text):
        """
        Clean and preprocess text for embedding generation
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not text:
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-]', ' ', text)
        
        # Strip and ensure not empty
        text = text.strip()
        
        return text

class DocumentIndexer:
    """
    Handles document indexing, embedding generation, and storage in PostgreSQL
    Provides methods for adding, updating, and managing documents
    """
    
    def __init__(self, db_config=None, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize document indexer
        
        Args:
            db_config: Database configuration instance
            embedding_model: Name of embedding model to use
        """
        # Initialize database connection
        self.db = VectorDatabase(db_config)
        
        # Initialize embedding generator
        self.embedding_generator = TextEmbeddingGenerator(embedding_model)
        
        logger.info(f"DocumentIndexer initialized with {embedding_model}")
    
    def add_document(self, title, content, category=None, url=None):
        """
        Add a single document to the database with embedding
        
        Args:
            title: Document title
            content: Document content/text
            category: Optional category classification
            url: Optional source URL
            
        Returns:
            Document ID if successful, None if failed
        """
        try:
            # Combine title and content for embedding
            full_text = f"{title}. {content}"
            
            # Generate embedding
            start_time = time.time()
            embedding = self.embedding_generator.generate_embedding(full_text)
            embedding_time = time.time() - start_time
            
            # Store in database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                insert_query = """
                INSERT INTO documents (title, content, category, url, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                
                cursor.execute(insert_query, (
                    title,
                    content,
                    category,
                    url,
                    embedding.tolist(),  # Convert numpy array to list for PostgreSQL
                    datetime.now()
                ))
                
                doc_id = cursor.fetchone()['id']
                
                logger.info(f"✅ Document added (ID: {doc_id}) in {embedding_time:.3f}s")
                return doc_id
                
        except Exception as e:
            logger.error(f"Error adding document: {e}")
            return None
    
    def add_documents_batch(self, documents):
        """
        Add multiple documents efficiently using batch processing
        
        Args:
            documents: List of dictionaries with keys: title, content, category, url
            
        Returns:
            List of document IDs that were successfully added
        """
        if not documents:
            return []
        
        logger.info(f"📝 Processing batch of {len(documents)} documents...")
        
        try:
            # Extract texts for batch embedding generation
            texts = []
            for doc in documents:
                title = doc.get('title', '')
                content = doc.get('content', '')
                full_text = f"{title}. {content}"
                texts.append(full_text)
            
            # Generate embeddings in batch (much faster)
            start_time = time.time()
            embeddings = self.embedding_generator.generate_batch_embeddings(texts)
            embedding_time = time.time() - start_time
            
            logger.info(f"⚡ Generated {len(embeddings)} embeddings in {embedding_time:.2f}s")
            
            # Insert into database
            doc_ids = []
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                insert_query = """
                INSERT INTO documents (title, content, category, url, embedding, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                
                for doc, embedding in zip(documents, embeddings):
                    try:
                        cursor.execute(insert_query, (
                            doc.get('title', ''),
                            doc.get('content', ''),
                            doc.get('category'),
                            doc.get('url'),
                            embedding.tolist(),
                            datetime.now()
                        ))
                        
                        doc_id = cursor.fetchone()['id']
                        doc_ids.append(doc_id)
                        
                    except Exception as e:
                        logger.error(f"Error inserting document '{doc.get('title', 'Unknown')}': {e}")
                        continue
                
                logger.info(f"✅ Successfully added {len(doc_ids)} documents to database")
                return doc_ids
                
        except Exception as e:
            logger.error(f"Error in batch document processing: {e}")
            return []
    
    def update_document_embedding(self, doc_id):
        """
        Regenerate and update embedding for existing document
        
        Args:
            doc_id: Document ID to update
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get document content
                cursor.execute("SELECT title, content FROM documents WHERE id = %s;", (doc_id,))
                doc = cursor.fetchone()
                
                if not doc:
                    logger.error(f"Document with ID {doc_id} not found")
                    return False
                
                # Generate new embedding
                full_text = f"{doc['title']}. {doc['content']}"
                embedding = self.embedding_generator.generate_embedding(full_text)
                
                # Update database
                cursor.execute(
                    "UPDATE documents SET embedding = %s, updated_at = %s WHERE id = %s;",
                    (embedding.tolist(), datetime.now(), doc_id)
                )
                
                logger.info(f"✅ Updated embedding for document {doc_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating document embedding: {e}")
            return False
    
    def reindex_all_documents(self):
        """
        Regenerate embeddings for all documents in the database
        Useful when switching embedding models or fixing corrupted embeddings
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get all documents without embeddings or needing updates
                cursor.execute("""
                    SELECT id, title, content 
                    FROM documents 
                    WHERE embedding IS NULL OR updated_at < created_at
                    ORDER BY id;
                """)
                
                documents = cursor.fetchall()
                
                if not documents:
                    logger.info("📄 No documents need reindexing")
                    return True
                
                logger.info(f"🔄 Reindexing {len(documents)} documents...")
                
                # Process in batches for efficiency
                batch_size = 50
                success_count = 0
                
                for i in range(0, len(documents), batch_size):
                    batch = documents[i:i + batch_size]
                    
                    # Generate embeddings for batch
                    texts = [f"{doc['title']}. {doc['content']}" for doc in batch]
                    embeddings = self.embedding_generator.generate_batch_embeddings(texts)
                    
                    # Update database
                    for doc, embedding in zip(batch, embeddings):
                        try:
                            cursor.execute(
                                "UPDATE documents SET embedding = %s, updated_at = %s WHERE id = %s;",
                                (embedding.tolist(), datetime.now(), doc['id'])
                            )
                            success_count += 1
                        except Exception as e:
                            logger.error(f"Error updating document {doc['id']}: {e}")
                    
                    # Commit batch
                    conn.commit()
                    logger.info(f"   Processed batch {i//batch_size + 1}/{(len(documents) + batch_size - 1)//batch_size}")
                
                logger.info(f"✅ Reindexing completed. Updated {success_count}/{len(documents)} documents")
                return True
                
        except Exception as e:
            logger.error(f"Error during reindexing: {e}")
            return False
    
    def get_documents_by_category(self, category, limit=100):
        """
        Retrieve documents by category
        
        Args:
            category: Category name to filter by
            limit: Maximum number of documents to return
            
        Returns:
            List of document dictionaries
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, title, content, category, url, created_at
                    FROM documents 
                    WHERE category = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                """, (category, limit))
                
                documents = cursor.fetchall()
                return [dict(doc) for doc in documents]
                
        except Exception as e:
            logger.error(f"Error retrieving documents by category: {e}")
            return []
    
    def delete_document(self, doc_id):
        """
        Delete a document from the database
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM documents WHERE id = %s;", (doc_id,))
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Document {doc_id} deleted successfully")
                    return True
                else:
                    logger.warning(f"Document {doc_id} not found")
                    return False
                    
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False

def create_sample_documents():
    """
    Create a collection of sample documents for testing
    Returns list of document dictionaries ready for indexing
    """
    sample_docs = [
        {
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a method of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention. Machine learning algorithms build mathematical models based on training data in order to make predictions or decisions without being explicitly programmed to do so.",
            "category": "technology",
            "url": "https://example.com/ml-intro"
        },
        {
            "title": "The Benefits of Regular Exercise",
            "content": "Regular physical activity is one of the most important things you can do for your health. Being physically active can improve your brain health, help manage weight, reduce the risk of disease, strengthen bones and muscles, and improve your ability to do everyday activities. Adults who sit less and do any amount of moderate-to-vigorous physical activity gain some health benefits.",
            "category": "health",
            "url": "https://example.com/exercise-benefits"
        },
        {
            "title": "Sustainable Cooking with Local Ingredients",
            "content": "Sustainable cooking focuses on using locally sourced, seasonal ingredients to create delicious meals while minimizing environmental impact. By choosing ingredients that are grown nearby, you reduce transportation emissions and support local farmers. Seasonal cooking also ensures that you're eating foods at their peak freshness and nutritional value.",
            "category": "food",
            "url": "https://example.com/sustainable-cooking"
        },
        {
            "title": "The History of Jazz Music",
            "content": "Jazz is a music genre that originated in the African-American communities of New Orleans in the late 19th and early 20th centuries. It developed from roots in blues and ragtime and features swing and blue notes, complex chords, call-and-response vocals, polyrhythms and improvisation. Jazz has influenced many other musical styles and has been called America's classical music.",
            "category": "music",
            "url": "https://example.com/jazz-history"
        },
        {
            "title": "Climate Change and Global Warming",
            "content": "Climate change refers to long-term shifts in global or regional climate patterns. Since the mid-20th century, the pace of climate change has been largely attributed to increased levels of atmospheric carbon dioxide and other greenhouse gases produced by human activities. The consequences include rising sea levels, extreme weather events, and shifts in precipitation patterns.",
            "category": "environment",
            "url": "https://example.com/climate-change"
        },
        {
            "title": "Python Programming Best Practices",
            "content": "Writing clean, maintainable Python code requires following established best practices. Use meaningful variable names, write docstrings for functions and classes, follow PEP 8 style guidelines, handle exceptions gracefully, and write unit tests. Code should be readable and self-documenting. Use virtual environments to manage dependencies and keep your codebase organized with proper module structure.",
            "category": "technology",
            "url": "https://example.com/python-best-practices"
        },
        {
            "title": "Meditation and Mindfulness Techniques",
            "content": "Meditation is a practice where an individual uses techniques such as mindfulness, or focusing the mind on a particular object, thought, or activity to train attention and awareness. Regular meditation can reduce stress, improve concentration, increase self-awareness, and promote emotional health. Start with just 5-10 minutes daily and gradually increase the duration.",
            "category": "health",
            "url": "https://example.com/meditation-guide"
        },
        {
            "title": "The Art of French Pastry Making",
            "content": "French pastry making is a culinary art that requires precision, patience, and practice. From delicate croissants to elaborate croquembouches, French pastries are known for their intricate techniques and refined flavors. Key skills include working with laminated doughs, tempering chocolate, making proper custards, and understanding the science behind baking temperatures and timing.",
            "category": "food",
            "url": "https://example.com/french-pastry"
        },
        {
            "title": "Modern Photography Techniques",
            "content": "Digital photography has revolutionized how we capture and process images. Understanding composition rules like the rule of thirds, leading lines, and framing can dramatically improve your photos. Master your camera's manual settings including aperture, shutter speed, and ISO. Post-processing software like Lightroom and Photoshop allows for creative enhancement while maintaining image quality.",
            "category": "arts",
            "url": "https://example.com/photography-techniques"
        },
        {
            "title": "Renewable Energy Solutions",
            "content": "Renewable energy comes from natural sources that are constantly replenished, such as sunlight, wind, rain, tides, waves, and geothermal heat. These energy sources are sustainable alternatives to fossil fuels and can help reduce greenhouse gas emissions. Solar panels, wind turbines, and hydroelectric systems are becoming increasingly efficient and cost-effective.",
            "category": "environment",
            "url": "https://example.com/renewable-energy"
        }
    ]
    
    return sample_docs

def main():
    """
    Main function to demonstrate document indexing functionality
    """
    print("📚 Document Indexing and Embedding Generation")
    print("=" * 50)
    
    # Initialize indexer
    print("1. Initializing document indexer...")
    indexer = DocumentIndexer()
    
    # Create sample documents
    print("\n2. Creating sample documents...")
    sample_docs = create_sample_documents()
    print(f"   Created {len(sample_docs)} sample documents")
    
    # Add documents to database
    print("\n3. Adding documents to database...")
    doc_ids = indexer.add_documents_batch(sample_docs)
    print(f"   Successfully indexed {len(doc_ids)} documents")
    
    # Show statistics
    print("\n4. Database statistics:")
    stats = indexer.db.get_database_stats()
    if stats:
        print(f"   📄 Total documents: {stats['documents']['total_documents']}")
        print(f"   🔢 With embeddings: {stats['documents']['documents_with_embeddings']}")
        print(f"   📑 Categories: {stats['documents']['unique_categories']}")
        
        # Show category breakdown
        if stats['categories']:
            print("\n   Category breakdown:")
            for cat in stats['categories']:
                print(f"     {cat['category']}: {cat['count']} documents")
    
    # Test individual document addition
    print("\n5. Testing individual document addition...")
    new_doc_id = indexer.add_document(
        title="Test Document",
        content="This is a test document to verify individual indexing works correctly.",
        category="test"
    )
    
    if new_doc_id:
        print(f"   ✅ Added test document with ID: {new_doc_id}")
    
    print("\n🎉 Document indexing completed successfully!")
    print("\n🔍 Next step: Run semantic_search.py to test search functionality")

if __name__ == "__main__":
    main()
