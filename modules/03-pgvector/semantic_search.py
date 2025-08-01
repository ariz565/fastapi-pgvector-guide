# Semantic Search Engine with PostgreSQL and pgvector
# This module implements high-performance semantic search functionality

import logging
import time
from datetime import datetime
import numpy as np
from typing import List, Dict, Optional, Tuple
import math

# Import our database and indexing modules
from database_config import VectorDatabase, DatabaseConfig
from document_indexer import TextEmbeddingGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticSearchEngine:
    """
    High-performance semantic search engine using PostgreSQL + pgvector
    Supports various similarity metrics and hybrid search capabilities
    """
    
    def __init__(self, db_config=None, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize semantic search engine
        
        Args:
            db_config: Database configuration instance
            embedding_model: Name of embedding model to use
        """
        # Initialize database connection
        self.db = VectorDatabase(db_config)
        
        # Initialize embedding generator (same model as used for indexing)
        self.embedding_generator = TextEmbeddingGenerator(embedding_model)
        
        # Search configuration
        self.default_limit = 10
        self.max_limit = 100
        
        logger.info("SemanticSearchEngine initialized and ready")
    
    def search(self, query, limit=None, category_filter=None, similarity_threshold=0.0):
        """
        Perform semantic search for documents similar to the query
        
        Args:
            query: Search query text
            limit: Maximum number of results to return
            category_filter: Optional category to filter results
            similarity_threshold: Minimum similarity score (0.0 to 1.0)
            
        Returns:
            Dictionary containing search results and metadata
        """
        # Validate and set parameters
        limit = min(limit or self.default_limit, self.max_limit)
        
        logger.info(f"🔍 Searching for: '{query[:50]}{'...' if len(query) > 50 else ''}'")
        
        try:
            # Start timing
            search_start = time.time()
            
            # Generate query embedding
            embedding_start = time.time()
            query_embedding = self.embedding_generator.generate_embedding(query)
            embedding_time = (time.time() - embedding_start) * 1000
            
            # Perform database search
            db_start = time.time()
            results = self._execute_vector_search(
                query_embedding, 
                limit, 
                category_filter, 
                similarity_threshold
            )
            db_time = (time.time() - db_start) * 1000
            
            # Calculate total search time
            total_time = (time.time() - search_start) * 1000
            
            # Log search to database for analytics
            self._log_search(query, query_embedding, len(results), total_time)
            
            # Format response
            response = {
                'query': query,
                'results': results,
                'total_results': len(results),
                'search_time_ms': round(total_time, 2),
                'embedding_time_ms': round(embedding_time, 2),
                'database_time_ms': round(db_time, 2),
                'filters': {
                    'category': category_filter,
                    'similarity_threshold': similarity_threshold
                },
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Found {len(results)} results in {total_time:.2f}ms")
            return response
            
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _execute_vector_search(self, query_embedding, limit, category_filter, similarity_threshold):
        """
        Execute the actual vector similarity search in PostgreSQL
        
        Args:
            query_embedding: Query vector as numpy array
            limit: Maximum results to return
            category_filter: Optional category filter
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of search result dictionaries
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Build SQL query with optional filters
                base_query = """
                SELECT 
                    id,
                    title,
                    content,
                    category,
                    url,
                    created_at,
                    1 - (embedding <=> %s) as similarity_score
                FROM documents 
                WHERE embedding IS NOT NULL
                """
                
                # Add category filter if specified
                params = [query_embedding.tolist()]
                if category_filter:
                    base_query += " AND category = %s"
                    params.append(category_filter)
                
                # Add similarity threshold filter
                if similarity_threshold > 0:
                    base_query += " AND (1 - (embedding <=> %s)) >= %s"
                    params.extend([query_embedding.tolist(), similarity_threshold])
                
                # Order by similarity and limit results
                base_query += " ORDER BY embedding <=> %s LIMIT %s"
                params.extend([query_embedding.tolist(), limit])
                
                # Execute query
                cursor.execute(base_query, params)
                raw_results = cursor.fetchall()
                
                # Format results
                results = []
                for row in raw_results:
                    result = {
                        'id': row['id'],
                        'title': row['title'],
                        'content': row['content'][:500] + ('...' if len(row['content']) > 500 else ''),  # Truncate content
                        'content_preview': self._generate_content_preview(row['content'], 150),
                        'category': row['category'],
                        'url': row['url'],
                        'similarity_score': round(float(row['similarity_score']), 4),
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Error executing vector search: {e}")
            return []
    
    def hybrid_search(self, query, text_weight=0.3, vector_weight=0.7, limit=None, category_filter=None):
        """
        Perform hybrid search combining text search and vector similarity
        
        Args:
            query: Search query text
            text_weight: Weight for text search component (0.0 to 1.0)
            vector_weight: Weight for vector search component (0.0 to 1.0)
            limit: Maximum number of results
            category_filter: Optional category filter
            
        Returns:
            Dictionary containing hybrid search results
        """
        limit = min(limit or self.default_limit, self.max_limit)
        
        logger.info(f"🔍 Hybrid search: '{query[:50]}{'...' if len(query) > 50 else ''}' (text:{text_weight}, vector:{vector_weight})")
        
        try:
            search_start = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_embedding(query)
            
            # Execute hybrid search query
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                hybrid_query = """
                SELECT 
                    id,
                    title,
                    content,
                    category,
                    url,
                    created_at,
                    -- Text search score using PostgreSQL full-text search
                    ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', %s)) as text_score,
                    -- Vector similarity score
                    1 - (embedding <=> %s) as vector_score,
                    -- Combined weighted score
                    (%s * ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', %s))) +
                    (%s * (1 - (embedding <=> %s))) as combined_score
                FROM documents 
                WHERE embedding IS NOT NULL
                  AND to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', %s)
                """
                
                params = [query, query_embedding.tolist(), text_weight, query, vector_weight, query_embedding.tolist(), query]
                
                # Add category filter if specified
                if category_filter:
                    hybrid_query += " AND category = %s"
                    params.append(category_filter)
                
                # Order by combined score
                hybrid_query += " ORDER BY combined_score DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(hybrid_query, params)
                raw_results = cursor.fetchall()
                
                # Format results
                results = []
                for row in raw_results:
                    result = {
                        'id': row['id'],
                        'title': row['title'],
                        'content_preview': self._generate_content_preview(row['content'], 150),
                        'category': row['category'],
                        'url': row['url'],
                        'text_score': round(float(row['text_score']), 4),
                        'vector_score': round(float(row['vector_score']), 4),
                        'combined_score': round(float(row['combined_score']), 4),
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None
                    }
                    results.append(result)
                
                total_time = (time.time() - search_start) * 1000
                
                # Log hybrid search
                self._log_search(f"HYBRID: {query}", query_embedding, len(results), total_time)
                
                response = {
                    'query': query,
                    'search_type': 'hybrid',
                    'weights': {'text': text_weight, 'vector': vector_weight},
                    'results': results,
                    'total_results': len(results),
                    'search_time_ms': round(total_time, 2),
                    'timestamp': datetime.now().isoformat()
                }
                
                logger.info(f"✅ Hybrid search found {len(results)} results in {total_time:.2f}ms")
                return response
                
        except Exception as e:
            logger.error(f"Error during hybrid search: {e}")
            return {
                'query': query,
                'search_type': 'hybrid',
                'results': [],
                'total_results': 0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def find_similar_documents(self, document_id, limit=None, exclude_same_category=False):
        """
        Find documents similar to a given document
        
        Args:
            document_id: ID of the reference document
            limit: Maximum number of similar documents to return
            exclude_same_category: Whether to exclude documents from same category
            
        Returns:
            Dictionary containing similar documents
        """
        limit = min(limit or self.default_limit, self.max_limit)
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get the reference document
                cursor.execute("SELECT title, embedding, category FROM documents WHERE id = %s", (document_id,))
                ref_doc = cursor.fetchone()
                
                if not ref_doc:
                    return {'error': f'Document {document_id} not found'}
                
                # Find similar documents
                similarity_query = """
                SELECT 
                    id,
                    title,
                    content,
                    category,
                    url,
                    created_at,
                    1 - (embedding <=> %s) as similarity_score
                FROM documents 
                WHERE id != %s AND embedding IS NOT NULL
                """
                
                params = [ref_doc['embedding'], document_id]
                
                if exclude_same_category and ref_doc['category']:
                    similarity_query += " AND category != %s"
                    params.append(ref_doc['category'])
                
                similarity_query += " ORDER BY embedding <=> %s LIMIT %s"
                params.extend([ref_doc['embedding'], limit])
                
                cursor.execute(similarity_query, params)
                raw_results = cursor.fetchall()
                
                # Format results
                results = []
                for row in raw_results:
                    result = {
                        'id': row['id'],
                        'title': row['title'],
                        'content_preview': self._generate_content_preview(row['content'], 150),
                        'category': row['category'],
                        'url': row['url'],
                        'similarity_score': round(float(row['similarity_score']), 4),
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None
                    }
                    results.append(result)
                
                response = {
                    'reference_document': {
                        'id': document_id,
                        'title': ref_doc['title'],
                        'category': ref_doc['category']
                    },
                    'similar_documents': results,
                    'total_results': len(results),
                    'filters': {
                        'exclude_same_category': exclude_same_category
                    }
                }
                
                logger.info(f"✅ Found {len(results)} similar documents for document {document_id}")
                return response
                
        except Exception as e:
            logger.error(f"Error finding similar documents: {e}")
            return {'error': str(e)}
    
    def search_by_category(self, category, limit=None):
        """
        Get documents from a specific category ordered by recency
        
        Args:
            category: Category name to search
            limit: Maximum number of documents to return
            
        Returns:
            List of documents in the category
        """
        limit = min(limit or self.default_limit, self.max_limit)
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT id, title, content, category, url, created_at
                    FROM documents 
                    WHERE category = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (category, limit))
                
                raw_results = cursor.fetchall()
                
                results = []
                for row in raw_results:
                    result = {
                        'id': row['id'],
                        'title': row['title'],
                        'content_preview': self._generate_content_preview(row['content'], 150),
                        'category': row['category'],
                        'url': row['url'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None
                    }
                    results.append(result)
                
                logger.info(f"✅ Found {len(results)} documents in category '{category}'")
                return results
                
        except Exception as e:
            logger.error(f"Error searching by category: {e}")
            return []
    
    def get_search_analytics(self, days=7):
        """
        Get search analytics for the past N days
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary containing search analytics
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get search statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_searches,
                        AVG(search_time_ms) as avg_search_time,
                        AVG(results_count) as avg_results_count,
                        MAX(search_time_ms) as max_search_time,
                        MIN(search_time_ms) as min_search_time
                    FROM searches 
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                """, (days,))
                
                stats = cursor.fetchone()
                
                # Get top search terms
                cursor.execute("""
                    SELECT query_text, COUNT(*) as frequency
                    FROM searches 
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY query_text
                    ORDER BY frequency DESC
                    LIMIT 10
                """, (days,))
                
                top_queries = cursor.fetchall()
                
                # Get search performance over time (daily)
                cursor.execute("""
                    SELECT 
                        DATE(created_at) as date,
                        COUNT(*) as searches,
                        AVG(search_time_ms) as avg_time
                    FROM searches 
                    WHERE created_at >= NOW() - INTERVAL '%s days'
                    GROUP BY DATE(created_at)
                    ORDER BY date
                """, (days,))
                
                daily_stats = cursor.fetchall()
                
                analytics = {
                    'period_days': days,
                    'summary': dict(stats) if stats else {},
                    'top_queries': [dict(q) for q in top_queries],
                    'daily_performance': [dict(d) for d in daily_stats],
                    'generated_at': datetime.now().isoformat()
                }
                
                return analytics
                
        except Exception as e:
            logger.error(f"Error getting search analytics: {e}")
            return {}
    
    def _generate_content_preview(self, content, max_length=150):
        """
        Generate a preview of document content with intelligent truncation
        
        Args:
            content: Full document content
            max_length: Maximum length of preview
            
        Returns:
            Truncated content with ellipsis if needed
        """
        if not content:
            return ""
        
        if len(content) <= max_length:
            return content
        
        # Find the last complete sentence within the limit
        truncated = content[:max_length]
        last_period = truncated.rfind('.')
        last_exclamation = truncated.rfind('!')
        last_question = truncated.rfind('?')
        
        # Use the last complete sentence if found
        last_sentence_end = max(last_period, last_exclamation, last_question)
        
        if last_sentence_end > max_length * 0.7:  # Only if it's not too short
            return content[:last_sentence_end + 1]
        else:
            # Otherwise, truncate at word boundary
            last_space = truncated.rfind(' ')
            if last_space > max_length * 0.8:
                return content[:last_space] + "..."
            else:
                return content[:max_length] + "..."
    
    def _log_search(self, query, query_embedding, results_count, search_time_ms):
        """
        Log search query for analytics and monitoring
        
        Args:
            query: Search query text
            query_embedding: Query embedding vector
            results_count: Number of results returned
            search_time_ms: Search execution time in milliseconds
        """
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO searches (query_text, query_embedding, results_count, search_time_ms, created_at)
                    VALUES (%s, %s, %s, %s, %s)
                """, (
                    query,
                    query_embedding.tolist(),
                    results_count,
                    search_time_ms,
                    datetime.now()
                ))
                
        except Exception as e:
            # Don't fail the search if logging fails
            logger.warning(f"Failed to log search: {e}")

def interactive_search_demo():
    """
    Interactive demo for testing semantic search functionality
    Allows users to enter queries and see results in real-time
    """
    print("🔍 Interactive Semantic Search Demo")
    print("=" * 40)
    print("Enter search queries to test semantic search.")
    print("Type 'quit' to exit, 'help' for commands.\n")
    
    # Initialize search engine
    search_engine = SemanticSearchEngine()
    
    while True:
        try:
            # Get user input
            query = input("🔍 Search query: ").strip()
            
            if not query:
                continue
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if query.lower() == 'help':
                print("\nAvailable commands:")
                print("  • Enter any text to search")
                print("  • 'hybrid:query' for hybrid search")
                print("  • 'category:tech' to browse by category")
                print("  • 'similar:5' to find documents similar to doc ID 5")
                print("  • 'analytics' to see search statistics")
                print("  • 'quit' to exit")
                print()
                continue
            
            # Handle special commands
            if query.startswith('hybrid:'):
                search_query = query[7:].strip()
                if search_query:
                    print(f"\n🔄 Performing hybrid search for: '{search_query}'")
                    results = search_engine.hybrid_search(search_query)
                    display_search_results(results)
                continue
            
            if query.startswith('category:'):
                category = query[9:].strip()
                if category:
                    print(f"\n📁 Browsing category: '{category}'")
                    results = search_engine.search_by_category(category)
                    if results:
                        for i, doc in enumerate(results[:5], 1):
                            print(f"\n{i}. {doc['title']}")
                            print(f"   {doc['content_preview']}")
                            print(f"   Created: {doc['created_at']}")
                    else:
                        print(f"   No documents found in category '{category}'")
                continue
            
            if query.startswith('similar:'):
                try:
                    doc_id = int(query[8:].strip())
                    print(f"\n🔗 Finding documents similar to document {doc_id}")
                    results = search_engine.find_similar_documents(doc_id)
                    if 'error' not in results:
                        print(f"Reference: {results['reference_document']['title']}")
                        for i, doc in enumerate(results['similar_documents'][:5], 1):
                            print(f"\n{i}. {doc['title']} (similarity: {doc['similarity_score']})")
                            print(f"   {doc['content_preview']}")
                    else:
                        print(f"   Error: {results['error']}")
                except ValueError:
                    print("   Error: Please provide a valid document ID")
                continue
            
            if query.lower() == 'analytics':
                print("\n📊 Search Analytics (last 7 days)")
                analytics = search_engine.get_search_analytics()
                if analytics.get('summary'):
                    summary = analytics['summary']
                    print(f"   Total searches: {summary.get('total_searches', 0)}")
                    print(f"   Average search time: {summary.get('avg_search_time', 0):.2f}ms")
                    print(f"   Average results: {summary.get('avg_results_count', 0):.1f}")
                else:
                    print("   No search data available")
                continue
            
            # Regular semantic search
            print(f"\n🔍 Searching for: '{query}'")
            results = search_engine.search(query)
            display_search_results(results)
            
        except KeyboardInterrupt:
            print("\n\n👋 Search interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")

def display_search_results(results):
    """
    Display search results in a formatted way
    
    Args:
        results: Search results dictionary
    """
    if 'error' in results:
        print(f"❌ Search error: {results['error']}")
        return
    
    if results['total_results'] == 0:
        print("🤷 No results found. Try a different query.")
        return
    
    print(f"\n✅ Found {results['total_results']} results in {results['search_time_ms']}ms")
    print("-" * 60)
    
    for i, result in enumerate(results['results'][:5], 1):
        print(f"\n{i}. {result['title']}")
        
        # Show similarity or combined score
        if 'similarity_score' in result:
            print(f"   Similarity: {result['similarity_score']}")
        elif 'combined_score' in result:
            print(f"   Score: {result['combined_score']} (text: {result['text_score']}, vector: {result['vector_score']})")
        
        print(f"   Category: {result.get('category', 'N/A')}")
        print(f"   Preview: {result['content_preview']}")
        
        if result.get('url'):
            print(f"   URL: {result['url']}")

def main():
    """
    Main function to demonstrate semantic search capabilities
    """
    print("🚀 Semantic Search Engine with PostgreSQL + pgvector")
    print("=" * 55)
    
    # Initialize search engine
    print("1. Initializing semantic search engine...")
    search_engine = SemanticSearchEngine()
    
    # Test basic search
    print("\n2. Testing basic semantic search...")
    test_queries = [
        "machine learning and artificial intelligence",
        "healthy eating and nutrition",
        "programming best practices"
    ]
    
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = search_engine.search(query, limit=3)
        
        if results['total_results'] > 0:
            print(f"   Found {results['total_results']} results in {results['search_time_ms']}ms")
            for i, result in enumerate(results['results'], 1):
                print(f"     {i}. {result['title']} (score: {result['similarity_score']})")
        else:
            print("   No results found")
    
    # Test hybrid search
    print("\n3. Testing hybrid search...")
    hybrid_results = search_engine.hybrid_search("python programming", text_weight=0.4, vector_weight=0.6, limit=3)
    
    if hybrid_results['total_results'] > 0:
        print(f"   Hybrid search found {hybrid_results['total_results']} results")
        for result in hybrid_results['results']:
            print(f"     • {result['title']} (combined: {result['combined_score']})")
    
    # Show analytics
    print("\n4. Search analytics:")
    analytics = search_engine.get_search_analytics()
    if analytics.get('summary'):
        summary = analytics['summary']
        print(f"   Total searches: {summary.get('total_searches', 0)}")
        print(f"   Average search time: {summary.get('avg_search_time', 0):.2f}ms")
    
    print("\n🎉 Semantic search testing completed!")
    print("\n🔄 Run interactive_search_demo() for hands-on testing")

if __name__ == "__main__":
    # Run main demo
    main()
    
    # Ask if user wants interactive demo
    print("\n" + "="*50)
    response = input("Would you like to try the interactive search demo? (y/n): ").strip().lower()
    if response in ['y', 'yes']:
        interactive_search_demo()
