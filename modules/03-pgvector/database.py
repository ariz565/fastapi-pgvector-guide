# Database Operations for Semantic Search
# Clean function-based database operations with proper error handling

import logging
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
import numpy as np

from config import get_db_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def insert_document(title: str, content: str, category: Optional[str] = None, 
                   url: Optional[str] = None, embedding: Optional[np.ndarray] = None) -> Optional[int]:
    """
    Insert a single document into the database
    
    Args:
        title: Document title
        content: Document content
        category: Optional category
        url: Optional source URL
        embedding: Optional pre-computed embedding vector
        
    Returns:
        Document ID if successful, None otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Insert document with or without embedding
                if embedding is not None:
                    cursor.execute("""
                        INSERT INTO documents (title, content, category, url, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                    """, (title, content, category, url, embedding.tolist()))
                else:
                    cursor.execute("""
                        INSERT INTO documents (title, content, category, url)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id
                    """, (title, content, category, url))
                
                doc_id = cursor.fetchone()['id']
                logger.info(f"✅ Inserted document {doc_id}: '{title[:50]}...'")
                return doc_id
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error inserting document: {e}")
        return None

def insert_documents_batch(documents: List[Dict[str, Any]]) -> List[int]:
    """
    Insert multiple documents in a single transaction
    
    Args:
        documents: List of document dictionaries with keys: title, content, category, url, embedding
        
    Returns:
        List of inserted document IDs
    """
    inserted_ids = []
    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Use executemany for efficient batch insertion
                insert_query = """
                    INSERT INTO documents (title, content, category, url, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """
                
                # Prepare data for batch insertion
                batch_data = []
                for doc in documents:
                    embedding_list = doc.get('embedding').tolist() if doc.get('embedding') is not None else None
                    batch_data.append((
                        doc['title'],
                        doc['content'],
                        doc.get('category'),
                        doc.get('url'),
                        embedding_list
                    ))
                
                # Execute batch insertion
                for data in batch_data:
                    cursor.execute(insert_query, data)
                    doc_id = cursor.fetchone()['id']
                    inserted_ids.append(doc_id)
                
                logger.info(f"✅ Batch inserted {len(inserted_ids)} documents")
                return inserted_ids
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error in batch insert: {e}")
        return []

def update_document_embedding(document_id: int, embedding: np.ndarray) -> bool:
    """
    Update the embedding for a specific document
    
    Args:
        document_id: ID of the document to update
        embedding: New embedding vector
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    UPDATE documents 
                    SET embedding = %s, updated_at = CURRENT_TIMESTAMP
                    WHERE id = %s
                """, (embedding.tolist(), document_id))
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Updated embedding for document {document_id}")
                    return True
                else:
                    logger.warning(f"⚠️ No document found with ID {document_id}")
                    return False
                    
    except psycopg2.Error as e:
        logger.error(f"❌ Error updating embedding: {e}")
        return False

def get_document_by_id(document_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a document by its ID
    
    Args:
        document_id: ID of the document to retrieve
        
    Returns:
        Document dictionary or None if not found
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, content, category, url, created_at, updated_at
                    FROM documents 
                    WHERE id = %s
                """, (document_id,))
                
                result = cursor.fetchone()
                if result:
                    return dict(result)
                else:
                    logger.warning(f"⚠️ Document {document_id} not found")
                    return None
                    
    except psycopg2.Error as e:
        logger.error(f"❌ Error retrieving document: {e}")
        return None

def get_documents_without_embeddings(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get documents that don't have embeddings yet
    
    Args:
        limit: Maximum number of documents to retrieve
        
    Returns:
        List of document dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, content, category, url
                    FROM documents 
                    WHERE embedding IS NULL
                    ORDER BY created_at ASC
                    LIMIT %s
                """, (limit,))
                
                results = cursor.fetchall()
                logger.info(f"Found {len(results)} documents without embeddings")
                return [dict(row) for row in results]
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error retrieving documents without embeddings: {e}")
        return []

def search_documents_by_vector(query_embedding: np.ndarray, limit: int = 10, 
                              category_filter: Optional[str] = None, 
                              similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
    """
    Search documents using vector similarity
    
    Args:
        query_embedding: Query vector for similarity search
        limit: Maximum number of results
        category_filter: Optional category filter
        similarity_threshold: Minimum similarity score
        
    Returns:
        List of search result dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Build query with optional filters
                base_query = """
                SELECT 
                    id, title, content, category, url, created_at,
                    1 - (embedding <=> %s) as similarity_score
                FROM documents 
                WHERE embedding IS NOT NULL
                """
                
                params = [query_embedding.tolist()]
                
                # Add category filter if specified
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
                
                cursor.execute(base_query, params)
                results = cursor.fetchall()
                
                # Convert to list of dictionaries
                formatted_results = []
                for row in results:
                    result = dict(row)
                    result['similarity_score'] = float(result['similarity_score'])
                    if result['created_at']:
                        result['created_at'] = result['created_at'].isoformat()
                    formatted_results.append(result)
                
                return formatted_results
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error in vector search: {e}")
        return []

def search_documents_hybrid(query: str, query_embedding: np.ndarray, 
                           text_weight: float = 0.3, vector_weight: float = 0.7,
                           limit: int = 10, category_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining text and vector similarity
    
    Args:
        query: Text query for full-text search
        query_embedding: Query vector for similarity search
        text_weight: Weight for text search component
        vector_weight: Weight for vector search component
        limit: Maximum number of results
        category_filter: Optional category filter
        
    Returns:
        List of hybrid search result dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Hybrid search query combining text and vector search
                hybrid_query = """
                SELECT 
                    id, title, content, category, url, created_at,
                    ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', %s)) as text_score,
                    1 - (embedding <=> %s) as vector_score,
                    (%s * ts_rank(to_tsvector('english', title || ' ' || content), plainto_tsquery('english', %s))) +
                    (%s * (1 - (embedding <=> %s))) as combined_score
                FROM documents 
                WHERE embedding IS NOT NULL
                  AND to_tsvector('english', title || ' ' || content) @@ plainto_tsquery('english', %s)
                """
                
                params = [query, query_embedding.tolist(), text_weight, query, 
                         vector_weight, query_embedding.tolist(), query]
                
                # Add category filter if specified
                if category_filter:
                    hybrid_query += " AND category = %s"
                    params.append(category_filter)
                
                # Order by combined score
                hybrid_query += " ORDER BY combined_score DESC LIMIT %s"
                params.append(limit)
                
                cursor.execute(hybrid_query, params)
                results = cursor.fetchall()
                
                # Convert to list of dictionaries
                formatted_results = []
                for row in results:
                    result = dict(row)
                    result['text_score'] = float(result['text_score'])
                    result['vector_score'] = float(result['vector_score'])
                    result['combined_score'] = float(result['combined_score'])
                    if result['created_at']:
                        result['created_at'] = result['created_at'].isoformat()
                    formatted_results.append(result)
                
                return formatted_results
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error in hybrid search: {e}")
        return []

def find_similar_documents_by_id(document_id: int, limit: int = 10, 
                                exclude_same_category: bool = False) -> Tuple[Optional[Dict], List[Dict[str, Any]]]:
    """
    Find documents similar to a given document by ID
    
    Args:
        document_id: Reference document ID
        limit: Maximum number of similar documents
        exclude_same_category: Whether to exclude same category documents
        
    Returns:
        Tuple of (reference_document, similar_documents)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get the reference document
                cursor.execute("""
                    SELECT id, title, embedding, category 
                    FROM documents 
                    WHERE id = %s AND embedding IS NOT NULL
                """, (document_id,))
                
                ref_doc = cursor.fetchone()
                if not ref_doc:
                    return None, []
                
                ref_doc_dict = dict(ref_doc)
                
                # Find similar documents
                similarity_query = """
                SELECT 
                    id, title, content, category, url, created_at,
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
                similar_docs = cursor.fetchall()
                
                # Format similar documents
                formatted_similar = []
                for row in similar_docs:
                    result = dict(row)
                    result['similarity_score'] = float(result['similarity_score'])
                    if result['created_at']:
                        result['created_at'] = result['created_at'].isoformat()
                    formatted_similar.append(result)
                
                return ref_doc_dict, formatted_similar
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error finding similar documents: {e}")
        return None, []

def get_documents_by_category(category: str, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Get all documents in a specific category
    
    Args:
        category: Category name to filter by
        limit: Maximum number of documents to return
        
    Returns:
        List of document dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT id, title, content, category, url, created_at
                    FROM documents 
                    WHERE category = %s
                    ORDER BY created_at DESC
                    LIMIT %s
                """, (category, limit))
                
                results = cursor.fetchall()
                
                # Format results
                formatted_results = []
                for row in results:
                    result = dict(row)
                    if result['created_at']:
                        result['created_at'] = result['created_at'].isoformat()
                    formatted_results.append(result)
                
                return formatted_results
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error getting documents by category: {e}")
        return []

def get_all_categories() -> List[Dict[str, Any]]:
    """
    Get all categories with document counts
    
    Returns:
        List of category dictionaries with counts
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        category,
                        COUNT(*) as document_count,
                        COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedded_count
                    FROM documents 
                    WHERE category IS NOT NULL
                    GROUP BY category
                    ORDER BY document_count DESC
                """)
                
                results = cursor.fetchall()
                return [dict(row) for row in results]
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error getting categories: {e}")
        return []

def log_search_query(query: str, search_type: str = "semantic", 
                    results_count: int = 0, search_time_ms: float = 0.0) -> bool:
    """
    Log a search query for analytics
    
    Args:
        query: Search query text
        search_type: Type of search performed
        results_count: Number of results returned
        search_time_ms: Search execution time in milliseconds
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO search_analytics (query, search_type, results_count, search_time_ms)
                    VALUES (%s, %s, %s, %s)
                """, (query, search_type, results_count, search_time_ms))
                
                return True
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error logging search: {e}")
        return False

def get_search_analytics(days: int = 7) -> Dict[str, Any]:
    """
    Get search analytics for the specified period
    
    Args:
        days: Number of days to analyze
        
    Returns:
        Dictionary containing analytics data
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get overall statistics
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_searches,
                        AVG(search_time_ms) as avg_search_time,
                        AVG(results_count) as avg_results_count
                    FROM search_analytics 
                    WHERE timestamp >= NOW() - INTERVAL '%s days'
                """, (days,))
                
                stats = cursor.fetchone()
                
                # Get top queries
                cursor.execute("""
                    SELECT query, COUNT(*) as frequency
                    FROM search_analytics 
                    WHERE timestamp >= NOW() - INTERVAL '%s days'
                    GROUP BY query
                    ORDER BY frequency DESC
                    LIMIT 10
                """, (days,))
                
                top_queries = cursor.fetchall()
                
                return {
                    'total_searches': stats['total_searches'] if stats else 0,
                    'avg_search_time_ms': float(stats['avg_search_time']) if stats['avg_search_time'] else 0,
                    'avg_results_count': float(stats['avg_results_count']) if stats['avg_results_count'] else 0,
                    'top_queries': [dict(q) for q in top_queries],
                    'period_days': days
                }
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error getting search analytics: {e}")
        return {}

def delete_document(document_id: int) -> bool:
    """
    Delete a document by ID
    
    Args:
        document_id: ID of document to delete
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM documents WHERE id = %s", (document_id,))
                
                if cursor.rowcount > 0:
                    logger.info(f"✅ Deleted document {document_id}")
                    return True
                else:
                    logger.warning(f"⚠️ Document {document_id} not found for deletion")
                    return False
                    
    except psycopg2.Error as e:
        logger.error(f"❌ Error deleting document: {e}")
        return False

def get_document_count() -> int:
    """
    Get total number of documents in the database
    
    Returns:
        Total document count
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT COUNT(*) as count FROM documents")
                result = cursor.fetchone()
                return result['count'] if result else 0
                
    except psycopg2.Error as e:
        logger.error(f"❌ Error getting document count: {e}")
        return 0

def get_embedding_coverage() -> Tuple[int, int, float]:
    """
    Get embedding coverage statistics
    
    Returns:
        Tuple of (total_docs, docs_with_embeddings, coverage_percentage)
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_docs,
                        COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as docs_with_embeddings
                    FROM documents
                """)
                
                result = cursor.fetchone()
                if result:
                    total = result['total_docs']
                    with_embeddings = result['docs_with_embeddings']
                    coverage = (with_embeddings / total * 100) if total > 0 else 0
                    return total, with_embeddings, coverage
                else:
                    return 0, 0, 0.0
                    
    except psycopg2.Error as e:
        logger.error(f"❌ Error getting embedding coverage: {e}")
        return 0, 0, 0.0
