# Search Functions for Semantic Search
# Clean function-based approach for semantic and hybrid search operations

import logging
import time
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime

from embeddings import generate_embedding, generate_embeddings_batch
from database import (
    search_documents_by_vector,
    search_documents_hybrid,
    find_similar_documents_by_id,
    get_documents_by_category,
    log_search_query
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def semantic_search(query: str, limit: int = 10, category_filter: Optional[str] = None, 
                   similarity_threshold: float = 0.0) -> Dict[str, Any]:
    """
    Perform semantic search using vector similarity
    
    Args:
        query: Search query text
        limit: Maximum number of results
        category_filter: Optional category filter
        similarity_threshold: Minimum similarity score (0.0 to 1.0)
        
    Returns:
        Dictionary containing search results and metadata
    """
    search_start = time.time()
    
    logger.info(f"🔍 Semantic search: '{query[:50]}{'...' if len(query) > 50 else ''}'")
    
    try:
        # Generate query embedding
        embedding_start = time.time()
        query_embedding = generate_embedding(query)
        embedding_time = (time.time() - embedding_start) * 1000
        
        if query_embedding is None:
            return {
                'query': query,
                'results': [],
                'total_results': 0,
                'error': 'Failed to generate query embedding',
                'timestamp': datetime.now().isoformat()
            }
        
        # Search documents using vector similarity
        db_start = time.time()
        raw_results = search_documents_by_vector(
            query_embedding, limit, category_filter, similarity_threshold
        )
        db_time = (time.time() - db_start) * 1000
        
        # Format results with content preview
        results = []
        for doc in raw_results:
            result = {
                'id': doc['id'],
                'title': doc['title'],
                'content_preview': generate_content_preview(doc['content'], 150),
                'category': doc['category'],
                'url': doc['url'],
                'similarity_score': round(doc['similarity_score'], 4),
                'created_at': doc['created_at']
            }
            results.append(result)
        
        # Calculate total search time
        total_time = (time.time() - search_start) * 1000
        
        # Log search for analytics
        log_search_query(query, "semantic", len(results), total_time)
        
        # Format response
        response = {
            'query': query,
            'search_type': 'semantic',
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
        logger.error(f"❌ Error in semantic search: {e}")
        return {
            'query': query,
            'search_type': 'semantic',
            'results': [],
            'total_results': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def hybrid_search(query: str, text_weight: float = 0.3, vector_weight: float = 0.7, 
                 limit: int = 10, category_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Perform hybrid search combining text and vector similarity
    
    Args:
        query: Search query text
        text_weight: Weight for text search component (0.0 to 1.0)
        vector_weight: Weight for vector search component (0.0 to 1.0)
        limit: Maximum number of results
        category_filter: Optional category filter
        
    Returns:
        Dictionary containing hybrid search results
    """
    search_start = time.time()
    
    logger.info(f"🔍 Hybrid search: '{query[:50]}{'...' if len(query) > 50 else ''}' (text:{text_weight}, vector:{vector_weight})")
    
    try:
        # Validate weights
        if abs((text_weight + vector_weight) - 1.0) > 0.01:
            return {
                'query': query,
                'search_type': 'hybrid',
                'results': [],
                'total_results': 0,
                'error': 'text_weight + vector_weight must equal 1.0',
                'timestamp': datetime.now().isoformat()
            }
        
        # Generate query embedding
        embedding_start = time.time()
        query_embedding = generate_embedding(query)
        embedding_time = (time.time() - embedding_start) * 1000
        
        if query_embedding is None:
            return {
                'query': query,
                'search_type': 'hybrid',
                'results': [],
                'total_results': 0,
                'error': 'Failed to generate query embedding',
                'timestamp': datetime.now().isoformat()
            }
        
        # Perform hybrid search
        db_start = time.time()
        raw_results = search_documents_hybrid(
            query, query_embedding, text_weight, vector_weight, limit, category_filter
        )
        db_time = (time.time() - db_start) * 1000
        
        # Format results with content preview
        results = []
        for doc in raw_results:
            result = {
                'id': doc['id'],
                'title': doc['title'],
                'content_preview': generate_content_preview(doc['content'], 150),
                'category': doc['category'],
                'url': doc['url'],
                'text_score': round(doc['text_score'], 4),
                'vector_score': round(doc['vector_score'], 4),
                'combined_score': round(doc['combined_score'], 4),
                'created_at': doc['created_at']
            }
            results.append(result)
        
        # Calculate total search time
        total_time = (time.time() - search_start) * 1000
        
        # Log search for analytics
        log_search_query(f"HYBRID: {query}", "hybrid", len(results), total_time)
        
        # Format response
        response = {
            'query': query,
            'search_type': 'hybrid',
            'weights': {'text': text_weight, 'vector': vector_weight},
            'results': results,
            'total_results': len(results),
            'search_time_ms': round(total_time, 2),
            'embedding_time_ms': round(embedding_time, 2),
            'database_time_ms': round(db_time, 2),
            'filters': {
                'category': category_filter
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Hybrid search found {len(results)} results in {total_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in hybrid search: {e}")
        return {
            'query': query,
            'search_type': 'hybrid',
            'results': [],
            'total_results': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def find_similar_documents(document_id: int, limit: int = 10, 
                          exclude_same_category: bool = False) -> Dict[str, Any]:
    """
    Find documents similar to a given document
    
    Args:
        document_id: ID of the reference document
        limit: Maximum number of similar documents to return
        exclude_same_category: Whether to exclude documents from same category
        
    Returns:
        Dictionary containing similar documents
    """
    try:
        # Find similar documents using database function
        ref_doc, similar_docs = find_similar_documents_by_id(
            document_id, limit, exclude_same_category
        )
        
        if ref_doc is None:
            return {
                'error': f'Document {document_id} not found or has no embedding',
                'reference_document_id': document_id
            }
        
        # Format similar documents with content preview
        formatted_similar = []
        for doc in similar_docs:
            result = {
                'id': doc['id'],
                'title': doc['title'],
                'content_preview': generate_content_preview(doc['content'], 150),
                'category': doc['category'],
                'url': doc['url'],
                'similarity_score': round(doc['similarity_score'], 4),
                'created_at': doc['created_at']
            }
            formatted_similar.append(result)
        
        response = {
            'reference_document': {
                'id': ref_doc['id'],
                'title': ref_doc['title'],
                'category': ref_doc['category']
            },
            'similar_documents': formatted_similar,
            'total_results': len(formatted_similar),
            'filters': {
                'exclude_same_category': exclude_same_category
            }
        }
        
        logger.info(f"✅ Found {len(formatted_similar)} similar documents for document {document_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error finding similar documents: {e}")
        return {
            'error': str(e),
            'reference_document_id': document_id
        }

def search_by_category(category: str, limit: int = 50) -> Dict[str, Any]:
    """
    Search for documents in a specific category
    
    Args:
        category: Category name to search
        limit: Maximum number of documents to return
        
    Returns:
        Dictionary containing category search results
    """
    try:
        # Get documents from the specified category
        raw_results = get_documents_by_category(category, limit)
        
        # Format results with content preview
        results = []
        for doc in raw_results:
            result = {
                'id': doc['id'],
                'title': doc['title'],
                'content_preview': generate_content_preview(doc['content'], 150),
                'category': doc['category'],
                'url': doc['url'],
                'created_at': doc['created_at']
            }
            results.append(result)
        
        response = {
            'category': category,
            'documents': results,
            'total_results': len(results)
        }
        
        logger.info(f"✅ Found {len(results)} documents in category '{category}'")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error searching by category: {e}")
        return {
            'category': category,
            'documents': [],
            'total_results': 0,
            'error': str(e)
        }

def multi_query_search(queries: List[str], limit: int = 10, 
                      strategy: str = "average") -> Dict[str, Any]:
    """
    Perform search with multiple queries using different combination strategies
    
    Args:
        queries: List of query strings
        limit: Maximum number of results
        strategy: How to combine results ("average", "union", "intersection")
        
    Returns:
        Dictionary containing multi-query search results
    """
    search_start = time.time()
    
    logger.info(f"🔍 Multi-query search: {len(queries)} queries, strategy: {strategy}")
    
    try:
        if not queries:
            return {
                'queries': queries,
                'strategy': strategy,
                'results': [],
                'total_results': 0,
                'error': 'No queries provided',
                'timestamp': datetime.now().isoformat()
            }
        
        if strategy == "average":
            # Generate embeddings for all queries
            embeddings = generate_embeddings_batch(queries)
            valid_embeddings = [emb for emb in embeddings if emb is not None]
            
            if not valid_embeddings:
                return {
                    'queries': queries,
                    'strategy': strategy,
                    'results': [],
                    'total_results': 0,
                    'error': 'Failed to generate embeddings for any query',
                    'timestamp': datetime.now().isoformat()
                }
            
            # Average the embeddings
            avg_embedding = np.mean(valid_embeddings, axis=0)
            
            # Search using averaged embedding
            raw_results = search_documents_by_vector(avg_embedding, limit)
            
        elif strategy == "union":
            # Perform separate searches and combine results
            all_results = []
            seen_docs = set()
            
            for query in queries:
                query_results = semantic_search(query, limit)
                for result in query_results.get('results', []):
                    if result['id'] not in seen_docs:
                        seen_docs.add(result['id'])
                        all_results.append(result)
            
            # Sort by best similarity score and limit
            all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            raw_results = all_results[:limit]
            
        elif strategy == "intersection":
            # Find documents that appear in results for all queries
            query_results = []
            for query in queries:
                results = semantic_search(query, limit * 2)  # Get more results for intersection
                query_results.append({doc['id']: doc for doc in results.get('results', [])})
            
            if not query_results:
                raw_results = []
            else:
                # Find intersection of document IDs
                common_ids = set(query_results[0].keys())
                for result_dict in query_results[1:]:
                    common_ids &= set(result_dict.keys())
                
                # Get documents and calculate average scores
                raw_results = []
                for doc_id in common_ids:
                    docs = [result_dict[doc_id] for result_dict in query_results]
                    avg_score = sum(doc.get('similarity_score', 0) for doc in docs) / len(docs)
                    
                    # Use the document data from the first query but with averaged score
                    result = docs[0].copy()
                    result['similarity_score'] = avg_score
                    raw_results.append(result)
                
                # Sort by averaged score and limit
                raw_results.sort(key=lambda x: x['similarity_score'], reverse=True)
                raw_results = raw_results[:limit]
        
        else:
            return {
                'queries': queries,
                'strategy': strategy,
                'results': [],
                'total_results': 0,
                'error': f'Unknown strategy: {strategy}',
                'timestamp': datetime.now().isoformat()
            }
        
        # Calculate total search time
        total_time = (time.time() - search_start) * 1000
        
        # Log multi-query search
        combined_query = " | ".join(queries)
        log_search_query(f"MULTI: {combined_query}", f"multi_{strategy}", len(raw_results), total_time)
        
        response = {
            'queries': queries,
            'strategy': strategy,
            'results': raw_results,
            'total_results': len(raw_results),
            'search_time_ms': round(total_time, 2),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Multi-query search ({strategy}) found {len(raw_results)} results in {total_time:.2f}ms")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in multi-query search: {e}")
        return {
            'queries': queries,
            'strategy': strategy,
            'results': [],
            'total_results': 0,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def faceted_search(query: str, facets: Dict[str, List[str]], limit: int = 10) -> Dict[str, Any]:
    """
    Perform faceted search with multiple filter dimensions
    
    Args:
        query: Search query text
        facets: Dictionary of facet filters (e.g., {'category': ['tech', 'science']})
        limit: Maximum number of results
        
    Returns:
        Dictionary containing faceted search results
    """
    try:
        # For this implementation, we'll focus on category facets
        # In a more complex system, you'd extend this to handle multiple facet types
        
        if 'category' in facets and facets['category']:
            # Perform searches for each category and combine
            all_results = []
            seen_docs = set()
            
            for category in facets['category']:
                # Perform semantic search within this category
                category_results = semantic_search(query, limit, category_filter=category)
                
                for result in category_results.get('results', []):
                    if result['id'] not in seen_docs:
                        seen_docs.add(result['id'])
                        result['matched_facets'] = {'category': category}
                        all_results.append(result)
            
            # Sort by similarity score and limit
            all_results.sort(key=lambda x: x.get('similarity_score', 0), reverse=True)
            results = all_results[:limit]
        else:
            # No facets specified, perform regular semantic search
            regular_results = semantic_search(query, limit)
            results = regular_results.get('results', [])
        
        response = {
            'query': query,
            'facets': facets,
            'results': results,
            'total_results': len(results),
            'search_type': 'faceted'
        }
        
        logger.info(f"✅ Faceted search found {len(results)} results")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error in faceted search: {e}")
        return {
            'query': query,
            'facets': facets,
            'results': [],
            'total_results': 0,
            'error': str(e)
        }

def generate_content_preview(content: str, max_length: int = 150) -> str:
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

def search_suggestions(partial_query: str, limit: int = 5) -> List[str]:
    """
    Generate search suggestions based on partial query
    
    Args:
        partial_query: Partial search query
        limit: Maximum number of suggestions
        
    Returns:
        List of suggested search terms
    """
    try:
        # This is a simple implementation
        # In production, you might use a dedicated suggestion engine
        
        if len(partial_query) < 2:
            return []
        
        # For now, return some basic suggestions
        # You could enhance this by analyzing popular searches, categories, etc.
        basic_suggestions = [
            "machine learning algorithms",
            "deep learning neural networks",
            "artificial intelligence applications",
            "data science techniques",
            "programming best practices",
            "web development frameworks",
            "database optimization",
            "cloud computing services"
        ]
        
        # Filter suggestions that contain the partial query
        suggestions = [
            suggestion for suggestion in basic_suggestions
            if partial_query.lower() in suggestion.lower()
        ]
        
        return suggestions[:limit]
        
    except Exception as e:
        logger.error(f"❌ Error generating search suggestions: {e}")
        return []
