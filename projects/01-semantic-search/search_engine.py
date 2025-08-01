# Semantic search engine that combines all components
# This is the main search logic that ties everything together

from database import get_database_manager
from embeddings import get_text_embedder
from document_processor import get_document_processor
from config import SEARCH_RESULTS_LIMIT, SIMILARITY_THRESHOLD
import os

class SemanticSearchEngine:
    """Main search engine that orchestrates document indexing and searching"""
    
    def __init__(self):
        """Initialize the search engine with all required components"""
        print("Initializing Semantic Search Engine...")
        
        # Initialize components
        self.db_manager = get_database_manager()
        self.text_embedder = get_text_embedder()
        self.document_processor = get_document_processor()
        
        # Check if all components initialized successfully
        if not self.db_manager:
            print("Error: Failed to initialize database manager")
        
        if not self.text_embedder:
            print("Error: Failed to initialize text embedder")
        
        if not self.document_processor:
            print("Error: Failed to initialize document processor")
        
        # Setup database if components are ready
        if self.db_manager:
            self.db_manager.setup_database()
        
        print("Search engine initialization complete")
    
    def index_document(self, file_path, title=None):
        """Index a document by processing it and storing embeddings"""
        if not all([self.db_manager, self.text_embedder, self.document_processor]):
            print("Error: Search engine not properly initialized")
            return False
        
        try:
            print(f"Starting to index document: {file_path}")
            
            # Process the document
            document_info = self.document_processor.process_document(file_path, title)
            if not document_info:
                print("Failed to process document")
                return False
            
            # Insert document into database
            document_id = self.db_manager.insert_document(
                title=document_info['title'],
                content=document_info['content'],
                file_path=document_info['file_path'],
                file_type=document_info['file_type']
            )
            
            if not document_id:
                print("Failed to insert document into database")
                return False
            
            # Generate and store embeddings for each chunk
            print(f"Generating embeddings for {len(document_info['chunks'])} chunks...")
            
            for chunk_index, chunk_text in enumerate(document_info['chunks']):
                # Generate embedding for this chunk
                embedding = self.text_embedder.embed_text(chunk_text)
                
                if embedding is not None:
                    # Store embedding in database
                    success = self.db_manager.insert_embedding(
                        document_id=document_id,
                        chunk_text=chunk_text,
                        chunk_index=chunk_index,
                        embedding=embedding
                    )
                    
                    if not success:
                        print(f"Failed to store embedding for chunk {chunk_index}")
                else:
                    print(f"Failed to generate embedding for chunk {chunk_index}")
            
            print(f"Document '{document_info['title']}' indexed successfully!")
            return True
            
        except Exception as e:
            print(f"Error indexing document: {e}")
            return False
    
    def search(self, query, limit=None):
        """Search for documents similar to the query"""
        if not all([self.db_manager, self.text_embedder]):
            print("Error: Search engine not properly initialized")
            return []
        
        if not limit:
            limit = SEARCH_RESULTS_LIMIT
        
        try:
            print(f"Searching for: '{query}'")
            
            # Generate embedding for the search query
            query_embedding = self.text_embedder.embed_text(query)
            if query_embedding is None:
                print("Failed to generate embedding for query")
                return []
            
            # Search for similar embeddings in database
            results = self.db_manager.search_similar_embeddings(query_embedding, limit)
            
            # Filter results by similarity threshold
            filtered_results = []
            for result in results:
                if result['similarity_score'] >= SIMILARITY_THRESHOLD:
                    filtered_results.append({
                        'title': result['title'],
                        'content': result['chunk_text'],
                        'file_path': result['file_path'],
                        'file_type': result['file_type'],
                        'similarity_score': round(result['similarity_score'], 3),
                        'distance': round(result['distance'], 3)
                    })
            
            print(f"Found {len(filtered_results)} relevant results")
            return filtered_results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []
    
    def list_documents(self):
        """List all indexed documents"""
        if not self.db_manager:
            print("Error: Database manager not initialized")
            return []
        
        try:
            documents = self.db_manager.get_all_documents()
            print(f"Found {len(documents)} indexed documents")
            return documents
            
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []
    
    def delete_document(self, document_id):
        """Delete a document from the index"""
        if not self.db_manager:
            print("Error: Database manager not initialized")
            return False
        
        try:
            success = self.db_manager.delete_document(document_id)
            return success
            
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
    
    def get_stats(self):
        """Get statistics about the search index"""
        if not self.db_manager:
            print("Error: Database manager not initialized")
            return {}
        
        try:
            documents = self.list_documents()
            
            if documents:
                total_docs = len(documents)
                total_words = sum(doc['word_count'] for doc in documents)
                total_chars = sum(doc['char_count'] for doc in documents)
                
                stats = {
                    'total_documents': total_docs,
                    'total_words': total_words,
                    'total_characters': total_chars,
                    'average_words_per_doc': round(total_words / total_docs, 2) if total_docs > 0 else 0
                }
            else:
                stats = {
                    'total_documents': 0,
                    'total_words': 0,
                    'total_characters': 0,
                    'average_words_per_doc': 0
                }
            
            return stats
            
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {}
    
    def close(self):
        """Clean up resources"""
        if self.db_manager:
            self.db_manager.close()
        print("Search engine closed")

# Helper function to create search engine
def get_search_engine():
    """Create and return a search engine instance"""
    return SemanticSearchEngine()

# Example usage and testing
def test_search_engine():
    """Test the complete search engine functionality"""
    print("Testing Semantic Search Engine...")
    
    # Create search engine
    engine = get_search_engine()
    
    # Create some test documents
    test_docs = [
        {
            'path': 'data/test_doc1.txt',
            'title': 'AI and Machine Learning',
            'content': '''
            Artificial Intelligence and Machine Learning are revolutionizing technology.
            Deep learning algorithms can process vast amounts of data.
            Neural networks mimic the human brain to solve complex problems.
            Computer vision enables machines to understand images and videos.
            Natural language processing helps computers understand human speech.
            '''
        },
        {
            'path': 'data/test_doc2.txt',
            'title': 'Web Development',
            'content': '''
            Web development involves creating websites and web applications.
            Frontend technologies include HTML, CSS, and JavaScript.
            Backend development uses languages like Python, Java, and Node.js.
            Databases store and manage application data efficiently.
            API design enables communication between different software components.
            '''
        },
        {
            'path': 'data/test_doc3.txt',
            'title': 'Data Science',
            'content': '''
            Data science combines statistics, programming, and domain expertise.
            Data analysis helps organizations make informed decisions.
            Visualization tools help communicate insights effectively.
            Statistical modeling predicts future trends and patterns.
            Big data technologies handle massive datasets efficiently.
            '''
        }
    ]
    
    # Create test documents
    os.makedirs('data', exist_ok=True)
    for doc in test_docs:
        with open(doc['path'], 'w', encoding='utf-8') as f:
            f.write(doc['content'])
        
        # Index the document
        print(f"\nIndexing: {doc['title']}")
        engine.index_document(doc['path'], doc['title'])
    
    # Test searches
    test_queries = [
        "artificial intelligence and neural networks",
        "building websites with JavaScript",
        "analyzing data and statistics",
        "computer vision algorithms"
    ]
    
    print("\n" + "="*50)
    print("TESTING SEARCHES")
    print("="*50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        results = engine.search(query, limit=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (Score: {result['similarity_score']})")
                print(f"   Content: {result['content'][:100]}...")
                print()
        else:
            print("No results found")
    
    # Show statistics
    print("\n" + "="*50)
    print("INDEX STATISTICS")
    print("="*50)
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Clean up
    engine.close()
    print("\nSearch engine test completed!")

if __name__ == "__main__":
    test_search_engine()
