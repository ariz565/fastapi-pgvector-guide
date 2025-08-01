# Command-line interface for Semantic Document Search
# Simple CLI tool to test the search engine functionality

import os
import sys
from search_engine import get_search_engine

def print_banner():
    """Print application banner"""
    print("=" * 60)
    print("    SEMANTIC DOCUMENT SEARCH ENGINE - CLI")
    print("=" * 60)
    print()

def print_menu():
    """Print main menu options"""
    print("\nChoose an option:")
    print("1. Index a document")
    print("2. Search documents")
    print("3. List all documents")
    print("4. Get statistics")
    print("5. Delete a document")
    print("6. Exit")
    print("-" * 30)

def index_document(engine):
    """Handle document indexing"""
    print("\n--- INDEX DOCUMENT ---")
    
    file_path = input("Enter file path: ").strip()
    
    if not file_path:
        print("File path cannot be empty")
        return
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    
    title = input("Enter document title (optional): ").strip()
    if not title:
        title = None
    
    print("Indexing document...")
    success = engine.index_document(file_path, title)
    
    if success:
        print("✓ Document indexed successfully!")
    else:
        print("✗ Failed to index document")

def search_documents(engine):
    """Handle document searching"""
    print("\n--- SEARCH DOCUMENTS ---")
    
    query = input("Enter search query: ").strip()
    
    if not query:
        print("Query cannot be empty")
        return
    
    try:
        limit = int(input("Number of results (default 5): ") or "5")
    except ValueError:
        limit = 5
    
    print(f"Searching for: '{query}'...")
    results = engine.search(query, limit)
    
    if not results:
        print("No results found")
        return
    
    print(f"\nFound {len(results)} results:")
    print("-" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']}")
        print(f"   Score: {result['similarity_score']}")
        print(f"   Content: {result['content'][:150]}...")
        print(f"   File: {result['file_path']}")
        print()

def list_documents(engine):
    """Handle document listing"""
    print("\n--- ALL DOCUMENTS ---")
    
    documents = engine.list_documents()
    
    if not documents:
        print("No documents found")
        return
    
    print(f"Found {len(documents)} documents:")
    print("-" * 50)
    
    for doc in documents:
        print(f"ID: {doc['id']} | {doc['title']}")
        print(f"   Type: {doc['file_type']} | Words: {doc['word_count']}")
        print(f"   Uploaded: {doc['upload_date']}")
        print()

def show_statistics(engine):
    """Handle statistics display"""
    print("\n--- STATISTICS ---")
    
    stats = engine.get_stats()
    
    if not stats:
        print("No statistics available")
        return
    
    print("Search Index Statistics:")
    print("-" * 30)
    print(f"Total Documents: {stats['total_documents']}")
    print(f"Total Words: {stats['total_words']}")
    print(f"Total Characters: {stats['total_characters']}")
    print(f"Average Words per Document: {stats['average_words_per_doc']}")

def delete_document(engine):
    """Handle document deletion"""
    print("\n--- DELETE DOCUMENT ---")
    
    # First show all documents
    documents = engine.list_documents()
    
    if not documents:
        print("No documents to delete")
        return
    
    print("Available documents:")
    for doc in documents:
        print(f"ID: {doc['id']} | {doc['title']}")
    
    try:
        doc_id = int(input("\nEnter document ID to delete: "))
    except ValueError:
        print("Invalid document ID")
        return
    
    confirm = input(f"Are you sure you want to delete document ID {doc_id}? (y/N): ")
    
    if confirm.lower() == 'y':
        success = engine.delete_document(doc_id)
        
        if success:
            print("✓ Document deleted successfully!")
        else:
            print("✗ Failed to delete document")
    else:
        print("Deletion cancelled")

def create_sample_documents():
    """Create sample documents for testing"""
    print("Creating sample documents for testing...")
    
    sample_docs = [
        {
            'filename': 'ai_basics.txt',
            'content': '''
            Artificial Intelligence (AI) is a branch of computer science that aims to create machines 
            capable of intelligent behavior. Machine learning is a subset of AI that enables computers 
            to learn and improve from experience without being explicitly programmed.
            
            Deep learning uses neural networks with multiple layers to model and understand complex patterns 
            in data. This technology has revolutionized fields like computer vision, natural language 
            processing, and speech recognition.
            
            AI applications are everywhere today - from recommendation systems on streaming platforms 
            to autonomous vehicles and virtual assistants like Siri and Alexa.
            '''
        },
        {
            'filename': 'web_development.txt',
            'content': '''
            Web development is the process of building and maintaining websites and web applications. 
            It involves both frontend development (what users see) and backend development (server-side logic).
            
            Frontend technologies include HTML for structure, CSS for styling, and JavaScript for 
            interactivity. Modern frameworks like React, Vue.js, and Angular make frontend development 
            more efficient and maintainable.
            
            Backend development involves server-side programming languages like Python, Java, Node.js, 
            and PHP. Databases store and manage data, while APIs enable communication between different 
            parts of an application.
            '''
        },
        {
            'filename': 'data_science.txt',
            'content': '''
            Data science is an interdisciplinary field that combines statistics, programming, and 
            domain expertise to extract insights from data. It involves collecting, cleaning, 
            analyzing, and interpreting large datasets.
            
            Key tools in data science include Python and R for programming, SQL for database queries, 
            and libraries like pandas, NumPy, and scikit-learn for data manipulation and machine learning.
            
            Data visualization is crucial for communicating findings effectively. Tools like Matplotlib, 
            Seaborn, and Tableau help create compelling charts and dashboards that tell a story with data.
            '''
        }
    ]
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Create sample files
    for doc in sample_docs:
        file_path = os.path.join('data', doc['filename'])
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc['content'])
    
    print(f"Created {len(sample_docs)} sample documents in 'data/' directory")
    print("You can now index these documents using option 1")

def main():
    """Main CLI application"""
    print_banner()
    
    # Initialize search engine
    print("Initializing search engine...")
    engine = get_search_engine()
    
    # Check if initialization was successful
    if not engine.db_manager or not engine.text_embedder:
        print("✗ Failed to initialize search engine")
        print("\nTroubleshooting:")
        print("1. Make sure PostgreSQL is running")
        print("2. Check database configuration in config.py")
        print("3. Install required packages: pip install -r requirements.txt")
        print("4. Run setup_database.py to create the database")
        return
    
    print("✓ Search engine initialized successfully!")
    
    # Check if sample documents exist
    sample_files = ['data/ai_basics.txt', 'data/web_development.txt', 'data/data_science.txt']
    if not any(os.path.exists(f) for f in sample_files):
        create_sample = input("\nWould you like to create sample documents for testing? (y/N): ")
        if create_sample.lower() == 'y':
            create_sample_documents()
    
    # Main application loop
    while True:
        try:
            print_menu()
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                index_document(engine)
            elif choice == '2':
                search_documents(engine)
            elif choice == '3':
                list_documents(engine)
            elif choice == '4':
                show_statistics(engine)
            elif choice == '5':
                delete_document(engine)
            elif choice == '6':
                print("\nGoodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")
    
    # Clean up
    engine.close()

if __name__ == "__main__":
    main()
