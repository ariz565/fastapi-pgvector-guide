# Example script demonstrating semantic search functionality
# Run this after setting up the database to see the system in action

from search_engine import get_search_engine
import os

def main():
    """Demonstrate semantic search with examples"""
    print("🔍 Semantic Document Search - Example Demo")
    print("=" * 50)
    
    # Initialize search engine
    print("Initializing search engine...")
    engine = get_search_engine()
    
    # Check if initialization was successful
    if not engine.db_manager or not engine.text_embedder:
        print("❌ Failed to initialize search engine")
        print("Make sure to run setup_database.py first")
        return
    
    print("✅ Search engine initialized successfully!")
    
    # Create sample documents
    sample_documents = [
        {
            'title': 'Introduction to Machine Learning',
            'content': '''
            Machine Learning is a branch of artificial intelligence that enables computers to learn 
            and make decisions from data without being explicitly programmed. It uses algorithms 
            to identify patterns in data and make predictions or classifications.
            
            There are three main types of machine learning: supervised learning (learning from 
            labeled examples), unsupervised learning (finding patterns in unlabeled data), and 
            reinforcement learning (learning through trial and error with rewards).
            
            Popular machine learning algorithms include linear regression, decision trees, random 
            forests, support vector machines, and neural networks. Deep learning is a subset of 
            machine learning that uses multi-layered neural networks.
            ''',
            'filename': 'ml_intro.txt'
        },
        {
            'title': 'Web Development Fundamentals',
            'content': '''
            Web development involves creating websites and web applications for the internet. 
            It consists of two main areas: frontend development (client-side) and backend 
            development (server-side).
            
            Frontend development focuses on what users see and interact with. It uses HTML for 
            structure, CSS for styling, and JavaScript for interactivity. Modern frameworks 
            like React, Vue.js, and Angular help build dynamic user interfaces.
            
            Backend development handles server-side logic, databases, and APIs. Popular backend 
            languages include Python, JavaScript (Node.js), Java, PHP, and Ruby. Databases 
            like MySQL, PostgreSQL, and MongoDB store application data.
            ''',
            'filename': 'web_dev.txt'
        },
        {
            'title': 'Data Science Overview',
            'content': '''
            Data Science is an interdisciplinary field that combines statistics, programming, 
            and domain knowledge to extract insights from data. It involves collecting, cleaning, 
            analyzing, and interpreting large datasets to solve business problems.
            
            The data science process typically follows these steps: problem definition, data 
            collection, data cleaning, exploratory data analysis, modeling, evaluation, and 
            deployment. Tools commonly used include Python, R, SQL, Jupyter notebooks, and 
            visualization libraries.
            
            Data scientists work with various types of data including structured data (databases), 
            unstructured data (text, images), and streaming data. Machine learning and statistical 
            modeling are key components of data science projects.
            ''',
            'filename': 'data_science.txt'
        }
    ]
    
    # Create documents and index them
    print("\n📝 Creating and indexing sample documents...")
    os.makedirs('data', exist_ok=True)
    
    for doc in sample_documents:
        # Write document to file
        file_path = os.path.join('data', doc['filename'])
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(doc['content'])
        
        # Index the document
        print(f"Indexing: {doc['title']}")
        success = engine.index_document(file_path, doc['title'])
        
        if success:
            print(f"✅ Successfully indexed: {doc['title']}")
        else:
            print(f"❌ Failed to index: {doc['title']}")
    
    # Demonstrate semantic search
    print("\n🔍 Demonstrating semantic search...")
    print("=" * 50)
    
    # Example searches that show semantic understanding
    example_queries = [
        {
            'query': 'artificial intelligence and neural networks',
            'description': 'Should find machine learning content'
        },
        {
            'query': 'building websites with HTML and CSS',
            'description': 'Should find web development content'
        },
        {
            'query': 'analyzing data and statistics',
            'description': 'Should find data science content'
        },
        {
            'query': 'deep learning algorithms',
            'description': 'Should find machine learning content (semantic match)'
        },
        {
            'query': 'frontend frameworks and user interfaces',
            'description': 'Should find web development content (semantic match)'
        }
    ]
    
    for example in example_queries:
        print(f"\nQuery: '{example['query']}'")
        print(f"Expected: {example['description']}")
        print("-" * 40)
        
        results = engine.search(example['query'], limit=3)
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']} (Score: {result['similarity_score']})")
                # Show first 100 characters of matched content
                content_preview = result['content'][:100].replace('\n', ' ').strip()
                print(f"   Preview: {content_preview}...")
                print()
        else:
            print("   No results found")
        
        print()
    
    # Show statistics
    print("📊 Search Index Statistics")
    print("=" * 30)
    stats = engine.get_stats()
    for key, value in stats.items():
        formatted_key = key.replace('_', ' ').title()
        print(f"{formatted_key}: {value}")
    
    # Clean up
    engine.close()
    print("\n✅ Demo completed successfully!")
    print("\nNext steps:")
    print("1. Try the web interface: python app/main.py")
    print("2. Try the CLI: python cli.py")
    print("3. Upload your own documents and test semantic search!")

if __name__ == "__main__":
    main()
