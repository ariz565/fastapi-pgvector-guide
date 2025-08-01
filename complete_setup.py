"""
Complete Vector Database Learning Setup

This script sets up everything you need to master vector databases,
including FAISS, OpenSearch, pgvector, and all the learning modules!
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    print("🎯" + "="*60 + "🎯")
    print("    VECTOR DATABASE MASTERY - COMPLETE SETUP")
    print("🎯" + "="*60 + "🎯")
    print()
    print("🚀 This setup will prepare you to learn:")
    print("   • Vector fundamentals and similarity search")
    print("   • FastAPI for building vector-powered APIs")
    print("   • PostgreSQL + pgvector for production databases")
    print("   • FAISS for high-performance vector search")
    print("   • OpenSearch for distributed vector systems")
    print("   • Real-world projects and applications")
    print()

def check_system_requirements():
    """Check system requirements."""
    print("🔍 Checking System Requirements")
    print("=" * 32)
    
    # Python version
    python_version = sys.version_info
    print(f"Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required!")
        return False
    
    # Operating system
    os_name = platform.system()
    print(f"OS: {os_name}")
    
    # Available space (rough check)
    try:
        import shutil
        free_space_gb = shutil.disk_usage('.').free / (1024**3)
        print(f"Free Space: {free_space_gb:.1f} GB")
        
        if free_space_gb < 2:
            print("⚠️  Low disk space - may have issues downloading dependencies")
    except:
        print("Free Space: Unable to check")
    
    print("✅ System requirements check passed!")
    return True

def install_basic_packages():
    """Install essential packages for learning."""
    print("\n📦 Installing Essential Packages")
    print("=" * 32)
    
    # Essential packages for all modules
    essential_packages = [
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0", 
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "python-dotenv>=1.0.0",
        "requests>=2.31.0"
    ]
    
    print("Installing core packages...")
    for package in essential_packages:
        print(f"  Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"  ✅ {package}")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ {package} - {e}")
    
    return True

def install_advanced_packages():
    """Install advanced packages (FAISS, OpenSearch, etc.)."""
    print("\n🚀 Installing Advanced Packages")
    print("=" * 33)
    
    advanced_packages = [
        ("faiss-cpu", "FAISS for high-performance vector search"),
        ("opensearch-py", "OpenSearch client for distributed search"),
        ("sentence-transformers", "Text embedding generation"),
        ("matplotlib", "Visualization for performance analysis"),
        ("psycopg2-binary", "PostgreSQL client (optional)")
    ]
    
    for package, description in advanced_packages:
        print(f"\n📦 {description}")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  {package} installation failed - you can install it later")
            print(f"   Command: pip install {package}")

def create_learning_structure():
    """Create additional learning resources."""
    print("\n📁 Creating Learning Resources")
    print("=" * 30)
    
    # Create data directory with sample files
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Sample documents for testing
    sample_docs = [
        ("tech_article.txt", "Machine learning and artificial intelligence are transforming technology."),
        ("health_guide.txt", "Regular exercise and proper nutrition are essential for good health."),
        ("travel_blog.txt", "Exploring new destinations and experiencing different cultures enriches life."),
        ("finance_tips.txt", "Smart budgeting and investing strategies help build financial security.")
    ]
    
    for filename, content in sample_docs:
        filepath = data_dir / filename
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Created sample file: {filename}")
    
    # Create configuration files
    config_files = {
        ".env": """# Vector Database Learning Configuration
DEBUG=true
LOG_LEVEL=info

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# Vector Settings
DEFAULT_VECTOR_DIMENSION=384
MAX_VECTOR_DIMENSION=2048

# Database Configuration (for Module 3)
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=vectordb
# DB_USER=postgres
# DB_PASSWORD=password

# OpenSearch Configuration (for Module 4)
# OPENSEARCH_HOST=localhost
# OPENSEARCH_PORT=9200
""",
        "learning_progress.md": """# Your Vector Database Learning Progress

Track your journey through the modules and projects!

## Module 1: Vector Basics ⏳
- [ ] vector_operations.py
- [ ] similarity_metrics.py  
- [ ] simple_search.py
- [ ] text_embeddings.py

## Module 2: FastAPI + Vectors ⏳
- [ ] Basic API (main.py)
- [ ] Document search API
- [ ] Advanced features

## Module 3: PostgreSQL + pgvector ⏳
- [ ] Database setup
- [ ] Basic operations
- [ ] Performance optimization

## Module 4: FAISS & OpenSearch ⏳
- [ ] FAISS basics
- [ ] OpenSearch integration
- [ ] Performance comparison

## Projects ⏳
- [ ] Semantic Document Search
- [ ] Recommendation Engine
- [ ] Image Similarity Search

## Notes
Add your learning notes and insights here!
"""
    }
    
    for filename, content in config_files.items():
        filepath = Path(filename)
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Created: {filename}")

def test_installations():
    """Test that key packages work correctly."""
    print("\n🧪 Testing Package Installations")
    print("=" * 32)
    
    tests = [
        ("NumPy", "import numpy as np; print(f'NumPy {np.__version__} works!')"),
        ("scikit-learn", "import sklearn; print(f'scikit-learn {sklearn.__version__} works!')"),
        ("FastAPI", "from fastapi import FastAPI; print('FastAPI works!')"),
        ("FAISS", "import faiss; print(f'FAISS works!')"),
        ("OpenSearch", "from opensearchpy import OpenSearch; print('OpenSearch client works!')"),
        ("Sentence Transformers", "from sentence_transformers import SentenceTransformer; print('SentenceTransformers works!')")
    ]
    
    working_packages = []
    failed_packages = []
    
    for package_name, test_code in tests:
        try:
            exec(test_code)
            working_packages.append(package_name)
            print(f"✅ {package_name}")
        except ImportError:
            failed_packages.append(package_name)
            print(f"⚠️  {package_name} - not available (can install later)")
        except Exception as e:
            failed_packages.append(package_name)
            print(f"❌ {package_name} - error: {e}")
    
    print(f"\n📊 Results: {len(working_packages)}/{len(tests)} packages working")
    
    if len(working_packages) >= 4:  # Core packages
        print("✅ You have enough packages to start learning!")
        return True
    else:
        print("⚠️  Some core packages missing - you may encounter issues")
        return False

def show_next_steps():
    """Show what to do next."""
    print("\n" + "🎓" + "="*50 + "🎓")
    print("    SETUP COMPLETE - START YOUR JOURNEY!")
    print("🎓" + "="*50 + "🎓")
    
    print("\n🚀 Your Learning Path:")
    print()
    print("1️⃣  START WITH VECTOR BASICS:")
    print("   cd modules/01-basics")
    print("   python vector_operations.py")
    print("   python similarity_metrics.py")
    print("   python simple_search.py")
    print("   python text_embeddings.py")
    
    print("\n2️⃣  BUILD WEB APIs:")
    print("   cd modules/02-fastapi")
    print("   python main.py")
    print("   # Visit: http://localhost:8000/docs")
    
    print("\n3️⃣  SETUP DATABASE (when ready):")
    print("   cd modules/03-pgvector")
    print("   # Follow setup.md for PostgreSQL installation")
    
    print("\n4️⃣  MASTER ADVANCED TOOLS:")
    print("   cd modules/04-advanced")
    print("   python faiss_basics.py")
    print("   python opensearch_vectors.py")
    print("   python vector_db_comparison.py")
    
    print("\n5️⃣  BUILD REAL PROJECTS:")
    print("   cd projects/01-semantic-search")
    print("   # Follow README.md for complete application")
    
    print("\n📚 Learning Resources:")
    print("   • Each module has detailed README.md files")
    print("   • Code examples include extensive comments")
    print("   • Track progress in learning_progress.md")
    print("   • Sample data available in data/ directory")
    
    print("\n💡 Pro Tips:")
    print("   • Start with Module 1 even if you know vectors")
    print("   • Run every code example - hands-on learning works best")
    print("   • Experiment with parameters and see what happens")
    print("   • Build your own variations of the examples")
    
    print("\n🆘 Need Help?")
    print("   • Check module README files for detailed explanations")
    print("   • Look at code comments for implementation details")
    print("   • Try different parameters in examples")
    print("   • Build small test cases to understand concepts")
    
    print(f"\n🌟 Ready to become a vector database expert!")
    print("🌟 Happy learning! 🌟")

def main():
    """Run the complete setup process."""
    print_banner()
    
    if not check_system_requirements():
        print("❌ System requirements not met. Please fix and try again.")
        return False
    
    install_basic_packages()
    install_advanced_packages()
    create_learning_structure()
    
    if test_installations():
        show_next_steps()
        return True
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("You can still start learning with the working packages.")
        print("Install missing packages as needed during your journey.")
        show_next_steps()
        return False

if __name__ == "__main__":
    main()
