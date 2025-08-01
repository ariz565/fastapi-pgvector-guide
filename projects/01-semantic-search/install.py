# Installation and setup script for Semantic Document Search
# Run this script to set up everything needed for the project

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and display the result"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or higher")
        return False

def check_postgresql():
    """Check if PostgreSQL is accessible"""
    print("🐘 Checking PostgreSQL connection...")
    try:
        import psycopg2
        from config import DATABASE_CONFIG
        
        # Try to connect to PostgreSQL
        connection = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['username'],
            password=DATABASE_CONFIG['password'],
            database='postgres'  # Connect to default database first
        )
        connection.close()
        print("✅ PostgreSQL connection successful")
        return True
        
    except ImportError:
        print("❌ psycopg2 not installed (will install with requirements)")
        return True  # Will be installed later
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        print("Please check your PostgreSQL installation and configuration")
        return False

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python packages...")
    
    # Check if requirements.txt exists
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    # Install requirements
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )

def setup_database():
    """Setup database and tables"""
    print("🗄️ Setting up database...")
    
    try:
        # Import and run database setup
        from setup_database import main as setup_db_main
        setup_db_main()
        print("✅ Database setup completed")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = ['data', 'data/uploads']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"✅ Directory already exists: {directory}")
    
    return True

def test_installation():
    """Test if everything is working"""
    print("🧪 Testing installation...")
    
    try:
        # Test imports
        from search_engine import get_search_engine
        from embeddings import get_text_embedder
        
        # Test search engine initialization
        engine = get_search_engine()
        
        if engine.db_manager and engine.text_embedder:
            print("✅ All components initialized successfully")
            engine.close()
            return True
        else:
            print("❌ Some components failed to initialize")
            return False
            
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        return False

def main():
    """Main installation process"""
    print("🚀 Semantic Document Search - Installation Script")
    print("=" * 60)
    print()
    
    # Step 1: Check Python version
    if not check_python_version():
        return
    
    # Step 2: Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        return
    
    # Step 3: Install Python packages
    if not install_requirements():
        print("❌ Failed to install Python packages")
        return
    
    # Step 4: Check PostgreSQL (after installing psycopg2)
    if not check_postgresql():
        print("❌ PostgreSQL check failed")
        print("\nPlease ensure:")
        print("1. PostgreSQL is installed and running")
        print("2. Update database credentials in config.py")
        print("3. Install pgvector extension")
        return
    
    # Step 5: Setup database
    if not setup_database():
        print("❌ Database setup failed")
        return
    
    # Step 6: Test installation
    if not test_installation():
        print("❌ Installation test failed")
        return
    
    # Success!
    print("\n" + "=" * 60)
    print("🎉 INSTALLATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print()
    print("📋 What's been set up:")
    print("✅ Python packages installed")
    print("✅ Database and tables created")
    print("✅ Directories created")
    print("✅ All components tested")
    print()
    print("🚀 Ready to use! Try these commands:")
    print("   python example_demo.py    - See a demo")
    print("   python cli.py             - Command-line interface")
    print("   python app/main.py        - Web interface")
    print()
    print("🌐 Web interface will be at: http://localhost:8000")
    print("📚 API docs will be at: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
