"""
Setup Script for Vector Database Learning Project

This script helps you set up the learning environment step by step.
Run this to get started with your vector database journey!
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors gracefully."""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Success!")
        if result.stdout:
            print(f"Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed!")
        print(f"Error: {e.stderr.strip()}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required!")
        return False
    
    print("✅ Python version is compatible!")
    return True

def install_basic_requirements():
    """Install basic requirements for getting started."""
    print("\n📦 Installing basic requirements...")
    
    # Try to install basic requirements first
    basic_packages = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0", 
        "pydantic>=2.5.0",
        "numpy>=1.26.0",
        "scikit-learn>=1.3.0",
        "python-dotenv>=1.0.0"
    ]
    
    failed_packages = []
    
    for package in basic_packages:
        print(f"\n Installing {package}...")
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n⚠️  Some packages failed to install: {failed_packages}")
        print("You can continue with the basic examples and install these later.")
    else:
        print("\n🎉 All basic packages installed successfully!")
    
    return len(failed_packages) == 0

def test_imports():
    """Test if essential packages can be imported."""
    print("\n🧪 Testing package imports...")
    
    test_packages = [
        ("numpy", "np"),
        ("sklearn", "sklearn"),
        ("fastapi", "FastAPI"),
        ("pydantic", "BaseModel")
    ]
    
    success_count = 0
    
    for package, import_name in test_packages:
        try:
            if package == "numpy":
                import numpy as np
                print(f"✅ {package} - version {np.__version__}")
            elif package == "sklearn":
                import sklearn
                print(f"✅ {package} - version {sklearn.__version__}")
            elif package == "fastapi":
                from fastapi import FastAPI
                print(f"✅ {package} - imported successfully")
            elif package == "pydantic":
                from pydantic import BaseModel
                print(f"✅ {package} - imported successfully")
            
            success_count += 1
            
        except ImportError as e:
            print(f"❌ {package} - import failed: {e}")
    
    print(f"\n📊 Import test results: {success_count}/{len(test_packages)} packages working")
    return success_count == len(test_packages)

def create_env_file():
    """Create a basic .env file."""
    env_file = Path(".env")
    
    if env_file.exists():
        print("\n📄 .env file already exists, skipping creation")
        return
    
    print("\n📄 Creating .env file...")
    
    env_content = """# Vector Database Learning Environment Configuration

# Development Settings
DEBUG=true
LOG_LEVEL=info

# API Settings
API_HOST=127.0.0.1
API_PORT=8000

# Vector Settings
DEFAULT_VECTOR_DIMENSION=384
MAX_VECTOR_DIMENSION=2048

# Database Settings (for Module 3 - PostgreSQL + pgvector)
# Uncomment and configure when you reach Module 3
# DB_HOST=localhost
# DB_PORT=5432
# DB_NAME=vectordb
# DB_USER=postgres
# DB_PASSWORD=password
"""
    
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ .env file created successfully!")
    except Exception as e:
        print(f"❌ Failed to create .env file: {e}")

def show_getting_started():
    """Show getting started instructions."""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE! Ready to start learning!")
    print("="*60)
    
    print("\n📚 Your Vector Database Learning Journey:")
    print("\n1️⃣  START WITH MODULE 1 - Vector Basics:")
    print("   cd modules/01-basics")
    print("   python vector_operations.py")
    
    print("\n2️⃣  BUILD WEB APIs (Module 2):")
    print("   cd modules/02-fastapi")
    print("   python main.py")
    print("   # Then visit: http://localhost:8000/docs")
    
    print("\n3️⃣  ADVANCED DATABASE (Module 3):")
    print("   cd modules/03-pgvector")
    print("   # Follow setup.md for PostgreSQL installation")
    
    print("\n🔗 Quick Links:")
    print("   📖 Project README: README.md")
    print("   📋 Module 1 Guide: modules/01-basics/README.md")
    print("   🌐 API Documentation: http://localhost:8000/docs (when running)")
    
    print("\n💡 Tips:")
    print("   • Run examples in order (01-basics → 02-fastapi → 03-pgvector)")
    print("   • Each module builds on the previous one")
    print("   • Check README.md files in each module for detailed instructions")
    print("   • Use VS Code for the best development experience")
    
    print("\n🚀 Ready to become a vector database expert!")

def main():
    """Main setup function."""
    print("🎯 Vector Database Learning Project Setup")
    print("="*50)
    print("This setup will prepare your environment for learning")
    print("vector databases, pgvector, FastAPI, and similarity search!")
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed - Python version not compatible")
        return False
    
    # Install basic requirements
    install_success = install_basic_requirements()
    
    # Test imports
    import_success = test_imports()
    
    # Create environment file
    create_env_file()
    
    # Show final instructions
    show_getting_started()
    
    if install_success and import_success:
        print("\n✅ Setup completed successfully!")
        return True
    else:
        print("\n⚠️  Setup completed with some issues.")
        print("You can still proceed with the learning modules.")
        print("Install missing packages as needed.")
        return False

if __name__ == "__main__":
    main()
