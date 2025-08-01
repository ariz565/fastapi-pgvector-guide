# Database setup script for Semantic Document Search
# Run this script to create the database and enable pgvector extension

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from config import DATABASE_CONFIG

def create_database():
    """Create the database if it doesn't exist"""
    try:
        # Connect to PostgreSQL server (not to specific database)
        connection = psycopg2.connect(
            host=DATABASE_CONFIG['host'],
            port=DATABASE_CONFIG['port'],
            user=DATABASE_CONFIG['username'],
            password=DATABASE_CONFIG['password'],
            database='postgres'  # Default database to connect to
        )
        
        connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()
        
        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DATABASE_CONFIG['database']}'")
        exists = cursor.fetchone()
        
        if not exists:
            # Create the database
            cursor.execute(f"CREATE DATABASE {DATABASE_CONFIG['database']}")
            print(f"Database '{DATABASE_CONFIG['database']}' created successfully")
        else:
            print(f"Database '{DATABASE_CONFIG['database']}' already exists")
        
        cursor.close()
        connection.close()
        return True
        
    except Exception as e:
        print(f"Error creating database: {e}")
        return False

def setup_pgvector():
    """Enable pgvector extension and create tables"""
    try:
        # Import here to avoid issues if database module has import errors
        from database import get_database_manager
        
        # Get database manager
        db_manager = get_database_manager()
        if not db_manager:
            print("Failed to connect to database")
            return False
        
        # Setup database (creates tables and enables pgvector)
        success = db_manager.setup_database()
        
        # Close connection
        db_manager.close()
        
        return success
        
    except Exception as e:
        print(f"Error setting up pgvector: {e}")
        return False

def main():
    """Main setup function"""
    print("Setting up Semantic Document Search Database...")
    print("=" * 50)
    
    # Step 1: Create database
    print("Step 1: Creating database...")
    if not create_database():
        print("Failed to create database. Please check your PostgreSQL connection.")
        return
    
    # Step 2: Setup pgvector and tables
    print("\nStep 2: Setting up pgvector extension and tables...")
    if not setup_pgvector():
        print("Failed to setup pgvector. Please ensure pgvector extension is installed.")
        return
    
    print("\n" + "=" * 50)
    print("Database setup completed successfully!")
    print("\nYou can now run the application with:")
    print("python app/main.py")
    print("\nOr install dependencies first:")
    print("pip install -r requirements.txt")

if __name__ == "__main__":
    main()
