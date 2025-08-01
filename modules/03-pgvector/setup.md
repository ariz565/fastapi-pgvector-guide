# PostgreSQL + pgvector Setup Guide

Complete guide to setting up PostgreSQL with pgvector extension for vector similarity search.

## 🎯 Setup Options

Choose the method that works best for your system:

### Option 1: Docker (Recommended for Learning)

**Easiest way to get started:**

```bash
# Pull and run PostgreSQL with pgvector
docker run --name postgres-vector \
  -e POSTGRES_PASSWORD=password \
  -e POSTGRES_DB=vectordb \
  -p 5432:5432 \
  -d ankane/pgvector

# Verify it's running
docker ps
```

**Connection details:**

- Host: `localhost`
- Port: `5432`
- Database: `vectordb`
- User: `postgres`
- Password: `password`

### Option 2: Local PostgreSQL Installation

**For Windows:**

1. **Download PostgreSQL:**

   - Go to https://www.postgresql.org/download/windows/
   - Download and install PostgreSQL 14+ (includes pgAdmin)

2. **Install pgvector extension:**

   ```bash
   # Using pre-compiled binaries (recommended)
   # Download from: https://github.com/pgvector/pgvector/releases

   # Or build from source (requires Visual Studio Build Tools)
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   make install
   ```

3. **Enable extension:**
   ```sql
   -- Connect to your database and run:
   CREATE EXTENSION vector;
   ```

### Option 3: Cloud Database

**For production or team use:**

Popular cloud providers with pgvector support:

- **Supabase**: Built-in pgvector support
- **Neon**: PostgreSQL with vector extensions
- **AWS RDS**: Requires manual extension installation

## 🔧 Verification

Test your setup with these steps:

### 1. Connect to Database

**Using psql:**

```bash
psql -h localhost -U postgres -d vectordb
```

**Using Python:**

```python
import psycopg2

conn = psycopg2.connect(
    host="localhost",
    database="vectordb",
    user="postgres",
    password="password"
)
print("✅ Connected successfully!")
conn.close()
```

### 2. Test pgvector Extension

```sql
-- Check if extension is available
SELECT * FROM pg_available_extensions WHERE name = 'vector';

-- Create extension if not exists
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

### 3. Test Vector Operations

```sql
-- Create a test table
CREATE TABLE test_vectors (
    id SERIAL PRIMARY KEY,
    embedding vector(3)
);

-- Insert test vectors
INSERT INTO test_vectors (embedding) VALUES
    ('[1,2,3]'),
    ('[4,5,6]'),
    ('[7,8,9]');

-- Test similarity search
SELECT id, embedding, embedding <-> '[1,2,3]' AS distance
FROM test_vectors
ORDER BY embedding <-> '[1,2,3]'
LIMIT 3;

-- Clean up
DROP TABLE test_vectors;
```

## 🛠 Environment Configuration

Create a `.env` file in the module directory:

```env
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=vectordb
DB_USER=postgres
DB_PASSWORD=password

# Connection Pool Settings
DB_MIN_POOL_SIZE=1
DB_MAX_POOL_SIZE=10

# Vector Settings
DEFAULT_VECTOR_DIMENSION=384
MAX_VECTOR_DIMENSION=2048
```

## 📊 Performance Tuning

### PostgreSQL Configuration

Add these settings to `postgresql.conf`:

```ini
# Memory settings (adjust based on your system)
shared_preload_libraries = 'vector'
shared_buffers = 256MB
work_mem = 4MB
maintenance_work_mem = 64MB

# Vector-specific settings
max_parallel_workers_per_gather = 2
```

### Connection Pool Setup

For production applications:

```python
# requirements.txt addition
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.23

# Connection pool example
from sqlalchemy.ext.asyncio import create_async_engine

engine = create_async_engine(
    "postgresql+asyncpg://postgres:password@localhost/vectordb",
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True
)
```

## 🚨 Troubleshooting

### Common Issues

**1. Extension not found:**

```
ERROR: extension "vector" is not available
```

**Solution:** Install pgvector properly, ensure PostgreSQL version 11+

**2. Permission denied:**

```
ERROR: permission denied to create extension "vector"
```

**Solution:** Connect as superuser (postgres) or ask admin to enable

**3. Vector dimension mismatch:**

```
ERROR: vector dimension 512 does not match column dimension 384
```

**Solution:** Use consistent vector dimensions or modify table schema

**4. Connection refused:**

```
psycopg2.OperationalError: connection refused
```

**Solution:** Check PostgreSQL is running, verify host/port/credentials

### Docker Troubleshooting

```bash
# Check if container is running
docker ps -a

# View container logs
docker logs postgres-vector

# Restart container
docker restart postgres-vector

# Connect to container shell
docker exec -it postgres-vector psql -U postgres -d vectordb
```

## ✅ Ready to Code!

Once setup is complete, verify everything works:

```python
# test_connection.py
import psycopg2
import numpy as np

def test_setup():
    try:
        # Connect
        conn = psycopg2.connect(
            host="localhost",
            database="vectordb",
            user="postgres",
            password="password"
        )
        cur = conn.cursor()

        # Test extension
        cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        if cur.fetchone():
            print("✅ pgvector extension is installed")
        else:
            print("❌ pgvector extension not found")
            return False

        # Test vector operations
        cur.execute("""
            CREATE TEMPORARY TABLE temp_test (
                id SERIAL PRIMARY KEY,
                vec vector(3)
            );
        """)

        cur.execute("INSERT INTO temp_test (vec) VALUES ('[1,2,3]'), ('[4,5,6]');")
        cur.execute("SELECT vec <-> '[1,2,3]' FROM temp_test;")

        results = cur.fetchall()
        print(f"✅ Vector similarity test passed: {results}")

        conn.close()
        print("🎉 Setup verification complete!")
        return True

    except Exception as e:
        print(f"❌ Setup verification failed: {e}")
        return False

if __name__ == "__main__":
    test_setup()
```

Run this test:

```bash
python test_connection.py
```

If you see all ✅ checks, you're ready to proceed with the pgvector examples!

## 🔜 Next Steps

1. Run `python basic_operations.py` to learn vector CRUD operations
2. Explore `indexing_performance.py` for optimization techniques
3. Build a complete application with `document_search_db.py`
