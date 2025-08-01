# 🚀 Quick Start Guide - Semantic Document Search

## Overview

This is a beginner-friendly semantic search engine that finds documents by meaning, not just keywords.

## What You Need

1. Python 3.8 or higher
2. PostgreSQL with pgvector extension
3. 15 minutes to set up

## Installation (3 Easy Steps)

### Step 1: Install Python Packages

```bash
pip install -r requirements.txt
```

### Step 2: Configure Database

1. Edit `config.py` and update your PostgreSQL password:

```python
DATABASE_CONFIG = {
    'password': 'your_actual_password',  # Change this!
}
```

### Step 3: Setup Database

```bash
python setup_database.py
```

## Try It Out!

### Option 1: See a Demo

```bash
python example_demo.py
```

### Option 2: Use Command Line

```bash
python cli.py
```

### Option 3: Use Web Interface

```bash
python app/main.py
```

Then visit: http://localhost:8000

## How It Works (Simple Explanation)

1. **Upload a document** → System reads the text
2. **Text becomes vectors** → AI converts words to numbers
3. **Search with meaning** → Your query also becomes vectors
4. **Find similar vectors** → Math finds similar documents
5. **Get ranked results** → Best matches come first

## Test Searches

Try these searches to see semantic understanding:

- "artificial intelligence" → finds docs about AI, ML, neural networks
- "building websites" → finds docs about web development, HTML, CSS
- "data analysis" → finds docs about data science, statistics

## File Structure

```
config.py           - Settings and configuration
database.py         - Database connection and operations
embeddings.py       - Convert text to vectors using AI
document_processor.py - Handle file uploads and text extraction
search_engine.py    - Main search logic
app/main.py         - Web interface
cli.py              - Command-line interface
```

## Common Issues

**Database connection error?**

- Check PostgreSQL is running
- Verify password in config.py

**Import errors?**

- Run: `pip install -r requirements.txt`

**pgvector extension error?**

- Install pgvector in PostgreSQL
- Run: `CREATE EXTENSION vector;` in your database

## Learn More

This project teaches you:

- ✅ Vector databases and semantic search
- ✅ Text embeddings with AI models
- ✅ FastAPI web development
- ✅ PostgreSQL with vector extensions
- ✅ Building complete applications

Have fun exploring semantic search! 🔍✨
