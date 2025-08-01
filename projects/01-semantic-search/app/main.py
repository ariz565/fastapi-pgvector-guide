# FastAPI web application for semantic document search
# This provides a web interface and REST API for the search engine

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import os
import shutil
from pathlib import Path
from search_engine import get_search_engine
from config import UPLOAD_FOLDER, create_directories

# Function to read HTML template
def read_html_template(template_name):
    """Read HTML template file from templates directory"""
    template_path = os.path.join('templates', template_name)
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        # Fallback if template file not found
        return "<html><body><h1>Template not found</h1></body></html>"

# Initialize FastAPI app
app = FastAPI(
    title="Semantic Document Search Engine",
    description="A semantic search engine that finds documents by meaning, not just keywords",
    version="1.0.0"
)

# Create necessary directories
create_directories()

# Initialize search engine
search_engine = get_search_engine()

# Root endpoint - serves the HTML interface from template file
@app.get("/", response_class=HTMLResponse)
def read_root():
    """Serve the main web interface from HTML template"""
    html_content = read_html_template('index.html')
    return HTMLResponse(content=html_content)

# API endpoint to upload and index documents
@app.post("/upload")
async def upload_document(file: UploadFile = File(...), title: str = Form(None)):
    """Upload and index a new document"""
    try:
        # Validate file type
        if not file.filename.endswith(('.txt', '.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Unsupported file type. Please use TXT, PDF, or DOCX files.")
        
        # Create unique filename
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Handle duplicate filenames
        counter = 1
        original_path = file_path
        while os.path.exists(file_path):
            name, ext = os.path.splitext(original_path)
            file_path = f"{name}_{counter}{ext}"
            counter += 1
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Index the document
        success = search_engine.index_document(file_path, title or file.filename)
        
        if success:
            return {"message": "Document uploaded and indexed successfully", "file_path": file_path}
        else:
            # Clean up file if indexing failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail="Failed to index document")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading document: {str(e)}")

# API endpoint to search documents
@app.get("/search")
def search_documents(query: str, limit: int = 10):
    """Search for documents using semantic similarity"""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    try:
        results = search_engine.search(query, limit)
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

# API endpoint to list all documents
@app.get("/documents")
def list_all_documents():
    """Get list of all indexed documents"""
    try:
        documents = search_engine.list_documents()
        return documents
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing documents: {str(e)}")

# API endpoint to get search index statistics
@app.get("/stats")
def get_statistics():
    """Get statistics about the search index"""
    try:
        stats = search_engine.get_stats()
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

# API endpoint to delete a document
@app.delete("/documents/{document_id}")
def delete_document(document_id: int):
    """Delete a document from the search index"""
    try:
        success = search_engine.delete_document(document_id)
        
        if success:
            return {"message": "Document deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Document not found or could not be deleted")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if the application is running properly"""
    return {"status": "healthy", "message": "Semantic Search Engine is running"}

# Run the application
if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    
    print("Starting Semantic Document Search Engine...")
    print(f"Web interface: http://{API_HOST}:{API_PORT}")
    print(f"API documentation: http://{API_HOST}:{API_PORT}/docs")
    
    uvicorn.run(app, host=API_HOST, port=API_PORT)
