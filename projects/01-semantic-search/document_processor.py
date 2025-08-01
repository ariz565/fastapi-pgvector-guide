# Document processing and text extraction module
# This module handles different file formats and prepares text for embedding

import os
import re
from pathlib import Path

# We'll use simple text processing first, then can add PDF/DOCX support later
class DocumentProcessor:
    """Handles document upload, processing, and text extraction"""
    
    def __init__(self):
        """Initialize document processor"""
        print("Document processor initialized")
    
    def extract_text_from_file(self, file_path):
        """Extract text content from different file types"""
        file_extension = Path(file_path).suffix.lower()
        
        try:
            if file_extension == '.txt':
                return self.extract_from_txt(file_path)
            elif file_extension == '.pdf':
                return self.extract_from_pdf(file_path)
            elif file_extension == '.docx':
                return self.extract_from_docx(file_path)
            else:
                print(f"Unsupported file type: {file_extension}")
                return None
                
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            return None
    
    def extract_from_txt(self, file_path):
        """Extract text from plain text files"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            print(f"Extracted {len(content)} characters from TXT file")
            return content
            
        except Exception as e:
            print(f"Error reading TXT file: {e}")
            return None
    
    def extract_from_pdf(self, file_path):
        """Extract text from PDF files (placeholder for now)"""
        # For beginners, we'll start with a simple placeholder
        # You can add PyPDF2 or pdfplumber later
        print("PDF extraction not implemented yet. Please use TXT files for now.")
        return None
    
    def extract_from_docx(self, file_path):
        """Extract text from Word documents (placeholder for now)"""
        # For beginners, we'll start with a simple placeholder
        # You can add python-docx later
        print("DOCX extraction not implemented yet. Please use TXT files for now.")
        return None
    
    def clean_text(self, text):
        """Clean and preprocess text content"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def split_text_into_chunks(self, text, chunk_size=500, overlap=50):
        """Split long text into smaller chunks for better embedding"""
        if not text:
            return []
        
        # Clean the text first
        clean_text = self.clean_text(text)
        
        # Split into words
        words = clean_text.split()
        
        if len(words) <= chunk_size:
            # Text is short enough, return as single chunk
            return [clean_text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            # Get chunk of words
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            
            # Avoid infinite loop if overlap is too large
            if start >= end:
                break
        
        print(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def validate_file(self, file_path):
        """Validate if file exists and is readable"""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"File does not exist: {file_path}")
                return False
            
            # Check if file is readable
            if not os.access(file_path, os.R_OK):
                print(f"File is not readable: {file_path}")
                return False
            
            # Check file size (avoid very large files)
            file_size = os.path.getsize(file_path)
            max_size = 10 * 1024 * 1024  # 10MB
            
            if file_size > max_size:
                print(f"File too large: {file_size} bytes (max: {max_size})")
                return False
            
            print(f"File validation passed: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error validating file: {e}")
            return False
    
    def process_document(self, file_path, title=None):
        """Complete document processing pipeline"""
        try:
            # Validate file first
            if not self.validate_file(file_path):
                return None
            
            # Extract text content
            content = self.extract_text_from_file(file_path)
            if not content:
                return None
            
            # Generate title if not provided
            if not title:
                title = Path(file_path).stem  # Filename without extension
            
            # Split into chunks
            chunks = self.split_text_into_chunks(content)
            
            # Prepare document info
            document_info = {
                'title': title,
                'content': content,
                'file_path': file_path,
                'file_type': Path(file_path).suffix.lower(),
                'chunks': chunks,
                'word_count': len(content.split()),
                'char_count': len(content),
                'chunk_count': len(chunks)
            }
            
            print(f"Document processed successfully: {title}")
            print(f"  - Word count: {document_info['word_count']}")
            print(f"  - Character count: {document_info['char_count']}")
            print(f"  - Chunks created: {document_info['chunk_count']}")
            
            return document_info
            
        except Exception as e:
            print(f"Error processing document: {e}")
            return None

# Helper function to create document processor
def get_document_processor():
    """Create and return a document processor instance"""
    return DocumentProcessor()

# Test function for document processing
def test_document_processing():
    """Test document processing with a sample file"""
    print("Testing document processing...")
    
    # Create a sample text file for testing
    test_file_path = "data/test_document.txt"
    test_content = """
    This is a sample document for testing our semantic search system.
    
    The document contains multiple paragraphs with different topics.
    We will use this to test text extraction, cleaning, and chunking.
    
    Artificial intelligence and machine learning are transforming many industries.
    Natural language processing enables computers to understand human language.
    Vector databases allow us to search by meaning rather than exact keywords.
    
    This semantic search system will help users find relevant information
    even when they don't use the exact words that appear in the documents.
    """
    
    # Create test file
    os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
    with open(test_file_path, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    # Process the document
    processor = get_document_processor()
    document_info = processor.process_document(test_file_path, "Test Document")
    
    if document_info:
        print("Document processing test successful!")
        print(f"Generated {len(document_info['chunks'])} text chunks")
    else:
        print("Document processing test failed!")

if __name__ == "__main__":
    test_document_processing()
