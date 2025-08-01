# Pydantic Models for Semantic Search API
# Clean data validation and serialization models

from pydantic import BaseModel, Field, validator
from typing import List, Dict, Optional, Union
from datetime import datetime

class DocumentCreate(BaseModel):
    """Model for creating new documents"""
    title: str = Field(..., min_length=1, max_length=500, description="Document title")
    content: str = Field(..., min_length=1, description="Document content/text")
    category: Optional[str] = Field(None, max_length=100, description="Document category")
    url: Optional[str] = Field(None, max_length=1000, description="Source URL")
    
    @validator('title', 'content')
    def validate_text_fields(cls, v):
        # Remove excessive whitespace and validate content
        if not v or not v.strip():
            raise ValueError('Field cannot be empty')
        return v.strip()

class DocumentResponse(BaseModel):
    """Model for document responses"""
    id: int
    title: str
    content_preview: str
    category: Optional[str]
    url: Optional[str]
    similarity_score: Optional[float] = None
    text_score: Optional[float] = None
    vector_score: Optional[float] = None
    combined_score: Optional[float] = None
    created_at: Optional[str]

class DocumentBatch(BaseModel):
    """Model for batch document creation"""
    documents: List[DocumentCreate] = Field(..., min_items=1, max_items=100)

class SearchRequest(BaseModel):
    """Model for search requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum results")
    category_filter: Optional[str] = Field(None, max_length=100, description="Filter by category")
    similarity_threshold: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity")

class HybridSearchRequest(BaseModel):
    """Model for hybrid search requests"""
    query: str = Field(..., min_length=1, max_length=1000, description="Search query")
    text_weight: Optional[float] = Field(0.3, ge=0.0, le=1.0, description="Text search weight")
    vector_weight: Optional[float] = Field(0.7, ge=0.0, le=1.0, description="Vector search weight")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum results")
    category_filter: Optional[str] = Field(None, max_length=100, description="Filter by category")
    
    @validator('vector_weight')
    def validate_weights(cls, v, values):
        # Ensure weights sum to 1.0
        text_weight = values.get('text_weight', 0.3)
        total_weight = text_weight + v
        if abs(total_weight - 1.0) > 0.01:  # Allow small floating point errors
            raise ValueError('text_weight + vector_weight must equal 1.0')
        return v

class SimilarDocumentsRequest(BaseModel):
    """Model for finding similar documents"""
    document_id: int = Field(..., gt=0, description="Reference document ID")
    limit: Optional[int] = Field(10, ge=1, le=100, description="Maximum results")
    exclude_same_category: Optional[bool] = Field(False, description="Exclude same category")

class SearchResponse(BaseModel):
    """Model for search responses"""
    query: str
    results: List[DocumentResponse]
    total_results: int
    search_time_ms: float
    search_type: Optional[str] = "semantic"
    filters: Optional[Dict] = None
    timestamp: str
    error: Optional[str] = None

class HybridSearchResponse(BaseModel):
    """Model for hybrid search responses"""
    query: str
    search_type: str = "hybrid"
    weights: Dict[str, float]
    results: List[DocumentResponse]
    total_results: int
    search_time_ms: float
    timestamp: str
    error: Optional[str] = None

class SimilarDocumentsResponse(BaseModel):
    """Model for similar documents responses"""
    reference_document: Dict
    similar_documents: List[DocumentResponse]
    total_results: int
    filters: Optional[Dict] = None
    error: Optional[str] = None

class CategoryResponse(BaseModel):
    """Model for category-based responses"""
    category: str
    documents: List[DocumentResponse]
    total_results: int

class HealthResponse(BaseModel):
    """Model for health check responses"""
    status: str
    database_connected: bool
    embedding_model_loaded: bool
    total_documents: int
    documents_with_embeddings: int
    embedding_coverage_percent: float
    uptime_seconds: Optional[float] = None
    timestamp: str

class DatabaseStats(BaseModel):
    """Model for database statistics"""
    total_documents: int
    documents_with_embeddings: int
    recent_searches_24h: int
    avg_search_time_ms: float
    embedding_coverage_percent: float
    categories: Optional[List[Dict]] = None

class AnalyticsResponse(BaseModel):
    """Model for analytics responses"""
    period_days: int
    total_searches: int
    avg_search_time_ms: float
    avg_results_count: float
    top_queries: List[Dict]
    daily_performance: List[Dict]
    generated_at: str

class ErrorResponse(BaseModel):
    """Model for error responses"""
    error: str
    detail: Optional[str] = None
    timestamp: str
    request_id: Optional[str] = None
