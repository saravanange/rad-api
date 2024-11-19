from fastapi import APIRouter, HTTPException
from app.services import (
    add_document_to_vector_store,
    query_vector_store,
    query_rag_chain,
    query_with_semantic_cache,
)
from pydantic import BaseModel
from typing import List

router = APIRouter()

class AddDocumentRequest(BaseModel):
    document: str

class QueryWithHistoryRequest(BaseModel):
    query: str
    chat_history: List[str]


@router.post("/vector/add")
def add_document_endpoint(request: AddDocumentRequest):
    """Add a document to the vector store."""
    try:
        add_document_to_vector_store(request.document)
        return {"message": "Document added to vector store"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/vector/query")
def query_vector_endpoint(query: str):
    """Query the vector store."""
    try:
        results = query_vector_store(query)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag/query")
def query_rag_endpoint(request: QueryWithHistoryRequest):
    """Query the RAG chain with chat history."""
    try:
        response = query_rag_chain(request.query, request.chat_history)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/semantic/query")
def query_with_cache_endpoint(request: QueryWithHistoryRequest):
    """Query with semantic caching."""
    try:
        response = query_with_semantic_cache(request.query, request.chat_history)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
