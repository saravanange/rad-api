from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services import (
    ingest_data, 
    query_vector_store, 
    create_rag_chain, 
    setup_semantic_cache, 
    check_mongo_connection, 
    import_embedded_movies_dataset, 
    analyze_data,
    OPENAI_API_KEY,
    vector_store  # Import vector_store
)


import pandas as pd
import pandas as pd

router = APIRouter()

# Pydantic model for ingest request
class IngestRequest(BaseModel):
    data: list

# Endpoint to ingest data into MongoDB
@router.post("/vector/ingest")
def ingest(request: IngestRequest):
    try:
        df = pd.DataFrame(request.data)
        response = ingest_data(df)
        return {"message": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to query the vector store
@router.get("/vector/query")
def vector_query(query: str):
    try:
        result = query_vector_store(query)
        return {"result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to set up semantic cache
@router.post("/semantic/setup")
def semantic_cache_setup():
    try:
        setup_semantic_cache()
        return {"message": "Semantic cache set up successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@router.get("/health")
async def health_check():
    try:
        await check_mongo_connection()
        return {"status": "MongoDB is connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB connection failed: {str(e)}")

# Endpoint to import dataset
@router.post("/dataset/import")
async def import_dataset():
    try:
        result = import_embedded_movies_dataset()
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for dataset analysis
@router.get("/dataset/analyze")
async def analyze():
    try:
        result = analyze_data()
        return {"analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Pydantic model for chat prompt
class ChatPromptRequest(BaseModel):
    query: str


class QueryRequest(BaseModel):
    query: str

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
from pydantic import BaseModel

class ChatPromptRequest(BaseModel):
    prompt: str
    query: str


# Endpoint to get chat response from the RAG chain
@router.post("/chat/prompt")
async def get_chat_response(request: ChatPromptRequest):
    try:
        # Pass both prompt and query to create_rag_chain
        response = create_rag_chain(request.prompt, request.query)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
        

class NaiveRAGRequest(BaseModel):
    question: str

from app.services import naive_rag_chain


@router.post("/chat/rag")
async def chat_rag(request: NaiveRAGRequest):
    try:
        logging.info(f"Received question: {request.question}")

        # Assuming retriever is available globally or passed as a param
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Call the naive_rag_chain function to get the response
        response = naive_rag_chain(request.question, retriever, openai_api_key=OPENAI_API_KEY)
        logging.info(f"Response: {response}")

        return {"response": response}
    except Exception as e:
        logging.error(f"Error in /chat/rag: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")