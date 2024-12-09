from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb.chat_message_histories import MongoDBChatMessageHistory
from langchain_mongodb.cache import MongoDBAtlasSemanticCache
from langchain_core.globals import set_llm_cache
from fastapi import HTTPException
from datasets import load_dataset

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_NAME = os.getenv("DATABASE_NAME", "langchain_chatbot")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "data")
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# MongoDB client
client = MongoClient(MONGODB_URI, appname="rad.openai.project")
collection = client[DB_NAME][COLLECTION_NAME]

# Function to check MongoDB connection
async def check_mongo_connection(mongo_uri: str):
    try:
        client = MongoClient(mongo_uri)
        client.admin.command('ping')  # Ping MongoDB server
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MongoDB connection failed: {e}")

# Initialize embeddings
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-ada-002")

# Initialize vector store
vector_store = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=MONGODB_URI,
    namespace=f"{DB_NAME}.{COLLECTION_NAME}",
    embedding=embedding,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    text_key="fullplot"
)

def ingest_data(data: pd.DataFrame):
    """Ingests data into MongoDB."""
    records = data.to_dict('records')
    collection.delete_many({})  # Clear existing data
    collection.insert_many(records)  # Insert new data
    return "Data ingestion successful."

'''def query_vector_store(query: str):
    """Queries the vector store."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    results = retriever.invoke(query)  # Use `invoke()` for querying
    return results
'''
def query_vector_store(query: str):
    """Queries the vector store."""
    try:
        # Get retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        print(f"Retriever initialized for query: {query}")

        # Retrieve results
        results = retriever.invoke(query)
        print(f"Retrieved results: {results}")

        # Return results
        return results

    except Exception as e:
        print(f"Error during query_vector_store: {e}")
        raise HTTPException(status_code=500, detail=f"Error during query: {str(e)}")

def create_rag_chain():
    """Creates a simple RAG chain."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    template = "Answer the question based only on the following context:\n{context}\n\nQuestion: {question}\n"
    prompt = ChatPromptTemplate.from_template(template)
    parse_output = StrOutputParser()
    return retriever | prompt | model | parse_output

# Semantic cache setup
def setup_semantic_cache():
    set_llm_cache(MongoDBAtlasSemanticCache(
        connection_string=MONGODB_URI,
        embedding=embedding,
        collection_name="semantic_cache",
        database_name=DB_NAME,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    ))

def import_embedded_movies_dataset():
    """Imports MongoDB's embedded_movies dataset from Hugging Face."""
    data = load_dataset("MongoDB/embedded_movies")
    df = pd.DataFrame(data["train"])
    ingest_data(df)  # Ingest data into MongoDB
    return "embedded_movies dataset imported successfully."

def analyze_data():
    """Performs analysis on the dataset."""
    data = load_dataset("MongoDB/embedded_movies")
    df = pd.DataFrame(data["train"])
    
    df_cleaned = df[df["fullplot"].notna()]  # Remove rows with None in 'fullplot'
    analysis_result = {
        "total_movies": len(df_cleaned),
        "missing_fullplot": df["fullplot"].isna().sum(),
        "columns": df.columns.tolist()
    }
    
    return analysis_result

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def create_rag_chain(prompt: str, query: str):
    """Creates a RAG chain and returns the response."""
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    # Create the prompt template with the passed prompt
    chat_prompt = ChatPromptTemplate.from_template(prompt)
    
    # Create the model with OpenAI API
    model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

    # Parse the output from the model
    parse_output = StrOutputParser()

    # Define the RAG chain
    rag_chain = retriever | chat_prompt | model | parse_output

    # Get the response using the provided query
    response = rag_chain.invoke({"query": query})
    
    return response

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough

def create_rag_chain(prompt: str, query: str):
    try:
        logging.info(f"Prompt: {prompt}")
        logging.info(f"Query: {query}")

        # Create retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

        # Fetch documents
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        logging.info(f"Retrieved Context: {context}")

        # Create ChatPromptTemplate
        chat_prompt = ChatPromptTemplate.from_template(prompt)

        # Initialize ChatOpenAI model
        model = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

        # Parse output
        parse_output = StrOutputParser()

        # Define RAG chain
        rag_chain = chat_prompt | model | parse_output

        # Run chain
        response = rag_chain.invoke({"context": context, "question": query})
        logging.info(f"RAG Chain Response: {response}")

        return response
    except Exception as e:
        logging.error(f"Error in create_rag_chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


def naive_rag_chain(question: str, retriever, openai_api_key: str):
    """Creates a naive RAG chain and returns the response."""
    try:
        # Retrieve documents and format them into a string
        retriever_results = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in retriever_results])  # Join the content into a string
        
        # Define the template as a string
        template = """Answer the question based only on the following context: {context}
        Question: {question}"""
        
        # Create the prompt template
        chat_prompt = ChatPromptTemplate.from_template(template)
        
        # Initialize the model with the OpenAI API key
        model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        
        # Parse the output with the appropriate parser
        parse_output = StrOutputParser()
        
        # Define the RAG chain
        rag_chain = chat_prompt | model | parse_output
        
        # Return the response from the chain
        response = rag_chain.invoke({"context": context, "question": question})
        
        return response
    except Exception as e:
        logging.error(f"Error in naive_rag_chain: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

