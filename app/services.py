from pymongo import MongoClient
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv("MONGO_URI")
if not MONGODB_URI:
    raise ValueError("MongoDB URI is not set in the environment variables.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key is not set in the environment variables.")

client = MongoClient(MONGODB_URI, appname="devrel.content.python")
DB_NAME = "langchain_chatbot"
COLLECTION_NAME = "data"
collection = client[DB_NAME][COLLECTION_NAME]

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

vector_db = MongoDBAtlasVectorSearch(
    collection=collection,
    embedding=embedding,
    index_name="vector_index"
)


def add_document_to_vector_store(document: str):
    # Convert the input into a Document object
    doc = Document(page_content=document, metadata={})
    vector_db.add_documents([doc])


def query_vector_store(query: str):
    results = vector_db.similarity_search(query)
    return results


def create_rag_chain():
    retriever = vector_db.as_retriever()
    rag_chain = ConversationalRetrievalChain.from_chain_type(
        retriever=retriever,
        llm=embedding
    )
    return rag_chain


def query_rag_chain(query: str, chat_history: list):
    rag_chain = create_rag_chain()
    response = rag_chain.run(input_documents=chat_history, query=query)
    return response


def query_with_semantic_cache(query: str, cache: dict):
    if query in cache:
        return cache[query]
    results = query_vector_store(query)
    cache[query] = results
    return results
