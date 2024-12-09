from fastapi import FastAPI, HTTPException
from app.routers import router
from app.services import check_mongo_connection
from dotenv import load_dotenv
import os

app = FastAPI(title="My FastAPI with MongoDB")

# Load environment variables
load_dotenv()

# Include the router with API endpoints
app.include_router(router)

# Startup event to check MongoDB connection
@app.on_event("startup")
async def startup_event():
    mongo_uri = os.getenv("MONGODB_URI")
    if not mongo_uri:
        raise HTTPException(status_code=500, detail="MONGO_URI is not set in environment variables")
    
    try:
        await check_mongo_connection(mongo_uri)
        print("MongoDB connection successful!")
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        raise HTTPException(status_code=500, detail=f"MongoDB connection failed: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAD OpenAI API"}
