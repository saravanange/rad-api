from fastapi import FastAPI
from app.routers import router

app = FastAPI(title="Semantic Caching and Memory API")

# Include the router
app.include_router(router)

@app.get("/")
def read_root():
    """Root endpoint."""
    return {"message": "Welcome to the Semantic Caching API!"}
