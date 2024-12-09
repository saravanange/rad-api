# rad-api
1.Create a virtual environment:
python -m venv venv
2.Activate the virtual environment:
.\venv\Scripts\activate
mac:
source venv/bin/activate

3.Install the required dependencies:
pip install -r requirements.txt
4.Set up the .env file with the necessary environment variables (like MONGODB_URI, OPENAI_API_KEY).

5.Run the FastAPI application:
uvicorn app.main:app --reload

