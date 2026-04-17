To test out the chatbot, create a .env file with the following fields

LANGSMITH_TRACING=true  <br>
LANGSMITH_ENDPOINT=https://api.smith.langchain.com <br>
LANGSMITH_API_KEY= <br>
LANGSMITH_PROJECT= <br>
GROQ_API_KEY= <br>

then create a virtual env using <br>
python -m venv .venv

activate it using <br>
source .venv/bin/activate

start the FastAPI server <br>
uvicorn api:app --reload

run the chatbot through terminal using <br>
python3 chatbot.py