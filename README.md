To test out the chatbot, create a .env file with the following fields

LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=
GROQ_API_KEY=

then create a virtual env using 
python -m venv .venv

activate it using
source .venv/bin/activate

run the chatbot through terminal using
python3 chatbot.py