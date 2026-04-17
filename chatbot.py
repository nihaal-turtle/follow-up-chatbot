import json
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, HumanMessage
from langgraph.graph import START, END, StateGraph
from dotenv import load_dotenv
from pydantic import BaseModel
import requests
from typing import Optional, TypedDict

load_dotenv()

model = init_chat_model(
    model_provider='groq',
    model='qwen/qwen3-32b',
    temperature=0
)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3

class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    content: str
    score: float

class State(TypedDict):
    user_query: str
    formatted_rag: str
    rag_chunks : list[RetrievedChunk]
    is_ambiguous: bool
    final_response: str
    follow_up: list[str]

@tool
def query_rag(user_query:str) -> dict:
    """Tool to query RAG service for relevant, up to date information about health insurance"""
    url = f"http://127.0.0.1:8000/rag/query"
    response = requests.get(url,params={"query":user_query,"top_k":3})
    data = response.json()
    chunks = data['retrieved_chunks']
    return {"chunks":chunks}


def check_ambiguous(state:State) -> dict:
    rag_text = format_chunks_for_llm(state["rag_chunks"])
    response = model.invoke([
        SystemMessage(content="""
            You are an ambiguity detector for a health insurance chatbot.
            Given the user query and the RAG answer, return ONLY one word:
            Ambiguous or Clear
 
            Flag as ambiguous ONLY if the answer would be meaningfully different
            with more context. Do NOT flag clear factual questions.
 
            Ambiguous:     "how do I pick a plan?", "is my doctor covered?"
            Clear : "what is a deductible?", "when is open enrollment?"
        """),
        HumanMessage(content=f"""
            User query: {state["user_query"]} \n\n
            RAG chunks: {rag_text}
        """)
    ])
    response = response.content.strip().lower()
    return {
        "is_ambiguous": response == "ambiguous",
        "formatted_rag": rag_text    
    }


def clarifying_question(state:State):
    return {"final_response": "Could you give me more context? For example: are you comparing plans, checking coverage for a specific service, or trying to understand your costs?"}


def format_chunks_for_llm(chunks) -> str:
    formatted = ""
    for i,chunk in enumerate(chunks,1):
        formatted += f"""
    Chunk {i}:
        Source : {chunk.get('source','N/A')}
        Content : {chunk.get('content','N/A')}
        Relevance Score : {chunk.get('score','N/A')}
    """
    return formatted

def retrieve_rag_data(state:State):
    response = query_rag.invoke({"user_query":state["user_query"]})
    return {"rag_chunks": response["chunks"]}


def answer(state:State):
    # state['formatted_rag'] = format_chunks_for_llm(state['rag_chunks'])
    messages = [
        SystemMessage("You are a helpful assistant designed to help people understand about health insurance policies and various terms related to health insurance. Use the user query and RAG answer provided to generate a informative,concise answer to the user's query"),
        HumanMessage(content=f"""
            User query: {state["user_query"]}
            RAG answer: {state["formatted_rag"]}
        """)
    ]
    response = model.invoke(messages)
    final_response = response.content.strip()
    return {"final_response":final_response}


def gen_followup(state:State):
    messages = [
        SystemMessage("""You are a helpful assistant chatbot designed to generate 2-3 thought-provoking follow up questions to make the user's experience more interactive. 
                      Ensure that the follow-up questions are related to the user's query and the response from RAG service. Follow-up questions can include the logical next step, 
                      common issues users might face, etc. 
                      Return ONLY valid JSON in this format:
        {"follow_up_questions": ["question 1", "question 2", "question 3"]}"""),
        HumanMessage(content=f"""
            User query: {state["user_query"]}
            RAG Chunks: {state["formatted_rag"]}
            Final response : {state['final_response']}
        """)
    ]
    response = model.invoke(messages)
    try:
        content = response.content.strip()
        json_start = content.find('{')
        json_end = content.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = content[json_start:json_end]
            parsed = json.loads(json_str)
            return {"follow_up": parsed['follow_up_questions']}
    except Exception as e:
        print(f"Error: {e}")
    return {"follow_up": []}



def route(state: State) -> str:
    return "clarifying_question" if state["is_ambiguous"] else "answer"


def input_node(state: State) -> dict:
    """Node that accepts user input"""
    if state["is_ambiguous"]:
        print(state["final_response"])
        user_query = input("\nUser: ")
        return {
            "user_query":user_query,
            "is_ambiguous":False
        }
    elif state["user_query"] == "":
        user_query = input("User: ")  # Reads from command line
        return {"user_query": user_query}
    else:
        return {"user_query":state["user_query"]}


agent_builder = StateGraph(State)
agent_builder.add_node("check_ambiguous",check_ambiguous)
agent_builder.add_node("clarifying_question",clarifying_question)
agent_builder.add_node("answer",answer)
agent_builder.add_node("input_node",input_node)
agent_builder.add_node("retrieve_rag_data",retrieve_rag_data)
agent_builder.add_node("gen_followup",gen_followup)
agent_builder.add_edge(START, "input_node")
agent_builder.add_edge("input_node", "retrieve_rag_data")  # Retrieve first
agent_builder.add_edge("retrieve_rag_data", "check_ambiguous")  # Then check
agent_builder.add_conditional_edges("check_ambiguous", route, ["clarifying_question", "answer"])
agent_builder.add_edge("clarifying_question", "input_node")
agent_builder.add_edge("answer", "gen_followup")
agent_builder.add_edge("gen_followup", END)
agent = agent_builder.compile()

response = agent.invoke({
    "user_query":"",
    "formatted_rag":"",
    "rag_chunks":[],
    "is_ambiguous":False,
    "final_response":"",
    "follow_up":[]
})

print(f"RAG Text:{response['formatted_rag']}")
print(f"Final response:{response['final_response']}")
print(f"Follow up questions \n{"\n".join(response['follow_up'])}")
