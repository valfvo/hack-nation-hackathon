import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from typing import List, Optional, TypedDict, Annotated
import operator

from pydantic.v1 import BaseModel, Field

# NOUVEAUX IMPORTS POUR L'AGENT
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
os.environ["OPENAI_API_KEY"] = api_key

# ==============================================================================
# SECTION 1: TRANSFORMER LES FONCTIONS EN OUTILS DÉCOUVERABLES
# Les docstrings sont CRUCIALES ici : l'agent les lit pour savoir quel outil utiliser.
# ==============================================================================


@tool
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a specified PDF file and returns it as a single string."""
    print(f"--- TOOL: extract_text_from_pdf, INPUT: {pdf_path} ---")
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        return f"Error while reading the PDF: {e}"


# Les modèles Pydantic sont maintenant utilisés pour structurer la sortie des outils
class ParsedDetails(BaseModel):
    """Structure for the factual information extracted from a resume."""

    full_name: Optional[str] = Field(description="The full name of the candidate")
    email: Optional[str] = Field(description="The email address")
    skills: List[str] = Field(description="A list of technical skills")


@tool
def parse_resume_details(resume_text: str) -> ParsedDetails:
    """Parses raw text from a resume to extract structured details like name, email, and skills."""
    print(f"--- TOOL: parse_resume_details ---")
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    structured_llm = llm.with_structured_output(ParsedDetails)
    prompt = f"Analyze the following resume text and extract the key information.\n\nText:\n{resume_text}"
    return structured_llm.invoke(prompt)


@tool
def create_summary(resume_text: str, parsed_details: dict) -> str:
    """Creates a concise 2-3 sentence summary of a candidate's profile based on their full resume text and already parsed details."""
    print(f"--- TOOL: create_summary ---")
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
    prompt = f"Based on the resume text and structured details, write a concise summary.\nDetails: {json.dumps(parsed_details)}\nText: {resume_text}\nSummary:"
    response = llm.invoke(prompt)
    return response.content


@tool
def send_rejection_email(
    candidate_name: str, candidate_email: str, required_skill: str
) -> str:
    """Sends a templated rejection email to a candidate when they are missing a specific required skill."""
    print(f"--- TOOL: send_rejection_email, RECIPIENT: {candidate_email} ---")
    email_subject = "Update on your application with Yubu.ai Inc."
    email_body = f"Dear {candidate_name},\n\nThank you for your interest... particularly regarding experience with '{required_skill}'.\n\nSincerely,\nThe Yubu.ai Inc."

    # Simulation
    print("\n--- 📧 SIMULATING EMAIL SEND ---")
    print(f"To: {candidate_email}\nSubject: {email_subject}\n---\n{email_body}\n---")

    return f"Rejection email successfully sent to {candidate_email}."


# ==============================================================================
# SECTION 2: DÉFINIR L'ÉTAT ET LE GRAPHE DE L'AGENT
# ==============================================================================


# L'état de l'agent est simplement la liste des messages échangés.
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


# Le "cerveau" de l'agent
def agent_node(state: AgentState, llm, tools):
    """Ce noeud décide de l'action à prendre (appeler un outil ou répondre à l'utilisateur)."""
    print("--- AGENT: Thinking... ---")
    result = llm.invoke(state["messages"])
    return {"messages": [result]}


# La logique pour décider si on continue la boucle ou si on s'arrête
def should_continue(state: AgentState):
    """Décide si l'on continue à appeler des outils ou si l'agent a fini."""
    last_message = state["messages"][-1]
    # Si le dernier message n'a pas d'appel d'outil, le travail est terminé.
    if not last_message.tool_calls:
        print("--- AGENT: Work finished. ---")
        return "end"
    else:
        print("--- AGENT: Decided to use a tool. ---")
        return "continue"


def build_react_agent_graph():
    # 1. Définir les outils et le LLM
    tools = [
        extract_text_from_pdf,
        parse_resume_details,
        create_summary,
        send_rejection_email,
    ]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini").bind_tools(tools)

    # 2. Définir les noeuds du graphe
    # Le noeud "agent" qui appelle le LLM pour décider
    bound_agent_node = lambda state: agent_node(state, llm, tools)
    # Le noeud "action" qui exécute l'outil choisi par l'agent
    tool_node = ToolNode(tools)

    # 3. Construire le graphe
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", bound_agent_node)
    workflow.add_node("action", tool_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",  # Le noeud de départ de la décision
        should_continue,  # La fonction qui décide
        {
            "continue": "action",  # Si on continue, on exécute l'outil
            "end": END,  # Si on a fini, on arrête
        },
    )
    # Après avoir exécuté un outil, on retourne toujours à l'agent pour qu'il réfléchisse à la suite
    workflow.add_edge("action", "agent")

    return workflow.compile()


# ==============================================================================
# SECTION 3: EXÉCUTER L'AGENT REACT
# ==============================================================================


def main():
    app = build_react_agent_graph()

    # --- SCÉNARIO 1 : Le candidat NE CORRESPOND PAS ---
    print("\n\n--- 🚀 SCENARIO 1: Candidate does NOT match ---")
    task1 = "Analyze the resume at 'resume/android-developer-1559034496.pdf'. The required skill for this job is 'Machine Learning'. If the candidate has this skill, provide a summary. If not, send them a rejection email."

    # On lance la tâche comme une conversation
    initial_messages = [HumanMessage(content=task1)]
    result1 = app.invoke({"messages": initial_messages})

    print("\n--- AGENT FINAL RESPONSE (Scenario 1) ---")
    print(result1["messages"][-1].content)

    # --- SCÉNARIO 2 : Le candidat CORRESPOND ---
    print("\n\n--- 🚀 SCENARIO 2: Candidate MATCHES ---")
    task2 = "Analyze the resume at 'resume/android-developer-1559034496.pdf'. The required skill is 'Java'. If the candidate has this skill, provide a summary. If not, send them a rejection email."

    initial_messages = [HumanMessage(content=task2)]
    result2 = app.invoke({"messages": initial_messages})

    print("\n--- AGENT FINAL RESPONSE (Scenario 2) ---")
    print(result2["messages"][-1].content)


if __name__ == "__main__":
    main()
