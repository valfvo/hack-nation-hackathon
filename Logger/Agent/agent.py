import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from typing import List, Optional, TypedDict, Annotated
import operator

from pydantic.v1 import BaseModel, Field

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.checkpoint.memory import InMemorySaver
from wrapper import (
    LoggingAgentWrapper,
)

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
os.environ["OPENAI_API_KEY"] = api_key


@tool
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a specified PDF file."""
    print(f"--- TOOL: extract_text_from_pdf, INPUT: {pdf_path} ---")
    try:
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        return f"Error while reading the PDF: {e}"


class ParsedDetails(BaseModel):
    """Structure for the factual information extracted from a resume."""

    full_name: Optional[str] = Field(description="The full name of the candidate")
    email: Optional[str] = Field(description="The email address")
    skills: List[str] = Field(description="A list of technical skills")


@tool
def parse_resume_details(resume_text: str) -> ParsedDetails:
    """Parses raw text from a resume to extract structured details like name, email, and skills."""
    print(f"--- TOOL: parse_resume_details ---")
    prompt = f"Analyze the following resume text and extract the key information.\n\nText:\n{resume_text}"
    return prompt


@tool
def create_summary(resume_text: str) -> str:
    """Creates a concise 2-3 sentence summary of a candidate's profile."""
    print(f"--- TOOL: create_summary ---")
    prompt = f"Based on the following full resume text, write a concise and impactful 2-3 sentence summary for a recruiter.\n\nText:\n{resume_text}\n\nSummary:"
    return prompt


@tool
def send_rejection_email(
    candidate_name: str, candidate_email: str, required_skill: str
) -> str:
    """Sends a templated rejection email to a candidate when they are missing a specific required skill."""
    print(f"--- TOOL: send_rejection_email, RECIPIENT: {candidate_email} ---")
    email_subject = "Update on your application with Yubu.ai Inc."
    email_body = f"""Dear {candidate_name},

Thank you for your interest... particularly regarding experience with '{required_skill}'.

Sincerely,
The Yubu.ai Inc."""
    print(
        f"\n--- 📧 SIMULATING EMAIL SEND ---\nTo: {candidate_email}\nSubject: {email_subject}\n---\n{email_body}\n---"
    )
    return f"Rejection email successfully sent to {candidate_email}."


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]


def agent_node(state: AgentState, llm):
    """Le "cerveau" de l'agent qui décide de l'action à prendre."""
    print("--- AGENT: Thinking... ---")
    result = llm.invoke(state["messages"])
    return {"messages": [result]}


def should_continue(state: AgentState):
    """Décide si l'on continue la boucle d'action ou si le travail est terminé."""
    if not state["messages"][-1].tool_calls:
        print("--- AGENT: Work finished. ---")
        return "end"
    else:
        print("--- AGENT: Decided to use a tool. ---")
        return "continue"


def build_react_agent_graph():
    tools = [
        extract_text_from_pdf,
        parse_resume_details,
        create_summary,
        send_rejection_email,
    ]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini").bind_tools(tools)

    bound_agent_node = lambda state: agent_node(state, llm)
    tool_node = ToolNode(tools)

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", bound_agent_node)
    workflow.add_node("action", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "action", "end": END}
    )
    workflow.add_edge("action", "agent")

    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


def main():
    app = build_react_agent_graph()

    app = LoggingAgentWrapper(app)

    print("\n\n--- 🚀 SCENARIO 1: Candidate does NOT match ---")
    task1 = "Analyze the resume at 'resume/android-developer-1559034496.pdf'. The required skill for this job is 'Machine Learning'. If the candidate has this skill, provide a summary. If not, send them a rejection email."
    initial_messages = [HumanMessage(content=task1)]
    config1 = {"configurable": {"thread_id": "thread-1-rejection"}}
    result1 = app.invoke({"messages": initial_messages}, config=config1)

    print("\n--- AGENT FINAL RESPONSE (Scenario 1) ---")
    print(result1["messages"][-1].content)

    print("\n\n--- 🚀 SCENARIO 2: Candidate MATCHES ---")
    task2 = """
    Analyze the resume at 'resume/android-developer-1559034496.pdf'. The required skill is 'Java'. 
    If the candidate has this skill, provide a summary of their profile.
    """
    initial_messages = [HumanMessage(content=task2)]
    config2 = {"configurable": {"thread_id": "thread-2-success"}}
    result2 = app.invoke({"messages": initial_messages}, config=config2)

    print("\n--- AGENT FINAL RESPONSE (Scenario 2) ---")
    print(result2["messages"][-1].content)


if __name__ == "__main__":
    main()
