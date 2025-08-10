import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from typing import List, Optional, TypedDict, Annotated
import operator

from pydantic.v1 import BaseModel, Field

# Imports for the agent architecture
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
# SECTION 1: TRANSFORMING FUNCTIONS INTO DISCOVERABLE TOOLS
# Docstrings are CRUCIAL here: the agent reads them to know which tool to use.
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
def create_summary(resume_text: str) -> str:
    """Creates a concise 2-3 sentence summary of a candidate's profile based on their full resume text."""
    print(f"--- TOOL: create_summary ---")
    llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
    prompt = f"Based on the following full resume text, write a concise and impactful 2-3 sentence summary for a recruiter.\n\nText:\n{resume_text}\n\nSummary:"
    response = llm.invoke(prompt)
    return response.content


@tool
def send_rejection_email(
    candidate_name: str, candidate_email: str, required_skill: str
) -> str:
    """Sends a templated rejection email to a candidate when they are missing a specific required skill."""
    print(f"--- TOOL: send_rejection_email, RECIPIENT: {candidate_email} ---")
    email_subject = "Update on your application with Yubu.ai Inc."
    email_body = f"""Dear {candidate_name},

Thank you for your interest in a position at Yubu.ai Inc. and for taking the time to submit your application.

We received a high volume of qualified applicants. After careful review, we found that while your background is impressive, it does not fully align with the specific requirements for this role, particularly regarding experience with '{required_skill}'.

We will keep your resume on file for any future openings that may be a better match for your skills and experience.

We wish you the best of luck in your job search.

Sincerely,
The Yubu.ai Inc."""
    print(
        f"\n--- 📧 SIMULATING EMAIL SEND ---\nTo: {candidate_email}\nSubject: {email_subject}\n---\n{email_body}\n---"
    )
    return f"Rejection email successfully sent to {candidate_email}."


# === NEW TOOL HERE ===
@tool
def save_candidate_profile_as_json(
    filename: str, parsed_details: dict, summary: str
) -> str:
    """Saves the candidate's final profile, including parsed details and the summary, to a JSON file."""
    print(f"--- TOOL: save_candidate_profile_as_json, FILENAME: {filename} ---")
    try:
        final_profile = parsed_details
        final_profile["summary"] = summary

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(final_profile, f, ensure_ascii=False, indent=4)

        return f"Successfully saved the complete profile to '{filename}'."
    except Exception as e:
        return f"Error while saving the JSON file: {e}"


# ==============================================================================
# SECTION 2: DEFINING THE AGENT STATE AND GRAPH
# ==============================================================================


class AgentState(TypedDict):
    # The list of messages serves as the memory of the agent
    messages: Annotated[list, operator.add]


def agent_node(state: AgentState, llm):
    """This node is the "brain" of the agent. It decides which action to take."""
    print("--- AGENT: Thinking... ---")
    result = llm.invoke(state["messages"])
    return {"messages": [result]}


def should_continue(state: AgentState):
    """This function decides whether to continue calling tools or if the agent is finished."""
    if not state["messages"][-1].tool_calls:
        print("--- AGENT: Work finished. ---")
        return "end"
    else:
        print("--- AGENT: Decided to use a tool. ---")
        return "continue"


def build_react_agent_graph():
    # Define the list of tools the agent can choose from
    tools = [
        extract_text_from_pdf,
        parse_resume_details,
        create_summary,
        send_rejection_email,
        save_candidate_profile_as_json,
    ]
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini").bind_tools(tools)

    # Define the nodes of the graph
    bound_agent_node = lambda state: agent_node(state, llm)
    tool_node = ToolNode(tools)

    # Build the graph structure
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", bound_agent_node)
    workflow.add_node("action", tool_node)

    workflow.set_entry_point("agent")

    # Define the conditional logic for the agent loop
    workflow.add_conditional_edges(
        "agent", should_continue, {"continue": "action", "end": END}
    )

    # After a tool is executed, the flow always goes back to the agent to decide the next step
    workflow.add_edge("action", "agent")

    return workflow.compile()


# ==============================================================================
# SECTION 3: EXECUTING THE REACT AGENT
# ==============================================================================


def main():
    app = build_react_agent_graph()

    # --- SCENARIO 1: Unchanged ---
    print("\n\n--- 🚀 SCENARIO 1: Candidate does NOT match ---")
    task1 = "Analyze the resume at 'resume/android-developer-1559034496.pdf'. The required skill for this job is 'Machine Learning'. If the candidate has this skill, provide a summary. If not, send them a rejection email."
    initial_messages = [HumanMessage(content=task1)]
    result1 = app.invoke({"messages": initial_messages})
    print("\n--- AGENT FINAL RESPONSE (Scenario 1) ---")
    print(result1["messages"][-1].content)

    # --- SCENARIO 2: Task modified to include saving the result ---
    print("\n\n--- 🚀 SCENARIO 2: Candidate MATCHES and profile is saved ---")
    task2 = """
    Analyze the resume at 'resume/android-developer-1559034496.pdf'. The required skill is 'Java'. 
    If the candidate has this skill, first create a summary. 
    Then, using the parsed details and the new summary, save the complete profile to a JSON file named 'successful_candidate_profile.json'.
    """
    initial_messages = [HumanMessage(content=task2)]
    result2 = app.invoke({"messages": initial_messages})
    print("\n--- AGENT FINAL RESPONSE (Scenario 2) ---")
    print(result2["messages"][-1].content)

    # Verify that the file was created
    if os.path.exists("successful_candidate_profile.json"):
        print(
            "\n✅ Verification: 'successful_candidate_profile.json' was successfully created."
        )


if __name__ == "__main__":
    main()
