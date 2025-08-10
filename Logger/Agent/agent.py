import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from typing import List, Optional

from pydantic.v1 import BaseModel, Field

from langgraph.graph import StateGraph, END, START
from wrapper import LoggingAgentWrapper
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
os.environ["OPENAI_API_KEY"] = api_key

# ==============================================================================
# SECTION 1: DATA STRUCTURES AND GRAPH STATE DEFINITION
# ==============================================================================


class Education(BaseModel):
    degree: Optional[str] = Field(description="The name of the degree")
    school: Optional[str] = Field(description="The name of the school")
    year: Optional[str] = Field(description="The year of graduation")


class Experience(BaseModel):
    role: Optional[str] = Field(description="The job title")
    company: Optional[str] = Field(description="The name of the company")
    duration: Optional[str] = Field(description="The dates of the experience")
    description: Optional[str] = Field(description="A summary of responsibilities")


class ParsedDetails(BaseModel):
    """Structure for the factual information extracted from a resume."""

    full_name: Optional[str] = Field(description="The full name of the candidate")
    email: Optional[str] = Field(description="The email address")
    phone_number: Optional[str] = Field(description="The phone number")
    skills: List[str] = Field(description="A list of technical skills")
    education: List[Education] = Field(description="A list of academic backgrounds")
    work_experience: List[Experience] = Field(
        description="A list of professional experiences"
    )


# Added a field to store the email content
class GraphState(BaseModel):
    """The state that flows through our modular graph."""

    pdf_path: str
    raw_text: Optional[str] = None
    parsed_details: Optional[ParsedDetails] = None
    concise_summary: Optional[str] = None
    error: Optional[str] = None
    required_skill: Optional[str] = None
    # Field to store the outcome of the rejection task
    rejection_email_content: Optional[str] = None


# ==============================================================================
# SECTION 2: PLACEHOLDER FUNCTIONS
# ==============================================================================


def log_event(event_type: str, data: dict):
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {event_type} {data}")


# ==============================================================================
# SECTION 3: DEFINITION OF GRANULAR NODES (THE AGENT'S "TOOLS")
# ==============================================================================


def extract_text_from_pdf(state: GraphState) -> GraphState:
    """Tool: Extracts text. Skips if text is already present (for reruns)."""
    if state.raw_text:
        return state
    log_event(
        "TOOL_EXECUTE", {"name": "extract_text_from_pdf", "input": state.pdf_path}
    )
    try:
        reader = PdfReader(state.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        state.raw_text = text
    except Exception as e:
        state.error = f"Error while reading the PDF: {e}"
    return state


def parse_resume_details_with_validation(state: GraphState) -> GraphState:
    """Tool: Parses details and performs a validation check."""
    if state.error or not state.raw_text:
        return state
    log_event(
        "TOOL_EXECUTE",
        {"name": "parse_resume_details_with_validation", "input": "Raw resume text"},
    )
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        structured_llm = llm.with_structured_output(ParsedDetails)
        prompt = f"Analyze the following resume text and extract the information in a structured way.\n\nText:\n{state.raw_text}"
        response = structured_llm.invoke(prompt)
        state.parsed_details = response

        if state.required_skill:
            log_event(
                "VALIDATION",
                {"message": f"Checking for required skill: '{state.required_skill}'"},
            )
            parsed_skills_lower = [skill.lower() for skill in response.skills or []]
            if state.required_skill.lower() not in parsed_skills_lower:
                error_message = f"Validation Failed: The required skill '{state.required_skill}' was not found in the resume."
                state.error = error_message
                log_event(
                    "TOOL_FAILURE",
                    {
                        "name": "parse_resume_details_with_validation",
                        "reason": error_message,
                    },
                )
    except Exception as e:
        state.error = f"Technical Error during details parsing: {e}"
    return state


def create_concise_summary(state: GraphState) -> GraphState:
    """Tool: Creates a summary. Only runs if parsing was successful."""
    log_event(
        "TOOL_EXECUTE",
        {"name": "summarize_profile", "input": "Raw text and parsed details"},
    )
    try:
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")
        parsing_context = state.parsed_details.json(indent=2)
        prompt = f"Based on the resume text and structured details, write a concise 2-3 sentence summary.\nDetails: {parsing_context}\nText: {state.raw_text}\nSummary:"
        response = llm.invoke(prompt)
        state.concise_summary = response.content
    except Exception as e:
        state.error = f"Error during summary creation: {e}"
    return state


def send_rejection_email(state: GraphState) -> GraphState:
    """Tool: Composes and 'sends' a rejection email if validation failed."""
    if not state.error or not state.parsed_details:
        return state

    candidate_name = state.parsed_details.full_name or "Candidate"
    candidate_email = state.parsed_details.email
    required_skill = state.required_skill or "a specific skill"

    log_event(
        "TOOL_EXECUTE", {"name": "send_rejection_email", "recipient": candidate_email}
    )

    email_subject = "Update on your application with Yubu.ai Inc."
    email_body = f"""Dear {candidate_name},

Thank you for your interest in a position at Yubu.ai Inc. and for taking the time to submit your application.

We received a high volume of qualified applicants. After careful review, we found that while your background is impressive, it does not fully align with the specific requirements for this role, particularly regarding experience with '{required_skill}'.

We will keep your resume on file for any future openings that may be a better match for your skills and experience.

We wish you the best of luck in your job search.

Sincerely,
The Yubu.ai Inc."""

    print("\n--- 📧 SIMULATING EMAIL SEND ---")
    print(f"To: {candidate_email}\nSubject: {email_subject}\n---\n{email_body}\n---")

    state.rejection_email_content = email_body
    return state


# ==============================================================================
# SECTION 4: BUILDING THE GRAPH WITH CONDITIONAL LOGIC
# ==============================================================================


# The decision logic is now more specific
def decide_after_parsing(state: GraphState) -> str:
    """Decides the next step after parsing based on the outcome."""
    if state.error:
        # Check if it's a validation error and if we have enough info to send an email
        if "Validation Failed" in state.error and state.parsed_details.email:
            log_event(
                "ROUTING",
                {"decision": "Validation failed. Routing to rejection email task."},
            )
            return "send_rejection"
        else:
            log_event(
                "ROUTING",
                {
                    "decision": f"A technical error occurred: {state.error}. Ending process."
                },
            )
            return "technical_error"
    else:
        log_event("ROUTING", {"decision": "No error. Proceeding to summary."})
        return "continue_to_summary"


def build_graph_with_failure_task():
    workflow = StateGraph(GraphState)
    workflow.add_node("extract_text", extract_text_from_pdf)
    workflow.add_node("parse_and_validate", parse_resume_details_with_validation)
    workflow.add_node("create_summary", create_concise_summary)
    # Added the email node
    workflow.add_node("send_rejection_email", send_rejection_email)

    workflow.add_edge(START, "extract_text")
    workflow.add_edge("extract_text", "parse_and_validate")

    # The routing table now includes the new path
    workflow.add_conditional_edges(
        "parse_and_validate",
        decide_after_parsing,
        {
            "continue_to_summary": "create_summary",
            "send_rejection": "send_rejection_email",
            "technical_error": END,
        },
    )
    workflow.add_edge("create_summary", END)
    # The rejection path also leads to the end
    workflow.add_edge("send_rejection_email", END)
    checkpointer = InMemorySaver()
    return workflow.compile(checkpointer=checkpointer)


# ==============================================================================
# SECTION 5: MAIN EXECUTION DEMONSTRATING THE FAIL & REPLAY SCENARIO
# ==============================================================================


def main():
    pdf_file = "android-developer-1559034496.pdf"
    pdf_path = os.path.join("resume/", pdf_file)
    app = build_graph_with_failure_task()
    app = LoggingAgentWrapper(app)

    # --- STEP 1: Initial run that is designed to FAIL ---
    print("--- 🚀 Initial Run: Testing validation failure ---")
    initial_state_to_fail = {"pdf_path": pdf_path, "required_skill": "Machine Learning"}
    log_event("TASK_START", {"cv": pdf_file, "params": initial_state_to_fail})
    config = {"configurable": {"thread_id": "1"}}
    failed_state = app.invoke(initial_state_to_fail)

    # Check the outcome of the failure
    if failed_state.get("error"):
        print(f"\n❌ RUN FAILED (as expected)! A follow-up task was triggered.")
        print(f"   Reason: {failed_state['error']}")
        if failed_state.get("rejection_email_content"):
            print("   Outcome: A rejection email was composed and 'sent'.")
    else:
        print("✅ The run succeeded unexpectedly.")
        return

    # --- STEP 2: REPLAY - The user decides to approve the candidate instead ---
    print("\n--- 🔄 Replay Scenario: The user overrides the failure ---")
    print(
        "   The user sees the validation failure but decides to proceed anyway by removing the requirement."
    )

    state_for_replay = failed_state.copy()
    state_for_replay["required_skill"] = None
    state_for_replay["error"] = None
    log_event("REPLAY_START", {"cv": pdf_file, "new_params": {"required_skill": None}})
    successful_state = app.invoke(state_for_replay)

    if successful_state.get("error"):
        print(f"\n❌ REPLAY FAILED! Error: {successful_state['error']}")
    else:
        print("\n✅ REPLAY SUCCEEDED!")
        print("   The graph bypassed the failure and continued to the end.")
        final_result = successful_state["parsed_details"].dict()
        final_result["concise_summary"] = successful_state.get(
            "concise_summary", "Summary could not be generated."
        )

        print("\n--- Final Result ---")
        print(json.dumps(final_result, indent=2))

        output_filename = "result.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)
        print(f"\n✅ Final result saved to '{output_filename}'")


if __name__ == "__main__":
    main()
