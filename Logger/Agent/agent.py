import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from typing import List, Optional

from pydantic.v1 import BaseModel, Field

from langgraph.graph import StateGraph, END, START

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


class GraphState(BaseModel):
    """The state that flows through our modular graph."""

    pdf_path: str
    raw_text: Optional[str] = None
    parsed_details: Optional[ParsedDetails] = None
    concise_summary: Optional[str] = None
    error: Optional[str] = None
    # Parameter that can be changed by the user to make the tool succeed/fail
    required_skill: Optional[str] = None


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

        # --- VALIDATION LOGIC ---
        if state.required_skill:
            log_event(
                "VALIDATION",
                {"message": f"Checking for required skill: '{state.required_skill}'"},
            )
            parsed_skills_lower = [
                skill.lower() for skill in state.parsed_details.skills
            ]
            if state.required_skill.lower() not in parsed_skills_lower:
                # Generating the controlled error
                error_message = f"Validation Failed: The required skill '{state.required_skill}' was not found in the resume."
                state.error = error_message
                log_event(
                    "TOOL_FAILURE",
                    {
                        "name": "parse_resume_details_with_validation",
                        "reason": error_message,
                    },
                )
                return state
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


# ==============================================================================
# SECTION 4: BUILDING THE GRAPH WITH CONDITIONAL LOGIC
# ==============================================================================


# Decision node
def decide_after_parsing(state: GraphState) -> str:
    """Decides the next step after parsing based on the presence of an error."""
    if state.error:
        log_event("ROUTING", {"decision": "Error detected. Ending process."})
        return "end_process"
    else:
        log_event("ROUTING", {"decision": "No error. Proceeding to summary."})
        return "continue_to_summary"


def build_graph_with_replay_logic():
    workflow = StateGraph(GraphState)
    workflow.add_node("extract_text", extract_text_from_pdf)
    workflow.add_node("parse_and_validate", parse_resume_details_with_validation)
    workflow.add_node("create_summary", create_concise_summary)
    workflow.add_edge(START, "extract_text")
    workflow.add_edge("extract_text", "parse_and_validate")

    workflow.add_conditional_edges(
        "parse_and_validate",  # The node from which the decision is made
        decide_after_parsing,  # The function that makes the decision
        {
            # "decision_name": "next_node_name"
            "continue_to_summary": "create_summary",
            "end_process": END,
        },
    )
    workflow.add_edge("create_summary", END)

    return workflow.compile()


# ==============================================================================
# SECTION 5: MAIN EXECUTION DEMONSTRATING THE FAIL & REPLAY SCENARIO
# ==============================================================================


def main():
    pdf_file = "android-developer-1559034496.pdf"  # Make sure this file is in resume/
    pdf_path = os.path.join("resume/", pdf_file)

    app = build_graph_with_replay_logic()

    # --- STEP 1: Initial run that is designed to FAIL ---
    print("--- 🚀 Initial Run: Testing validation failure ---")
    initial_state_to_fail = {"pdf_path": pdf_path, "required_skill": "Machine Learning"}
    log_event("TASK_START", {"cv": pdf_file, "params": initial_state_to_fail})

    failed_state = app.invoke(initial_state_to_fail)

    if failed_state.get("error"):
        print(f"\n❌ RUN FAILED! The UI would display this error for replay:")
        print(f"   Error: {failed_state['error']}")
    else:
        print(
            "✅ The run succeeded unexpectedly. Try with a different 'required_skill'."
        )
        return

    # --- STEP 2: REPLAY - The user changes the parameter ---
    print("\n--- 🔄 Replay Scenario: The user corrects the parameter ---")
    print("   The user sees the error and decides to no longer require the skill.")

    # Prepare for replay. We start from the failed state
    state_for_replay = failed_state.copy()

    # The user changes the parameter via the UI
    state_for_replay["required_skill"] = None  # We remove the requirement

    # And we must reset the error so the graph re-evaluates the node.
    state_for_replay["error"] = None
    log_event("REPLAY_START", {"cv": pdf_file, "new_params": {"required_skill": None}})

    # We re-invoke the graph with the corrected state.
    successful_state = app.invoke(state_for_replay)

    if successful_state.get("error"):
        print(f"\n❌ REPLAY FAILED! Error: {successful_state['error']}")
    else:
        print("\n✅ REPLAY SUCCEEDED!")
        print("   The tool worked and the graph continued to the end.")

        # Assemble the final result
        final_result = successful_state["parsed_details"].dict()
        final_result["concise_summary"] = successful_state["concise_summary"]

        print("\n--- Final Result ---")
        print(json.dumps(final_result, indent=2))

        # --- Save the successful result to a JSON file ---
        output_filename = "result.json"
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(final_result, f, ensure_ascii=False, indent=4)

        print(f"\n✅ Final result saved to '{output_filename}'")


if __name__ == "__main__":
    main()
