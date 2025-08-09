import os
import json
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_openai import ChatOpenAI
from typing import List, Optional

from pydantic.v1 import BaseModel, Field

from langgraph.graph import StateGraph, END

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found. Please check your .env file.")
os.environ["OPENAI_API_KEY"] = api_key

# ==============================================================================
# SECTION 1: DATA STRUCTURES AND GRAPH STATE DEFINITION
# ==============================================================================


# Granular data structures for parsing
class Education(BaseModel):
    degree: Optional[str] = Field(
        description="The name of the degree, e.g., Master in Computer Science"
    )
    school: Optional[str] = Field(description="The name of the school or university")
    year: Optional[str] = Field(description="The year of graduation")


class Experience(BaseModel):
    role: Optional[str] = Field(description="The job title, e.g., Web Developer")
    company: Optional[str] = Field(description="The name of the company")
    duration: Optional[str] = Field(
        description="The dates or duration of the experience"
    )
    description: Optional[str] = Field(
        description="A brief summary of tasks and responsibilities"
    )


# Structure for parsing only (our first intelligent tool's output)
class ParsedDetails(BaseModel):
    """Structure for the factual information extracted from a resume."""

    full_name: Optional[str] = Field(description="The full name of the candidate")
    email: Optional[str] = Field(description="The email address of the candidate")
    phone_number: Optional[str] = Field(description="The phone number of the candidate")
    skills: List[str] = Field(description="A list of technical or language skills")
    education: List[Education] = Field(description="A list of academic backgrounds")
    work_experience: List[Experience] = Field(
        description="A list of professional experiences"
    )


# Granular graph state to track each step
class GraphState(BaseModel):
    """The state that flows through our modular graph."""

    pdf_path: str
    raw_text: Optional[str] = None
    parsed_details: Optional[ParsedDetails] = None
    concise_summary: Optional[str] = None
    error: Optional[str] = None


# ==============================================================================
# SECTION 2: PLACEHOLDER FUNCTIONS (UNCHANGED)
# ==============================================================================


def log_event(event_type: str, data: dict):
    """Placeholder for Jules's logger/wrapper."""
    from datetime import datetime

    timestamp = datetime.now().isoformat()
    print(f"[{timestamp}] {event_type} {data}")


# ==============================================================================
# SECTION 3: DEFINITION OF GRANULAR NODES (THE AGENT'S "TOOLS")
# ==============================================================================


def extract_text_from_pdf(state: GraphState) -> GraphState:
    """Node 1: Opens the PDF and extracts its raw text."""
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


def parse_resume_details(state: GraphState) -> GraphState:
    """Node 2 (Tool 1): Parses the structured details from the resume."""
    if state.error or not state.raw_text:
        return state

    log_event(
        "TOOL_EXECUTE", {"name": "parse_resume_details", "input": "Raw resume text"}
    )
    try:
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
        structured_llm = llm.with_structured_output(ParsedDetails)
        prompt = f"Analyze the following resume text and extract the information in a structured way. Do not summarize, just parse the requested fields.\n\nText:\n{state.raw_text}"
        response = structured_llm.invoke(prompt)
        state.parsed_details = response
    except Exception as e:
        state.error = f"Error during details parsing: {e}"
    return state


def create_concise_summary(state: GraphState) -> GraphState:
    """Node 3 (Tool 2): Creates a summary based on the text and parsed details."""
    if state.error or not state.raw_text:
        return state

    log_event(
        "TOOL_EXECUTE",
        {"name": "summarize_profile", "input": "Raw text and parsed details"},
    )
    try:
        llm = ChatOpenAI(temperature=0.2, model_name="gpt-4o-mini")

        # We provide the already-parsed details as context for a better summary!
        parsing_context = (
            state.parsed_details.json(indent=2)
            if state.parsed_details
            else "Not available"
        )

        prompt = f"""
        Based on the full resume text and the structured details extracted below, write a very concise and impactful 2-3 sentence summary.
        This summary should capture the essence of the candidate's profile for a busy recruiter.

        Structured details already extracted:
        {parsing_context}

        Full resume text:
        {state.raw_text}

        Write the summary here:
        """
        response = llm.invoke(prompt)
        state.concise_summary = response.content
    except Exception as e:
        state.error = f"Error during summary creation: {e}"
    return state


# ==============================================================================
# SECTION 4: BUILDING THE NEW GRANULAR GRAPH
# ==============================================================================


def build_graph():
    workflow = StateGraph(GraphState)

    # Add our more granular nodes/tools
    workflow.add_node("extract_text", extract_text_from_pdf)
    workflow.add_node("parse_details", parse_resume_details)
    workflow.add_node("create_summary", create_concise_summary)

    # Define the new execution chain
    workflow.set_entry_point("extract_text")
    workflow.add_edge("extract_text", "parse_details")
    workflow.add_edge("parse_details", "create_summary")
    workflow.add_edge("create_summary", END)

    return workflow.compile()


# ==============================================================================
# SECTION 5: MAIN EXECUTION WITH JSON OUTPUT
# ==============================================================================


def main():
    resume_folder = "resume/"
    if not os.path.exists(resume_folder):
        print(
            f"The '{resume_folder}' directory does not exist. Please create it and place PDF resumes inside."
        )
        return

    pdf_files = [f for f in os.listdir(resume_folder) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in the '{resume_folder}' directory.")
        return

    app = build_graph()
    all_results = []
    print(f"Starting granular processing of {len(pdf_files)} resume(s)...")

    for pdf_file in pdf_files:
        pdf_path = os.path.join(resume_folder, pdf_file)
        print(f"\n--- Processing: {pdf_file} ---")
        log_event("TASK_START", {"cv": pdf_file})

        final_state = app.invoke({"pdf_path": pdf_path})

        if final_state.get("error"):
            print(f"Error during processing: {final_state['error']}")
            log_event(
                "TASK_END",
                {"status": "ERROR", "cv": pdf_file, "error": final_state["error"]},
            )
        else:
            # We recombine the results from the different tools for the final output
            details = final_state.get("parsed_details")
            summary = final_state.get("concise_summary")

            if details:
                # Convert the Pydantic object to a dictionary
                final_record = details.dict()
                # Add the summary to the dictionary
                final_record["concise_summary"] = summary
                all_results.append(final_record)
                log_event("TASK_END", {"status": "SUCCESS", "cv": pdf_file})
            else:
                log_event(
                    "TASK_END",
                    {
                        "status": "WARNING",
                        "cv": pdf_file,
                        "message": "No details could be parsed.",
                    },
                )

    if all_results:
        output_filename = "summarized_resume.json"
        print(f"\n\n--- COMPILATION COMPLETE ---")

        # Write the final JSON file
        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)

        print(
            f"All resumes have been processed and the results are saved in '{output_filename}'"
        )


if __name__ == "__main__":
    main()
