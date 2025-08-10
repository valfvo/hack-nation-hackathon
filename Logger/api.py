from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional
from wrapper import LoggingAgentWrapper
from Agent.agent import build_react_agent_graph
from langchain_core.messages import HumanMessage


# ----- FastAPI Models -----
class InvokeRequest(BaseModel):
    args: Dict[str, Any]
    config: Dict[str, Any]

class ReplayRequest(BaseModel):
    args: Dict[str, Any]
    checkpoint_id: Optional[str] = None
    thread_id: Optional[str] = None
    patch: Optional[Dict[str, Any]] = None


# ----- App Setup -----
app = FastAPI()
agent_wrapper = LoggingAgentWrapper(build_react_agent_graph())  # Assuming build_react_agent_graph is defined in Agent/agent.py


# ----- API Endpoints -----
@app.post("/invoke")
def invoke_endpoint(request: InvokeRequest):
    pdf_path = request.args.get("pdf_path")
    if not pdf_path:
        pdf_path = "resume/android-developer-1559034496.pdf"
    thread_id = request.config.get("run_id", "default-thread-id")
    required_skill = request.args.get("required_skill")
    prompt = f"Analyze the resume at '{pdf_path}'. The required skill for this job is '{required_skill}'. If the candidate has this skill, provide a summary. If not, send them a rejection email."
    initial_messages = [HumanMessage(content=prompt)]
    config = {"configurable": {"thread_id": thread_id}}
    result = agent_wrapper.invoke({"messages": initial_messages}, config=config)
    return result

@app.post("/replay")
def replay_endpoint(request: ReplayRequest):
    return agent_wrapper.replay(
        args=request.args,
        checkpoint_id=request.checkpoint_id,
        thread_id=request.thread_id,
        patch=request.patch
    )