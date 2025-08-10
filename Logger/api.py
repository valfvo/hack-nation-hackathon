from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional
from agent_wrapper import LoggingAgentWrapper, DummyAgent


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
agent_wrapper = LoggingAgentWrapper(DummyAgent())

@app.post("/invoke")
def invoke_endpoint(request: InvokeRequest):
    return agent_wrapper.invoke(request.args, request.config)

@app.post("/stream")
def stream_endpoint(request: InvokeRequest):
    return list(agent_wrapper.stream(request.args, request.config))

@app.post("/replay")
def replay_endpoint(request: ReplayRequest):
    return agent_wrapper.replay(
        args=request.args,
        checkpoint_id=request.checkpoint_id,
        thread_id=request.thread_id,
        patch=request.patch
    )