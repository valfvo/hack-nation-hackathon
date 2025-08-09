# backend/app/main.py
from __future__ import annotations
import os, threading, time, uuid
from typing import Any, Dict, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ======================================================================================
# In-memory DB
# ======================================================================================
DB: Dict[str, Any] = {"runs": {}}
DB_LOCK = threading.Lock()

def now_ms() -> int:
    return int(time.time() * 1000)

# ======================================================================================
# Data models
# ======================================================================================
class Metrics(BaseModel):
    exec_time_ms: Optional[int] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    model_latency_ms: Optional[int] = None

class RunOut(BaseModel):
    run_id: str

class TaskOut(BaseModel):
    run_id: str
    task_id: str

class TaskCreate(BaseModel):
    run_id: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class PromptIn(BaseModel):
    run_id: str
    task_id: str
    prompt: Dict[str, Any]
    metrics: Optional[Metrics] = None
    ts_ms: Optional[int] = None

class ThinkingIn(BaseModel):
    run_id: str
    task_id: str
    thinking: Dict[str, Any]
    metrics: Optional[Metrics] = None
    ts_ms: Optional[int] = None

class ToolIn(BaseModel):
    run_id: str
    task_id: str
    tool: Dict[str, Any]
    metrics: Optional[Metrics] = None
    ts_ms: Optional[int] = None

class OutputIn(BaseModel):
    run_id: str
    task_id: str
    output: Dict[str, Any]
    metrics: Optional[Metrics] = None
    ts_ms: Optional[int] = None

class ReplayIn(BaseModel):
    run_id: str
    task_id: str
    step_id: str
    patch: Optional[Dict[str, Any]] = None

# ======================================================================================
# App setup
# ======================================================================================
app = FastAPI(title="AgentOps Replay – prompt_chain model (dict DB)")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================================================================
# Helpers
# ======================================================================================
def _require_run(run_id: str) -> Dict[str, Any]:
    run = DB["runs"].get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run

def _require_task(run: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    task = run["tasks"].get(task_id)
    if not task:
        raise HTTPException(404, "task not found")
    return task

def _current_prompt_chain(task: Dict[str, Any]) -> Dict[str, Any]:
    if not task["prompt_chains"]:
        raise HTTPException(400, "no prompt_chain yet; call /prompt first")
    pc = task["prompt_chains"][-1]
    if pc["closed"]:
        raise HTTPException(400, "current prompt_chain already closed with output; call /prompt to start a new one")
    return pc

def _find_step(task: Dict[str, Any], step_id: str) -> Optional[Dict[str, Any]]:
    for pc in task["prompt_chains"]:
        if pc["prompt"] and pc["prompt"]["step_id"] == step_id:
            return pc["prompt"]
        for s in pc["thinking"]:
            if s["step_id"] == step_id:
                return s
        for s in pc["tools"]:
            if s["step_id"] == step_id:
                return s
        if pc["output"] and pc["output"]["step_id"] == step_id:
            return pc["output"]
    return None

def _make_step(step_type: str, ts_ms: Optional[int], data: Dict[str, Any], metrics: Optional[Metrics]) -> Dict[str, Any]:
    return {
        "step_id": uuid.uuid4().hex,
        "type": step_type,
        "ts_ms": ts_ms or now_ms(),
        "data": data,
        "metrics": metrics.model_dump() if metrics else None,
        "tags": {"replay": 0}
    }

# ======================================================================================
# Endpoints
# ======================================================================================
@app.get("/health")
def health():
    return {"ok": True, "runs": len(DB["runs"])}

@app.post("/run", response_model=RunOut)
def create_run():
    run_id = uuid.uuid4().hex
    with DB_LOCK:
        DB["runs"][run_id] = {
            "run_id": run_id,
            "created_at_ms": now_ms(),
            "status": "running",
            "tasks": {},
            "task_order": []
        }
    return {"run_id": run_id}

@app.post("/task", response_model=TaskOut)
def create_task(payload: TaskCreate):
    run = _require_run(payload.run_id)
    task_id = uuid.uuid4().hex
    task_obj = {
        "task_id": task_id,
        "name": payload.name or "task",
        "created_at_ms": now_ms(),
        "metadata": payload.metadata,
        "prompt_chains": []   # list of prompt_chains
    }
    with DB_LOCK:
        run["tasks"][task_id] = task_obj
        run["task_order"].append(task_id)
    return {"run_id": payload.run_id, "task_id": task_id}

@app.post("/prompt")
def add_prompt(p: PromptIn):
    run = _require_run(p.run_id)
    task = _require_task(run, p.task_id)
    step = _make_step("prompt", p.ts_ms, p.prompt, p.metrics)
    prompt_chain = {
        "prompt_chain_id": uuid.uuid4().hex,
        "prompt": step,
        "thinking": [],
        "tools": [],
        "output": None,
        "closed": False
    }
    with DB_LOCK:
        task["prompt_chains"].append(prompt_chain)
    return {
        "ok": True,
        "run_id": p.run_id,
        "task_id": p.task_id,
        "prompt_chain_id": prompt_chain["prompt_chain_id"],
        "prompt_step_id": step["step_id"]
    }

@app.post("/thinking")
def add_thinking(t: ThinkingIn):
    run = _require_run(t.run_id)
    task = _require_task(run, t.task_id)
    pc = _current_prompt_chain(task)
    step = _make_step("thinking", t.ts_ms, t.thinking, t.metrics)
    with DB_LOCK:
        pc["thinking"].append(step)
    return {
        "ok": True,
        "run_id": t.run_id,
        "task_id": t.task_id,
        "prompt_chain_id": pc["prompt_chain_id"],
        "thinking_step_id": step["step_id"]
    }

@app.post("/tool")
def add_tool(tp: ToolIn):
    run = _require_run(tp.run_id)
    task = _require_task(run, tp.task_id)
    pc = _current_prompt_chain(task)
    step = _make_step("tool", tp.ts_ms, tp.tool, tp.metrics)
    with DB_LOCK:
        pc["tools"].append(step)
    return {
        "ok": True,
        "run_id": tp.run_id,
        "task_id": tp.task_id,
        "prompt_chain_id": pc["prompt_chain_id"],
        "tool_step_id": step["step_id"]
    }

@app.post("/output")
def set_output(o: OutputIn):
    run = _require_run(o.run_id)
    task = _require_task(run, o.task_id)
    pc = _current_prompt_chain(task)
    if pc["output"] is not None:
        raise HTTPException(400, "this prompt_chain already has output; start a new one with /prompt")
    step = _make_step("output", o.ts_ms, o.output, o.metrics)
    with DB_LOCK:
        pc["output"] = step
        pc["closed"] = True
    return {
        "ok": True,
        "run_id": o.run_id,
        "task_id": o.task_id,
        "prompt_chain_id": pc["prompt_chain_id"],
        "output_step_id": step["step_id"]
    }

@app.post("/replay")
def replay_step(req: ReplayIn):
    run = _require_run(req.run_id)
    task = _require_task(run, req.task_id)
    step = _find_step(task, req.step_id)
    if not step:
        raise HTTPException(404, "step not found")
    with DB_LOCK:
        step["tags"]["replay"] = int(step["tags"].get("replay", 0)) + 1
        count = step["tags"]["replay"]
    recorded = {k: step[k] for k in ("type", "ts_ms", "data", "metrics")}
    patched = {**step["data"], **req.patch} if req.patch else None
    return {
        "run_id": req.run_id,
        "task_id": req.task_id,
        "step_id": req.step_id,
        "replay_count": count,
        "recorded": recorded,
        "patched_preview": patched
    }

@app.get("/runs")
def list_runs():
    with DB_LOCK:
        return [
            {"run_id": r["run_id"], "created_at_ms": r["created_at_ms"], "status": r["status"]}
            for r in DB["runs"].values()
        ]

@app.get("/runs/{run_id}")
def get_run(run_id: str):
    run = _require_run(run_id)
    return {
        "run_id": run["run_id"],
        "created_at_ms": run["created_at_ms"],
        "status": run["status"],
        "task_order": run["task_order"],
        "tasks": run["tasks"]
    }

@app.post("/run/{run_id}/end")
def end_run(run_id: str):
    run = _require_run(run_id)
    with DB_LOCK:
        run["status"] = "ended"
    return {"ok": True, "run_id": run_id}
