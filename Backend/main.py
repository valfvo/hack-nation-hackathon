from __future__ import annotations
import os
import io
import csv
import time
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional, List
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from helper import make_step, gather_task_texts, score_text, DEFAULT_THRESHOLD

# --------------------------------------------------------------------------
# In-memory DB + lock
# --------------------------------------------------------------------------
with open("runs.json", "r", encoding="utf-8") as f:
    DB = {"runs": json.load(f)}
DB_LOCK = Lock()

# --------------------------------------------------------------------------
# App setup
# --------------------------------------------------------------------------
app = FastAPI(title="AgentOps Replay – Flat Steps Model with Compliance")

origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if origins == ["*"] else origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
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

class InputIn(BaseModel):
    run_id: str
    task_id: str
    input: Dict[str, Any]
    metrics: Optional[Metrics] = None

class ThinkingIn(BaseModel):
    run_id: str
    task_id: str
    thinking: Dict[str, Any]
    metrics: Optional[Metrics] = None

class ToolIn(BaseModel):
    run_id: str
    task_id: str
    tool: Dict[str, Any]
    metrics: Optional[Metrics] = None

class OutputIn(BaseModel):
    run_id: str
    task_id: str
    output: Dict[str, Any]
    metrics: Optional[Metrics] = None

class ReplayIn(BaseModel):
    run_id: str
    task_id: str
    step_id: str
    patch: Optional[Dict[str, Any]] = None

class ComplianceCheckIn(BaseModel):
    run_id: str
    task_id: Optional[str] = None
    threshold: Optional[float] = None

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def now_ms():
    return int(time.time() * 1000)

def to_iso(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")

def _require_run(run_id: str) -> Dict[str, Any]:
    run = DB["runs"].get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run

def _require_task(run, task_id):
    for t in run["tasks"]:
        if t["id"] == task_id:
            return t
    raise HTTPException(404, "task not found")

def _find_step(task: Dict[str, Any], step_id: str) -> Optional[Dict[str, Any]]:
    return next((s for s in task["steps"] if s["id"] == step_id), None)

# --------------------------------------------------------------------------
# Create run / task
# --------------------------------------------------------------------------
@app.post("/run", response_model=RunOut)
def create_run():
    run_id = f"run-{len(DB['runs'])+1}"
    with DB_LOCK:
        DB["runs"][run_id] = {
            "id": run_id,
            "created_at": now_ms(),
            "tasks": []
        }
    return {"run_id": run_id}

@app.post("/task", response_model=TaskOut)
def create_task(payload: TaskCreate):
    run = _require_run(payload.run_id)
    task_index = len(run["tasks"])
    task_id = f"{payload.run_id}-task-{task_index}"
    task_obj = {
        "id": task_id,
        "name": payload.name or f"task {task_index+1}",
        "created_at": now_ms(),
        "metadata": payload.metadata,
        "steps": []
    }
    with DB_LOCK:
        run["tasks"].append(task_obj)
    return {"run_id": payload.run_id, "task_id": task_id}

# --------------------------------------------------------------------------
# Add steps
# --------------------------------------------------------------------------
@app.post("/input")
def add_input(p: InputIn):
    run = _require_run(p.run_id)
    task = _require_task(run, p.task_id)
    step_index = len(task["steps"])
    step = make_step(p.run_id, task["id"].split("-task-")[1], "input", step_index, p.input, p.metrics.model_dump() if p.metrics else None)
    with DB_LOCK:
        task["steps"].append(step)
    return {"ok": True, "step_id": step["id"]}

@app.post("/thinking")
def add_thinking(t: ThinkingIn):
    run = _require_run(t.run_id)
    task = _require_task(run, t.task_id)
    step_index = len(task["steps"])
    step = make_step(t.run_id, task["id"].split("-task-")[1], "thinking", step_index, t.thinking, t.metrics.model_dump() if t.metrics else None)
    with DB_LOCK:
        task["steps"].append(step)
    return {"ok": True, "step_id": step["id"]}

@app.post("/tool")
def add_tool(tp: ToolIn):
    run = _require_run(tp.run_id)
    task = _require_task(run, tp.task_id)
    step_index = len(task["steps"])
    step = make_step(tp.run_id, task["id"].split("-task-")[1], "tool", step_index, tp.tool, tp.metrics.model_dump() if tp.metrics else None)
    with DB_LOCK:
        task["steps"].append(step)
    return {"ok": True, "step_id": step["id"]}

@app.post("/output")
def add_output(o: OutputIn):
    run = _require_run(o.run_id)
    task = _require_task(run, o.task_id)
    step_index = len(task["steps"])
    step = make_step(o.run_id, task["id"].split("-task-")[1], "output", step_index, o.output, o.metrics.model_dump() if o.metrics else None)
    with DB_LOCK:
        task["steps"].append(step)
    return {"ok": True, "step_id": step["id"]}

# --------------------------------------------------------------------------
# Replay
# --------------------------------------------------------------------------
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
    recorded = {k: step[k] for k in ("type", "timestamp", "data", "metrics")}
    patched = {**step["data"], **req.patch} if req.patch else None
    return {
        "run_id": req.run_id,
        "task_id": req.task_id,
        "step_id": req.step_id,
        "replay_count": count,
        "recorded": recorded,
        "patched_preview": patched
    }

# --------------------------------------------------------------------------
# Compliance
# --------------------------------------------------------------------------
@app.post("/compliance/check")
def compliance_check(payload: ComplianceCheckIn):
    run = _require_run(payload.run_id)
    threshold = payload.threshold if payload.threshold is not None else DEFAULT_THRESHOLD
    tasks = (
        [t for t in run["tasks"] if t["id"] == payload.task_id]
        if payload.task_id else run["tasks"]
    )
    if not tasks:
        raise HTTPException(404, "no tasks to check")
    results = []
    total = 0
    violations = 0
    for task in tasks:
        texts = gather_task_texts(task)
        for item in texts:
            total += 1
            scored = score_text(item["text"], threshold)
            if scored["is_violation"]:
                violations += 1
            results.append({
                "run_id": run["id"],
                "task_id": task["id"],
                "kind": item["kind"],
                "step_id": item["step_id"],
                "ts_ms": item["ts_ms"],
                "top_label": scored["top_label"],
                "top_score": scored["top_score"],
                "is_violation": scored["is_violation"],
                "violations": scored["violations"],
                "label_scores": scored["label_scores"],
                "snippet": item["text"][:200],
            })
    ratio = (violations / total) if total else 0.0
    with DB_LOCK:
        run.setdefault("compliance", []).append({
            "ts_ms": now_ms(),
            "threshold": threshold,
            "total_checked": total,
            "violations": violations,
            "ratio": ratio,
        })
    return {"summary": {"total_checked": total, "violations": violations, "ratio": ratio, "threshold": threshold}, "results": results}

@app.get("/compliance/audit/{run_id}.csv")
def compliance_audit_export(run_id: str, threshold: float = DEFAULT_THRESHOLD, task_id: Optional[str] = None):
    run = _require_run(run_id)
    tasks = (
        [t for t in run["tasks"] if t["id"] == task_id]
        if task_id else run["tasks"]
    )
    rows = []
    for task in tasks:
        for item in gather_task_texts(task):
            scored = score_text(item["text"], threshold)
            rows.append({
                "run_id": run_id,
                "task_id": task["id"],
                "kind": item["kind"],
                "step_id": item["step_id"],
                "ts_ms": item["ts_ms"],
                "top_label": scored["top_label"],
                "top_score": f"{scored['top_score']:.4f}",
                "is_violation": "yes" if scored["is_violation"] else "no",
                "violations": ";".join(f"{k}:{v:.3f}" for k, v in scored["violations"].items()),
                "snippet": item["text"][:200].replace("\n", " "),
            })
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else
                            ["run_id","task_id","kind","step_id","ts_ms","top_label","top_score","is_violation","violations","snippet"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    buf.seek(0)
    return StreamingResponse(buf, media_type="text/csv",
                             headers={"Content-Disposition": f"attachment; filename='compliance_audit_{run_id}.csv'"})

# --------------------------------------------------------------------------
# Get runs and run details
# --------------------------------------------------------------------------
@app.get("/runs")
def list_runs():
    return [
        {
            "id": run["id"],
            "created_at": to_iso(run["created_at"]),
            "tasks_count": len(run.get("tasks", [])),
        }
        for run in DB["runs"].values()
    ]

@app.get("/run/{run_id}")
def get_run(run_id: str):
    run = _require_run(run_id)
    return {
        "id": run["id"],
        "tasks": [
            {
                "id": task["id"],
                "name": task.get("name", ""),
                "steps": [
                    {
                        "id": step["id"],
                        "type": step["type"],
                        "title": step["type"].capitalize() if step["type"] in ("input", "thinking") else None,
                        "name": step["data"].get("name") if step["type"] == "tool" else None,
                        "text": step["data"].get("text") if step["type"] in ("input", "thinking") else None,
                        "input": step["data"].get("input") if step["type"] == "tool" else None,
                        "output": step["data"].get("output") if step["type"] == "tool" else None,
                        "attribute": {k: v for k, v in step["data"].items() if k not in ("name", "input", "output")} if step["type"] == "tool" else None,
                        "duration": step["metrics"].get("exec_time_ms") if step["type"] == "tool" else None,
                        "timestamp": to_iso(step["timestamp"])
                    }
                    for step in task["steps"]
                ]
            }
            for task in run["tasks"]
        ]
    }
