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

from helper import score_text, DEFAULT_THRESHOLD  # we no longer import make_step/gather_task_texts

# --------------------------------------------------------------------------
# In-memory DB + lock
# --------------------------------------------------------------------------
def _load_runs() -> Dict[str, Any]:
    with open("new_runs.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept either dict keyed by run-id OR list of runs with "id"
    if isinstance(data, list):
        return {r["id"]: r for r in data}
    elif isinstance(data, dict):
        return data
    else:
        raise RuntimeError("runs.json must be a dict or a list")

DB: Dict[str, Any] = {"runs": _load_runs()}
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
class RunOut(BaseModel):
    run_id: str

class TaskOut(BaseModel):
    run_id: str
    task_id: str

class TaskCreate(BaseModel):
    run_id: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StepIn(BaseModel):
    run_id: str
    task_id: str
    step_id: str
    step: Dict[str, Any]           # { content?, tool_calls?, message_type: "ai"|"human"|"tool", ... }
    metrics: Dict[str, Any] = Field(default_factory=dict)  # { created_at: ISO, duration: seconds, metadata: {...} }

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
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

def iso_to_ms(iso_str: Optional[str]) -> Optional[int]:
    if not iso_str:
        return None
    try:
        return int(datetime.fromisoformat(iso_str.replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None

def _require_run(run_id: str) -> Dict[str, Any]:
    run = DB["runs"].get(run_id)
    if not run:
        raise HTTPException(404, "run not found")
    return run

def _require_task(run, task_id):
    for t in run.get("tasks", []):
        if t.get("id") == task_id:
            return t
    raise HTTPException(404, "task not found")

def _find_step(task: Dict[str, Any], step_id: str) -> Optional[Dict[str, Any]]:
    return next((s for s in task.get("steps", []) if s.get("id") == step_id), None)

# --------------------------------------------------------------------------
# Create run / task
# --------------------------------------------------------------------------
@app.post("/run", response_model=RunOut)
def create_run():
    with DB_LOCK:
        idx = len(DB["runs"]) + 1
        run_id = f"run-{idx}"
        DB["runs"][run_id] = {
            "id": run_id,
            "created_at": now_iso(),
            "tasks": []
        }
    return {"run_id": run_id}

@app.post("/task", response_model=TaskOut)
def create_task(payload: TaskCreate):
    run = _require_run(payload.run_id)
    with DB_LOCK:
        task_index = len(run.get("tasks", []))
        task_id = f"{payload.run_id}-task-{task_index}"
        task_obj = {
            "id": task_id,
            "name": payload.name or f"task {task_index+1}",
            "created_at": now_iso(),
            "metadata": payload.metadata,
            "steps": []
        }
        run.setdefault("tasks", []).append(task_obj)
    return {"run_id": payload.run_id, "task_id": task_id}

# --------------------------------------------------------------------------
# Add steps (single endpoint)
# --------------------------------------------------------------------------
@app.post("/step")
def add_step(s: StepIn):
    run = _require_run(s.run_id)
    task = _require_task(run, s.task_id)

    # Normalize message_type -> type
    msg_type = (s.step.get("message_type") or "").lower()
    type_map = {"ai": "thinking", "human": "human", "tool": "tool"}
    step_type = type_map.get(msg_type, "thinking")

    # Extract useful fields
    created_at_iso = s.metrics.get("created_at") or now_iso()
    duration = s.metrics.get("duration")
    content = s.step.get("content", "")

    tool_calls: List[Dict[str, Any]] = s.step.get("tool_calls", []) or []
    first_tool = tool_calls[0] if tool_calls else {}
    tool_name = first_tool.get("name")
    tool_args = first_tool.get("args")

    stored = {
    "id": s.step_id,
    "type": step_type,                   # "thinking" | "human" | "tool"
    "name": tool_name if step_type == "tool" else (s.step.get("name") or None),
    "created_at": created_at_iso,        # ISO string
    "duration": duration,                # seconds (float)
    "content": content,                  # keep same field name as in the incoming payload
    "input": tool_args if step_type == "tool" else None,
    "output": content if (step_type == "tool" and content) else None,
    "tool_calls": tool_calls,
    "metrics": s.metrics or {},
    "raw": {"step": s.step},
    "tags": {"replay": 0},
}

    with DB_LOCK:
        task["steps"].append(stored)

    return {"ok": True, "step_id": s.step_id}

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
        step.setdefault("tags", {}).setdefault("replay", 0)
        step["tags"]["replay"] += 1
        count = step["tags"]["replay"]

    recorded = {
        "type": step.get("type"),
        "created_at": step.get("created_at"),
        "content": step.get("content"),   # <— changed
        "input": step.get("input"),
        "output": step.get("output"),
        "metrics": step.get("metrics"),
    }
    patched = None
    if req.patch:
        # Patch the *output-like* content without mutating the DB (preview only)
        patched = {**(step.get("output") if isinstance(step.get("output"), dict) else {}), **req.patch}

    return {
        "run_id": req.run_id,
        "task_id": req.task_id,
        "step_id": req.step_id,
        "replay_count": count,
        "recorded": recorded,
        "patched_preview": patched
    }


# --------------------------------------------------------------------------
# Compliance (on flat steps)
# --------------------------------------------------------------------------
def _collect_text_items(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for st in task.get("steps", []):
        kind = st.get("type")
        if kind in ("thinking", "human", "tool"):  # only allowed kinds
            txt = st.get("content")
            if txt:
                items.append({
                    "kind": kind,
                    "content": str(txt),
                    "step_id": st.get("id"),
                    "ts_ms": iso_to_ms(st.get("created_at")) or 0
                })
    return items


@app.post("/compliance/check")
def compliance_check(payload: ComplianceCheckIn):
    run = _require_run(payload.run_id)
    threshold = payload.threshold if payload.threshold is not None else DEFAULT_THRESHOLD
    tasks = (
        [t for t in run.get("tasks", []) if t.get("id") == payload.task_id]
        if payload.task_id else run.get("tasks", [])
    )
    if not tasks:
        raise HTTPException(404, "no tasks to check")

    results = []
    total = 0
    violations = 0
    for task in tasks:
        for item in _collect_text_items(task):
            total += 1
            scored = score_text(item["content"], threshold)
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
                "snippet": item["content"][:200],
            })

    ratio = (violations / total) if total else 0.0
    with DB_LOCK:
        run.setdefault("compliance", []).append({
            "ts_ms": int(time.time() * 1000),
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
        [t for t in run.get("tasks", []) if t.get("id") == task_id]
        if task_id else run.get("tasks", [])
    )
    rows = []
    for task in tasks:
        for item in _collect_text_items(task):
            scored = score_text(item["content"], threshold)
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
                "snippet": item["content"][:200].replace("\n", " "),
            })
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0].keys()) if rows else
                            ["run_id","task_id","kind","step_id","ts_ms","top_label","top_score","is_violation","violations","snippet"])
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    buf.seek(0)
    return StreamingResponse(buf, media_type="content/csv",
                             headers={"Content-Disposition": f"attachment; filename='compliance_audit_{run_id}.csv'"})

# --------------------------------------------------------------------------
# Get runs and run details
# --------------------------------------------------------------------------
def _created_at_iso_for_run(run: Dict[str, Any]) -> Optional[str]:
    # prefer ISO string if present; else convert *_ms
    if "created_at" in run and isinstance(run["created_at"], str):
        return run["created_at"]
    if "created_at_ms" in run:
        return datetime.fromtimestamp(run["created_at_ms"]/1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return None

@app.get("/runs")
def list_runs():
    return {"runs": list(DB["runs"].keys())}

@app.get("/run/{run_id}")
def get_run(run_id: str):
    run = DB["runs"].get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    return run
