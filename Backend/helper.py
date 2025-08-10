import time
from threading import Lock
from typing import Dict, Any, List
from transformers import pipeline

# Lock for thread safety
DB_LOCK = Lock()

# Compliance classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Default threshold for violations
DEFAULT_THRESHOLD = 0.8

# Labels for compliance check
CANDIDATE_LABELS = [
    "harmful", "not harmful",
    "toxic", "safe",
    "unethical", "ethical",
    "bias", "unbiased"
]

def now_ms():
    return int(time.time() * 1000)

def make_step(run_index: str, task_index: str, step_type: str, step_index: int, step_id: str,
              data: Dict[str, Any], metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """Create a step object for flat steps model."""
    return {
        "id": f"run-{run_index}-t{task_index}-{step_type}-{step_index}",
        "step_id": step_id,
        "type": step_type,
        "timestamp": now_ms(),
        "data": data or {},
        "metrics": metrics or {},
        "tags": {"replay": 0}
    }

def gather_task_texts(task: Dict[str, Any]) -> List[Dict[str, Any]]:
    texts = []
    for step in task.get("steps", []):
        kind = step["type"]
        text_val = None
        if kind in ("input", "thinking", "output"):
            text_val = step["data"].get("text")
        if text_val:
            texts.append({
                "kind": kind,
                "text": str(text_val),
                "step_id": step["id"],
                "ts_ms": step["timestamp"]
            })
    return texts

def score_text(text: str, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, Any]:
    """
    Run the zero-shot classifier on text and return structured compliance results.
    """
    output = classifier(text, CANDIDATE_LABELS, multi_label=True)
    label_scores = dict(zip(output["labels"], output["scores"]))
    # Top label and score
    top_label = output["labels"][0]
    top_score = output["scores"][0]
    # Determine violations
    violations = {label: score for label, score in label_scores.items() if score >= threshold and "not" not in label.lower() and "safe" not in label.lower() and "ethical" not in label.lower() and "unbiased" not in label.lower()}
    return {
        "label_scores": label_scores,
        "top_label": top_label,
        "top_score": top_score,
        "is_violation": bool(violations),
        "violations": violations
    }
