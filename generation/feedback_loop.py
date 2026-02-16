import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _feedback_path() -> Path:
    return _project_root() / "data" / "user_feedback.jsonl"


def _ensure_store() -> Path:
    path = _feedback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    return path


def _short(text: str, limit: int = 180) -> str:
    normalized = " ".join((text or "").strip().split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


def save_feedback(record: Dict[str, Any]) -> Dict[str, Any]:
    path = _ensure_store()
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "decision": (record.get("decision") or "").strip().lower(),
        "notes": (record.get("notes") or "").strip(),
        "topic": (record.get("topic") or "").strip(),
        "post_type": (record.get("post_type") or "").strip().lower(),
        "target_persona": (record.get("target_persona") or "").strip().lower(),
        "final_post": (record.get("final_post") or "").strip(),
        "hashtags": (record.get("hashtags") or "").strip(),
        "brand_score": int(record.get("brand_score") or 0),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return {"path": str(path), "decision": payload["decision"]}


def _load_feedback() -> List[Dict[str, Any]]:
    path = _feedback_path()
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def build_feedback_guidance(post_type: str, target_persona: str, max_items: int = 6) -> Tuple[str, Dict[str, Any]]:
    rows = _load_feedback()
    if not rows:
        return "", {"accepted_count": 0, "rejected_count": 0}

    post_type_norm = (post_type or "").strip().lower()
    persona_norm = (target_persona or "").strip().lower()

    scoped = [
        r
        for r in rows
        if (not post_type_norm or r.get("post_type") == post_type_norm)
        and (not persona_norm or r.get("target_persona") == persona_norm)
    ]
    if not scoped:
        scoped = rows

    accepted = [r for r in scoped if r.get("decision") == "accept"][-max_items:]
    rejected = [r for r in scoped if r.get("decision") == "reject"][-max_items:]

    lines: List[str] = []
    if accepted:
        lines.append("User feedback memory - patterns to preserve:")
        for r in accepted[:3]:
            seed = r.get("notes") or r.get("final_post") or ""
            lines.append(f"- {_short(seed)}")

    if rejected:
        lines.append("User feedback memory - patterns to avoid:")
        for r in rejected[:3]:
            seed = r.get("notes") or r.get("final_post") or ""
            lines.append(f"- {_short(seed)}")

    return "\n".join(lines), {"accepted_count": len(accepted), "rejected_count": len(rejected)}
