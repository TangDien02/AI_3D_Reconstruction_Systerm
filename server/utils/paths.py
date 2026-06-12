from __future__ import annotations
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

def safe_slug(value: object, default: str = "object") -> str:
    text = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(value or "").strip().lower()).strip("-")
    return (text[:48] or default).strip("-") or default


def build_job_id(label: object = "object") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{safe_slug(label)}_{uuid.uuid4().hex[:8]}"


def job_output_dir(root_dir: Path, job_id: str) -> Path:
    date_part = job_id[:8] if len(job_id) >= 8 else datetime.now().strftime("%Y%m%d")
    return root_dir / date_part / job_id


def mounted_url(root_dir: Path, mount_path: str, path: Path) -> str:
    try:
        relative_path = Path(path).relative_to(root_dir).as_posix()
        return f"{mount_path}/{relative_path}"
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
