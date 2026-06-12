from __future__ import annotations
from threading import Lock
from datetime import datetime

_reconstruction_jobs: dict[str, dict] = {}
_reconstruction_jobs_lock = Lock()
_texture_jobs: dict[str, dict] = {}
_texture_jobs_lock = Lock()

def set_reconstruction_job(job_id: str, payload: dict) -> None:
    with _reconstruction_jobs_lock:
        current = _reconstruction_jobs.get(job_id, {})
        current.update(payload)
        current["updated_at"] = datetime.utcnow().isoformat() + "Z"
        _reconstruction_jobs[job_id] = current


def get_reconstruction_job(job_id: str) -> dict | None:
    with _reconstruction_jobs_lock:
        job = _reconstruction_jobs.get(job_id)
        return dict(job) if job else None


def set_texture_job(job_id: str, payload: dict) -> None:
    with _texture_jobs_lock:
        current = _texture_jobs.get(job_id, {})
        current.update(payload)
        current["updated_at"] = datetime.utcnow().isoformat() + "Z"
        _texture_jobs[job_id] = current


def get_texture_job(job_id: str) -> dict | None:
    with _texture_jobs_lock:
        job = _texture_jobs.get(job_id)
        return dict(job) if job else None
