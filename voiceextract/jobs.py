"""On-disk job store shared by the web server and the CLI.

Each job lives in ``jobs/<id>/`` with a ``job.json`` holding its status.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from .models import ROOT

JOBS_DIR = Path(os.environ.get("AV_JOBS_DIR") or ROOT / "jobs")
_lock = threading.Lock()


def pid_alive(pid) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(0x1000, False, int(pid))  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _slug(name: str) -> str:
    s = re.sub(r"[^\w一-鿿-]+", "_", Path(name).stem).strip("_")
    return s[:40] or "job"


def _load(p: Path) -> dict:
    """The record in ``p``; raises OSError or ValueError if it is missing or damaged."""
    job = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(job, dict) or "id" not in job:
        raise ValueError(f"not a job record: {p}")
    return job


class JobStore:
    def __init__(self, root: Path = JOBS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, job_id: str) -> Path:
        if not re.fullmatch(r"[\w一-鿿-]+", job_id):
            raise KeyError(job_id)
        return self.root / job_id

    def create(self, source_name: str, options: dict, job_id: str | None = None) -> dict:
        job_id = job_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"
        d = self.dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        job = {
            "id": job_id,
            "name": source_name,
            "created": time.time(),
            "status": "queued",
            "stage": "",
            "progress": 0.0,
            "message": "排队中",
            "error": None,
            "options": options,
            "source_path": None,
        }
        self._write(job)
        return job

    def get(self, job_id: str) -> dict | None:
        try:
            return _load(self.dir(job_id) / "job.json")
        except (KeyError, OSError, ValueError):  # illegal ID, no record, damaged record
            return None

    def update(self, job_id: str, **fields) -> dict:
        with _lock:
            p = self.dir(job_id) / "job.json"
            # a damaged record raises here rather than being overwritten by a partial one
            job = _load(p) if p.exists() else {"id": job_id}
            job.update(fields)
            self._write(job)
        return job

    def list(self) -> list[dict]:
        jobs = []
        for p in self.root.glob("*/job.json"):
            try:
                jobs.append(_load(p))
            except (OSError, ValueError):
                continue
        return sorted(jobs, key=lambda j: -j.get("created", 0))

    def _write(self, job: dict) -> None:
        p = self.dir(job["id"]) / "job.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
