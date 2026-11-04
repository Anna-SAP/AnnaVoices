"""FastAPI backend for the SPA.

  .venv\\Scripts\\python -m voiceextract.server  [--port 8765]
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import tempfile
import threading
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import pipeline
from .jobs import JobStore, pid_alive
from .models import ROOT, models_ready

WEB_DIR = ROOT / "web"
store = JobStore()


@asynccontextmanager
async def lifespan(_app):
    _startup()
    yield


app = FastAPI(title="AnnaVoices · 目标说话人提取", lifespan=lifespan)

_analysis_q: queue.Queue[str] = queue.Queue()
_export_lock = threading.Lock()  # exports are CPU heavy: one at a time


# ------------------------------------------------------------------ background work


def _analysis_worker():
    while True:
        job_id = _analysis_q.get()
        job = store.get(job_id)
        if not job:
            continue
        store.update(job_id, status="running", error=None, started=time.time(), message="开始处理", pid=os.getpid())
        last = [0.0]

        def report(stage, frac, msg):
            now = time.time()
            if now - last[0] > 0.5 or frac >= 1:
                last[0] = now
                store.update(job_id, stage=stage, progress=round(frac, 4), message=msg)

        try:
            pipeline.analyze(Path(job["source_path"]), store.dir(job_id), job.get("options"), report)
            store.update(job_id, status="done", progress=1.0, message="分析完成", finished=time.time())
        except Exception as e:  # surfaced to the UI
            traceback.print_exc()
            store.update(job_id, status="error", error=f"{type(e).__name__}: {e}", message="处理失败")


def _run_export(job_id: str, key: str, speakers: list[str], opts: dict):
    def report(frac, msg):
        _set_export(job_id, key, status="running", progress=round(frac, 3), message=msg)

    with _export_lock:
        try:
            _set_export(job_id, key, status="running", progress=0.0, message="开始导出")
            meta = pipeline.export(store.dir(job_id), speakers, opts, report, name=key)
            _set_export(job_id, key, status="done", progress=1.0, message="完成", meta=meta)
        except Exception as e:
            traceback.print_exc()
            _set_export(job_id, key, status="error", message=f"{type(e).__name__}: {e}")


_exp_lock = threading.Lock()


def _set_export(job_id: str, key: str, **fields):
    with _exp_lock:
        job = store.get(job_id) or {}
        exports = job.get("exports") or {}
        exports[key] = {**exports.get(key, {}), **fields, "updated": time.time()}
        store.update(job_id, exports=exports)


def _startup():
    # jobs interrupted by a restart are re-queued (cached stages make this cheap);
    # jobs still being processed by another live process (e.g. the CLI) are left alone
    for job in reversed(store.list()):
        if job.get("status") in ("queued", "running") and job.get("source_path"):
            if job.get("status") == "running" and pid_alive(job.get("pid")):
                continue
            store.update(job["id"], status="queued", message="服务重启，重新排队")
            _analysis_q.put(job["id"])
        for key, ex in (job.get("exports") or {}).items():
            if ex.get("status") == "running":
                _set_export(job["id"], key, status="error", message="服务重启，导出中断")
    threading.Thread(target=_analysis_worker, daemon=True).start()


# ------------------------------------------------------------------------- helpers


def _job_or_404(job_id: str) -> dict:
    try:
        job = store.get(job_id)
    except KeyError:
        job = None
    if not job:
        raise HTTPException(404, "任务不存在")
    return job


def _result_or_409(job_id: str) -> dict:
    p = store.dir(job_id) / "result.json"
    if not p.exists():
        raise HTTPException(409, "分析尚未完成")
    return json.loads(p.read_text(encoding="utf-8"))


async def _save_upload(upload: UploadFile, dest: Path) -> None:
    with open(dest, "wb") as f:
        while chunk := await upload.read(1 << 20):
            f.write(chunk)


# ----------------------------------------------------------------------------- API


@app.get("/api/health")
def health():
    return {"ok": True, "models_ready": models_ready(), "threads": pipeline.NUM_THREADS, "stages": pipeline.STAGES}


@app.get("/api/jobs")
def list_jobs():
    keep = ("id", "name", "created", "status", "stage", "progress", "message", "error", "options")
    return [{k: j.get(k) for k in keep} for j in store.list()]


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    separate: bool = Form(True),
    num_speakers: int = Form(0),
):
    suffix = Path(file.filename or "input.mp4").suffix.lower() or ".mp4"
    opts = {"separate": separate, "num_speakers": max(0, min(20, num_speakers))}
    job = store.create(file.filename or "upload", opts)
    src = store.dir(job["id"]) / f"source{suffix}"
    await _save_upload(file, src)
    info = pipeline.probe_media(src)
    if not info["has_audio"]:
        shutil.rmtree(store.dir(job["id"]), ignore_errors=True)
        raise HTTPException(400, "无法识别该文件中的音轨")
    store.update(job["id"], source_path=str(src), media=info)
    _analysis_q.put(job["id"])
    return store.get(job["id"])


class ReanalyzeBody(BaseModel):
    num_speakers: int = Field(0, ge=0, le=20)
    separate: bool | None = None


@app.post("/api/jobs/{job_id}/reanalyze")
def reanalyze(job_id: str, body: ReanalyzeBody):
    job = _job_or_404(job_id)
    if job.get("status") in ("queued", "running"):
        raise HTTPException(409, "任务正在处理中")
    opts = {**(job.get("options") or {}), "num_speakers": body.num_speakers}
    if body.separate is not None:
        opts["separate"] = body.separate
    store.update(job_id, options=opts, status="queued", progress=0.0, message="排队中", error=None)
    _analysis_q.put(job_id)
    return store.get(job_id)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    job = _job_or_404(job_id)
    p = store.dir(job_id) / "result.json"
    if job.get("status") == "done" and p.exists():
        job["result"] = json.loads(p.read_text(encoding="utf-8"))
    return job


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    job = _job_or_404(job_id)
    if job.get("status") in ("queued", "running"):
        raise HTTPException(409, "任务正在处理中，无法删除")
    shutil.rmtree(store.dir(job_id), ignore_errors=True)
    return {"ok": True}


_SERVABLE = {"previews", "thumbs", "export"}


@app.get("/api/jobs/{job_id}/files/{path:path}")
def job_file(job_id: str, path: str, download: bool = False):
    _job_or_404(job_id)
    d = store.dir(job_id).resolve()
    p = (d / path).resolve()
    if d not in p.parents or not (p.parent.name in _SERVABLE or p.name == "vocals_browse.mp3") or not p.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(p, filename=p.name if download else None)


@app.get("/api/jobs/{job_id}/search")
def search(job_id: str, q: str):
    return pipeline.search_text(_result_or_409(job_id), q)


async def _with_temp_upload(upload: UploadFile, fn):
    suffix = Path(upload.filename or "ref").suffix or ".bin"
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td) / f"ref{suffix}"
        await _save_upload(upload, tmp)
        try:
            return fn(tmp)
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(400, f"无法处理该文件：{e}")


@app.post("/api/jobs/{job_id}/match/audio")
async def match_audio(job_id: str, file: UploadFile = File(...)):
    _result_or_409(job_id)
    scores = await _with_temp_upload(file, lambda p: pipeline.match_audio(store.dir(job_id), p))
    return {"scores": scores}


@app.post("/api/jobs/{job_id}/match/image")
async def match_image(job_id: str, file: UploadFile = File(...)):
    _result_or_409(job_id)
    job = _job_or_404(job_id)
    src = Path(job["source_path"])
    scores = await _with_temp_upload(file, lambda p: pipeline.match_image(store.dir(job_id), src, p))
    return {"scores": scores}


@app.put("/api/jobs/{job_id}/names")
def set_names(job_id: str, names: dict[str, str]):
    _job_or_404(job_id)
    clean = {k: v.strip()[:40] for k, v in names.items() if isinstance(v, str) and v.strip()}
    store.update(job_id, names=clean)
    return clean


class ExportBody(BaseModel):
    speakers: list[str]
    name: str | None = None
    denoise: bool = True
    denoise_db: float = Field(15.0, ge=3, le=100)
    loudnorm: bool = True
    lufs: float = Field(-18.0, ge=-30, le=-10)
    gap: float = Field(0.5, ge=0, le=3)
    min_sim: float = 0.0
    min_margin: float = -1.0
    min_segment: float = 1.0
    skip_events: bool = True
    exclude: list[int] = []
    slices: bool = False
    bitrate: str = "320k"


@app.post("/api/jobs/{job_id}/export")
def start_export(job_id: str, body: ExportBody):
    result = _result_or_409(job_id)
    known = {s["id"] for s in result["speakers"]}
    if not body.speakers or not set(body.speakers) <= known:
        raise HTTPException(400, "请选择有效的说话人")
    name = "".join(c for c in (body.name or "") if c.isalnum() or c in "-_") or "_".join(body.speakers)
    job = _job_or_404(job_id)
    if (job.get("exports") or {}).get(name, {}).get("status") == "running":
        raise HTTPException(409, "该导出正在进行中")
    opts = body.model_dump(exclude={"speakers", "name"})
    _set_export(job_id, name, status="queued", progress=0.0, message="排队中", speakers=body.speakers, meta=None)
    threading.Thread(target=_run_export, args=(job_id, name, body.speakers, opts), daemon=True).start()
    return {"key": name}


@app.get("/api/jobs/{job_id}/exports")
def exports(job_id: str):
    return _job_or_404(job_id).get("exports") or {}


@app.exception_handler(HTTPException)
async def _http_error(_, exc: HTTPException):
    return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)


class _RevalidatingStatic(StaticFiles):
    """Always revalidate the SPA files so an updated app.js is picked up."""

    async def get_response(self, path, scope):
        resp = await super().get_response(path, scope)
        resp.headers["Cache-Control"] = "no-cache"
        return resp


app.mount("/", _RevalidatingStatic(directory=WEB_DIR, html=True), name="web")


def main():
    import uvicorn

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8765)))
    ap.add_argument("--jobs-dir", help="where job folders live (default: ./jobs)")
    ap.add_argument("--open", action="store_true", help="open the browser once the server is up")
    a = ap.parse_args()
    if a.jobs_dir:
        global store
        store = JobStore(Path(a.jobs_dir))
    if a.open:
        import webbrowser

        threading.Timer(2.0, webbrowser.open, args=(f"http://{a.host}:{a.port}/",)).start()
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")


if __name__ == "__main__":
    main()
