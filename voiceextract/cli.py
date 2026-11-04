"""Command-line entry point.

  python -m voiceextract.cli analyze interview.mp4 --job demo [--speakers 4] [--no-separate]
  python -m voiceextract.cli identify --job demo --text "目标人物说过的一句话" [--audio ref.wav]
  python -m voiceextract.cli export --job demo --speaker S1 --name target --slices --out output/target.mp3
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

from . import pipeline
from .jobs import JobStore


def _printer():
    last = [0.0]

    def report(stage, frac, msg):
        now = time.time()
        if now - last[0] > 2 or frac >= 1:
            last[0] = now
            print(f"[{frac:6.1%}] {stage:9s} {msg}", flush=True)

    return report


def cmd_analyze(a):
    store = JobStore()
    src = Path(a.input).resolve()
    opts = {"separate": not a.no_separate, "num_speakers": a.speakers}
    job = store.get(a.job) or store.create(src.name, opts, job_id=a.job)
    store.update(job["id"], source_path=str(src), options=opts, status="running", error=None, pid=os.getpid(), started=time.time())
    report = _printer()

    def rep(stage, frac, msg):
        store.update(job["id"], stage=stage, progress=frac, message=msg)
        report(stage, frac, msg)

    t = time.time()
    try:
        res = pipeline.analyze(src, store.dir(job["id"]), opts, rep)
    except Exception as e:
        store.update(job["id"], status="error", error=str(e))
        raise
    store.update(job["id"], status="done", progress=1.0, message="分析完成", elapsed=round(time.time() - t, 1), finished=time.time())
    print(f"done in {time.time() - t:.0f}s")
    for s in res["speakers"]:
        print(f"  {s['id']}: clean {s['clean_seconds']:7.1f}s  raw {s['raw_seconds']:7.1f}s  segs {s['num_segments']:4d}  similar {s['similar']}")


def cmd_identify(a):
    d = JobStore().dir(a.job)
    res = json.loads((d / "result.json").read_text(encoding="utf-8"))
    if a.text:
        r = pipeline.search_text(res, a.text)
        print("text votes:", r["votes"])
        for h in r["hits"][:10]:
            print(f"  {h['spk']} {h['start']:8.2f}s {h['score']:.2f} {h['text']}")
    if a.audio:
        print("audio match:", pipeline.match_audio(d, Path(a.audio)))


def cmd_export(a):
    store = JobStore()
    d = store.dir(a.job)
    opts = {"denoise": not a.no_denoise, "slices": a.slices}
    key = a.name or "_".join(a.speaker)
    meta = pipeline.export(d, a.speaker, opts, lambda f, m: print(f"[{f:6.1%}] {m}", flush=True), name=key)
    print(json.dumps(meta, ensure_ascii=False, indent=1))
    # list it in the SPA's "exported files" too
    exports = (store.get(a.job) or {}).get("exports") or {}
    exports[key] = {"status": "done", "progress": 1.0, "message": "完成", "speakers": a.speaker, "meta": meta, "updated": time.time()}
    names = (store.get(a.job) or {}).get("names") or {}
    if a.name and len(a.speaker) == 1:
        names[a.speaker[0]] = a.name
    store.update(a.job, exports=exports, names=names)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        for kind, fn in meta["files"].items():
            target = out if kind == "mp3" else out.with_name(out.stem + Path(fn).name[len(Path(meta["files"]["mp3"]).stem):])
            shutil.copy2(d / "export" / fn, target)
            print("wrote", target)


def main(argv=None):
    p = argparse.ArgumentParser(prog="voiceextract")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("analyze")
    s.add_argument("input")
    s.add_argument("--job", required=True)
    s.add_argument("--speakers", type=int, default=0)
    s.add_argument("--no-separate", action="store_true")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("identify")
    s.add_argument("--job", required=True)
    s.add_argument("--text")
    s.add_argument("--audio")
    s.set_defaults(fn=cmd_identify)
    s = sub.add_parser("export")
    s.add_argument("--job", required=True)
    s.add_argument("--speaker", action="append", required=True)
    s.add_argument("--name")
    s.add_argument("--out")
    s.add_argument("--no-denoise", action="store_true")
    s.add_argument("--slices", action="store_true")
    s.set_defaults(fn=cmd_export)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
