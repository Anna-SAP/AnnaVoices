"""HTTP API: path safety, upload handling, job lifecycle and background state.

Requests run in-process through TestClient; analysis and export work is stubbed.
"""
from __future__ import annotations

import json
import os
import queue
from pathlib import Path

import pytest
from fastapi import HTTPException

from helpers import seg
from voiceextract import pipeline, server

RESULT = {
    "speakers": [{"id": "S1"}, {"id": "S2"}],
    "segments": [seg(id=0, text="今天天气不错"), seg(id=1, spk="S2", start=6.0, end=9.0, text="明天下雨")],
}
ILLEGAL_JOB_URLS = ["%2e%2e", "a%20b", "a.b"]
ONLY_ON_WINDOWS = pytest.mark.skipif(os.name != "nt", reason="these names are valid outside Windows")


def add_job(store, job_id="good", status="done", result=True):
    store.create("clip.mp4", {"separate": True, "num_speakers": 0}, job_id=job_id)
    d = store.dir(job_id)
    store.update(job_id, status=status, source_path=str(d / "source.mp4"))
    (d / "source.mp4").write_bytes(b"private source")
    if result:
        (d / "result.json").write_text(json.dumps(RESULT, ensure_ascii=False), encoding="utf-8")
    for sub in ("previews", "thumbs", "export"):
        (d / sub).mkdir()
    (d / "previews" / "S1.mp3").write_bytes(b"preview")
    (d / "thumbs" / "S1_0.jpg").write_bytes(b"thumb")
    (d / "export" / "S1.mp3").write_bytes(b"exported")
    (d / "vocals_browse.mp3").write_bytes(b"vocals")
    return d


@pytest.fixture
def job(store):
    d = add_job(store)
    add_job(store, "good2")  # shares a name prefix: catches string-prefix path checks
    (store.dir("good2") / "export" / "secret.mp3").write_bytes(b"sibling")
    (store.root.parent / "outside.txt").write_text("outside", encoding="utf-8")
    return d


@pytest.fixture
def has_audio(monkeypatch):
    monkeypatch.setattr(pipeline, "probe_media", lambda p: {"duration": 1.0, "has_video": False, "has_audio": True})


# ------------------------------------------------------------- serving job files


@pytest.mark.parametrize(
    "path, body",
    [
        ("previews/S1.mp3", b"preview"),
        ("thumbs/S1_0.jpg", b"thumb"),
        ("export/S1.mp3", b"exported"),
        ("vocals_browse.mp3", b"vocals"),
    ],
)
def test_files_serves_the_allowed_folders(client, job, path, body):
    r = client.get(f"/api/jobs/good/files/{path}")
    assert r.status_code == 200
    assert r.content == body


def test_files_only_forces_a_download_when_asked(client, job):
    assert "attachment" not in client.get("/api/jobs/good/files/export/S1.mp3").headers.get("content-disposition", "")
    r = client.get("/api/jobs/good/files/export/S1.mp3?download=true")
    assert "attachment" in r.headers["content-disposition"]


@pytest.mark.parametrize(
    "path",
    [
        "job.json",
        "result.json",
        "source.mp4",
        "previews",
        "export/",
        "export/missing.mp3",
        "..%2f..%2foutside.txt",
        "%2e%2e/%2e%2e/outside.txt",
        "%2e%2e%2f%2e%2e%2foutside.txt",
        "..%5c..%5coutside.txt",
        "..%2fgood2%2fexport%2fsecret.mp3",
        "previews%2f..%2fjob.json",
        "previews%2f..%2fsource.mp4",
        "export%2f..%2fresult.json",
        "previews%2f..%2f..%2f..%2foutside.txt",
        "C%3a%2fWindows%2fwin.ini",
        "C%3aWindows%2fwin.ini",
        "%5c%5clocalhost%5cc$%5cWindows%5cwin.ini",
        "job.json::$DATA",
        "previews/" + "a" * 5000,
    ],
)
def test_files_refuses_everything_else_over_http(client, job, path):
    r = client.get(f"/api/jobs/good/files/{path}")
    assert r.status_code == 404
    assert b"private" not in r.content and b"outside" not in r.content


def refused(path):
    with pytest.raises(HTTPException) as e:
        server.job_file("good", path)
    return e.value.status_code == 404


@pytest.mark.parametrize(
    "path",
    [
        "../../outside.txt",
        "..\\..\\outside.txt",
        "../good2/export/secret.mp3",
        "..\\good2\\export\\secret.mp3",
        "previews/../job.json",
        "previews\\..\\source.mp4",
        "previews/../../../outside.txt",
        "../good/job.json",
        "",
        ".",
        "..",
    ],
)
def test_job_file_refuses_raw_traversal(store, job, path):
    """Calls the handler directly so no client or router can normalise the path first."""
    assert refused(path)


def test_job_file_refuses_absolute_paths_to_real_files(store, job):
    assert refused(str(store.root.parent / "outside.txt"))
    assert refused(str(store.dir("good2") / "export" / "secret.mp3"))
    assert refused(str(store.dir("good") / "source.mp4"))


def test_job_file_refuses_a_symlink_out_of_the_job(store, job):
    link = store.dir("good") / "export" / "link.mp3"
    try:
        link.symlink_to(store.root.parent / "outside.txt")
    except (OSError, NotImplementedError):
        pytest.skip("creating symlinks needs extra rights here")
    assert refused("export/link.mp3")


@pytest.mark.parametrize("job_id", ["ghost", *ILLEGAL_JOB_URLS])
def test_files_of_an_unknown_job(client, job, job_id):
    assert client.get(f"/api/jobs/{job_id}/files/vocals_browse.mp3").status_code == 404


@pytest.mark.parametrize("job_id", ["export", "previews", "thumbs"])
@pytest.mark.parametrize("name", ["job.json", "source.mp4", "result.json"])
def test_files_whitelist_holds_for_jobs_named_like_a_folder(client, store, job_id, name):
    add_job(store, job_id)
    assert client.get(f"/api/jobs/{job_id}/files/{name}").status_code == 404


@pytest.mark.parametrize("job_id", ["export", "previews", "thumbs"])
def test_files_keeps_serving_jobs_named_like_a_folder(client, store, job_id):
    add_job(store, job_id)
    assert client.get(f"/api/jobs/{job_id}/files/export/S1.mp3").content == b"exported"
    assert client.get(f"/api/jobs/{job_id}/files/vocals_browse.mp3").content == b"vocals"


@pytest.mark.parametrize(
    "path", ["sub/export/S1.mp3", "export/sub/S1.mp3", "export/previews/S1.mp3", "sub/vocals_browse.mp3"]
)
def test_files_only_serves_the_top_of_the_allowed_folders(client, job, store, path):
    target = store.dir("good") / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"private")
    assert client.get(f"/api/jobs/good/files/{path}").status_code == 404


def test_files_with_a_nul_byte(client, job):
    assert client.get("/api/jobs/good/files/previews%2fa%00b.mp3").status_code == 404


@pytest.mark.parametrize("path", ["/..%2f..%2fREADME.md", "/%2e%2e/requirements.txt", "/%2e%2e%2frequirements.txt", "/..%5crequirements.txt"])
def test_static_files_cannot_leave_the_web_folder(client, path):
    assert client.get(path).status_code == 404


@pytest.mark.parametrize("path", ["/", "/app.js", "/style.css"])
def test_static_files_are_always_revalidated(client, path):
    r = client.get(path)
    assert r.status_code == 200
    assert r.headers["cache-control"] == "no-cache"


def test_health(client):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert isinstance(body["models_ready"], bool)
    assert [s[0] for s in body["stages"]] == [s[0] for s in pipeline.STAGES]


# ------------------------------------------------------------------------ uploads


@pytest.mark.parametrize(
    "filename",
    ["../../evil.mp4", "..\\..\\evil.mp4", "/etc/passwd", "C:\\Windows\\evil.mp4", "..%2f..%2fevil.mp4", "../../../../evil"],
)
def test_upload_name_cannot_escape_the_job_folder(client, store, has_audio, filename):
    r = client.post("/api/jobs", files={"file": (filename, b"data")})
    assert r.status_code == 200
    src = Path(r.json()["source_path"])
    assert src.parent == store.dir(r.json()["id"])
    assert src.stem == "source"
    assert src.read_bytes() == b"data"
    assert sorted(p.name for p in store.root.parent.iterdir()) == ["jobs"]


@pytest.mark.parametrize(
    "filename, suffix",
    [("a.mp4", ".mp4"), ("a.MP4", ".mp4"), ("a.WAV", ".wav"), ("noext", ".mp4"), (".mp4", ".mp4"), ("a.b.mkv", ".mkv")],
)
def test_upload_keeps_a_lowercase_suffix(client, has_audio, filename, suffix):
    r = client.post("/api/jobs", files={"file": (filename, b"data")})
    assert Path(r.json()["source_path"]).name == f"source{suffix}"


def test_upload_queues_the_job(client, store, has_audio):
    body = client.post("/api/jobs", files={"file": ("a.mp4", b"data")}).json()
    assert body["media"]["has_audio"] is True
    assert server._analysis_q.get_nowait() == body["id"]


def test_upload_of_an_empty_file_is_stored_as_is(client, has_audio):
    r = client.post("/api/jobs", files={"file": ("a.mp4", b"")})
    assert r.status_code == 200
    assert Path(r.json()["source_path"]).stat().st_size == 0


@pytest.mark.parametrize("sent, kept", [("0", 0), ("5", 5), ("20", 20), ("21", 20), ("999", 20), ("-5", 0)])
def test_upload_clamps_the_speaker_count(client, has_audio, sent, kept):
    r = client.post("/api/jobs", files={"file": ("a.mp4", b"x")}, data={"num_speakers": sent})
    assert r.json()["options"]["num_speakers"] == kept


def test_upload_without_audio_is_refused_and_cleaned_up(client, store, monkeypatch):
    monkeypatch.setattr(pipeline, "probe_media", lambda p: {"duration": 0.0, "has_video": False, "has_audio": False})
    r = client.post("/api/jobs", files={"file": ("a.mp4", b"not media")})
    assert r.status_code == 400
    assert list(store.root.iterdir()) == []
    assert server._analysis_q.empty()


@pytest.mark.parametrize(
    "request_args",
    [
        {},
        {"files": {"file": ("", b"data")}},
        {"files": {"file": ("a.mp4", b"x")}, "data": {"num_speakers": "abc"}},
    ],
)
def test_upload_validation_errors(client, store, request_args):
    assert client.post("/api/jobs", **request_args).status_code == 422
    assert list(store.root.iterdir()) == []


@ONLY_ON_WINDOWS
@pytest.mark.parametrize("filename", ["a.m?4", "a.<b>", "a." + "x" * 400])
def test_upload_with_an_unusable_suffix_is_handled_cleanly(client, store, has_audio, filename):
    r = client.post("/api/jobs", files={"file": (filename, b"data")})
    assert r.status_code < 500
    assert [j for j in store.list() if j["source_path"] is None] == []


@ONLY_ON_WINDOWS
@pytest.mark.parametrize("kind", ["audio", "image"])
def test_match_with_an_unusable_suffix_is_refused_cleanly(client, job, kind):
    r = client.post(f"/api/jobs/good/match/{kind}", files={"file": ("a.m?4", b"x")})
    assert r.status_code < 500


@pytest.mark.parametrize("filename", ["a.m?4", "a.<b>", "a.mp 4", "a.视频", "a." + "x" * 400])
def test_upload_replaces_an_unusable_suffix(client, has_audio, filename):
    src = Path(client.post("/api/jobs", files={"file": (filename, b"data")}).json()["source_path"])
    assert src.name == "source.mp4"
    assert src.read_bytes() == b"data"


def test_match_replaces_an_unusable_suffix(client, job, monkeypatch):
    seen = []
    monkeypatch.setattr(pipeline, "match_audio", lambda job_dir, ref: seen.append(ref.suffix) or {})
    assert client.post("/api/jobs/good/match/audio", files={"file": ("a.m?4", b"x")}).status_code == 200
    assert seen == [".bin"]


def test_a_failed_upload_leaves_no_job_behind(client, store, monkeypatch):
    def no_ffmpeg(src):
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(pipeline, "probe_media", no_ffmpeg)
    assert client.post("/api/jobs", files={"file": ("a.mp4", b"data")}).status_code == 500
    assert list(store.root.iterdir()) == []
    assert server._analysis_q.empty()


# ------------------------------------------------------------------ job lifecycle


def test_list_starts_empty(client):
    assert client.get("/api/jobs").json() == []


def test_list_only_exposes_summary_fields(client, job):
    entry = next(j for j in client.get("/api/jobs").json() if j["id"] == "good")
    assert "source_path" not in entry
    assert entry["name"] == "clip.mp4"


def test_list_skips_a_corrupt_job(client, job, store):
    (store.dir("good") / "job.json").write_text("{not json", encoding="utf-8")
    assert [j["id"] for j in client.get("/api/jobs").json()] == ["good2"]


@pytest.mark.parametrize("job_id", ["ghost", *ILLEGAL_JOB_URLS])
@pytest.mark.parametrize(
    "method, suffix, body",
    [
        ("GET", "", None),
        ("DELETE", "", None),
        ("GET", "/exports", None),
        ("POST", "/reanalyze", {"num_speakers": 0}),
        ("PUT", "/names", {"S1": "甲"}),
    ],
)
def test_unknown_jobs_are_404(client, method, suffix, body, job_id):
    assert client.request(method, f"/api/jobs/{job_id}{suffix}", json=body).status_code == 404


@ONLY_ON_WINDOWS
@pytest.mark.parametrize("job_id", ["NUL", "con", "COM1"])
def test_device_names_are_unknown_jobs(client, job_id):
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/search?q=x").status_code == 404


@pytest.mark.parametrize("job_id", ["ghost", *ILLEGAL_JOB_URLS])
@pytest.mark.parametrize(
    "method, suffix, kwargs",
    [
        ("GET", "/search?q=x", {}),
        ("POST", "/export", {"json": {"speakers": ["S1"]}}),
        ("POST", "/match/audio", {"files": {"file": ("a.wav", b"x")}}),
        ("POST", "/match/image", {"files": {"file": ("a.jpg", b"x")}}),
    ],
)
def test_unknown_jobs_are_404_on_result_endpoints(client, method, suffix, kwargs, job_id):
    assert client.request(method, f"/api/jobs/{job_id}{suffix}", **kwargs).status_code == 404


def test_a_corrupt_job_does_not_crash_its_page(client, job, store):
    (store.dir("good") / "job.json").write_text("{not json", encoding="utf-8")
    assert client.get("/api/jobs/good").status_code < 500


@pytest.mark.parametrize(
    "method, suffix, kwargs",
    [
        ("GET", "/exports", {}),
        ("GET", "/search?q=x", {}),
        ("GET", "/files/vocals_browse.mp3", {}),
        ("POST", "/reanalyze", {"json": {}}),
        ("PUT", "/names", {"json": {"S1": "甲"}}),
        ("DELETE", "", {}),
    ],
)
def test_a_corrupt_job_is_treated_as_missing(client, job, store, method, suffix, kwargs):
    (store.dir("good") / "job.json").write_text("{not json", encoding="utf-8")
    assert client.request(method, f"/api/jobs/good{suffix}", **kwargs).status_code == 404
    assert (store.dir("good") / "job.json").read_text(encoding="utf-8") == "{not json"


def test_get_includes_the_result_once_done(client, job):
    assert client.get("/api/jobs/good").json()["result"]["speakers"] == RESULT["speakers"]


def test_get_leaves_out_the_result_until_done(client, store):
    add_job(store, "busy", status="running")
    assert "result" not in client.get("/api/jobs/busy").json()


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_jobs_cannot_be_deleted(client, store, status):
    d = add_job(store, "busy", status=status)
    assert client.delete("/api/jobs/busy").status_code == 409
    assert d.exists()


@pytest.mark.parametrize("status", ["done", "error"])
def test_finished_jobs_can_be_deleted(client, store, status):
    d = add_job(store, "old", status=status)
    assert client.delete("/api/jobs/old").json() == {"ok": True}
    assert not d.exists()
    assert client.get("/api/jobs/old").status_code == 404


@pytest.mark.parametrize("status", ["queued", "running"])
def test_active_jobs_cannot_be_reanalyzed(client, store, status):
    add_job(store, "busy", status=status)
    assert client.post("/api/jobs/busy/reanalyze", json={}).status_code == 409
    assert server._analysis_q.empty()


def test_reanalyze_requeues_with_merged_options(client, job, store):
    store.update("good", error="old failure")
    body = client.post("/api/jobs/good/reanalyze", json={"num_speakers": 3}).json()
    assert body["status"] == "queued"
    assert body["error"] is None
    assert body["options"] == {"separate": True, "num_speakers": 3}
    assert server._analysis_q.get_nowait() == "good"


def test_reanalyze_can_change_separation(client, job):
    assert client.post("/api/jobs/good/reanalyze", json={"separate": False}).json()["options"]["separate"] is False


@pytest.mark.parametrize("n", [-1, 21])
def test_reanalyze_validates_the_speaker_count(client, job, n):
    assert client.post("/api/jobs/good/reanalyze", json={"num_speakers": n}).status_code == 422


def test_names_are_trimmed_cut_and_stored(client, job, store):
    names = {"S1": "  姜文  ", "S2": "   ", "S3": "x" * 50}
    assert client.put("/api/jobs/good/names", json=names).json() == {"S1": "姜文", "S3": "x" * 40}
    assert store.get("good")["names"] == {"S1": "姜文", "S3": "x" * 40}


def test_names_must_be_text(client, job):
    assert client.put("/api/jobs/good/names", json={"S1": 1}).status_code == 422


# ---------------------------------------------------- endpoints that need a result


@pytest.mark.parametrize(
    "method, suffix, kwargs",
    [
        ("GET", "/search?q=x", {}),
        ("POST", "/export", {"json": {"speakers": ["S1"]}}),
        ("POST", "/match/audio", {"files": {"file": ("a.wav", b"x")}}),
        ("POST", "/match/image", {"files": {"file": ("a.jpg", b"x")}}),
    ],
)
def test_result_endpoints_wait_for_the_analysis(client, store, method, suffix, kwargs):
    add_job(store, "busy", status="running", result=False)
    assert client.request(method, f"/api/jobs/busy{suffix}", **kwargs).status_code == 409


def test_search_reads_the_job_result(client, job):
    assert [h["segment"] for h in client.get("/api/jobs/good/search", params={"q": "天气"}).json()["hits"]] == [0]


def test_search_with_an_empty_query(client, job):
    assert client.get("/api/jobs/good/search", params={"q": ""}).json() == {"hits": [], "votes": []}


def test_search_needs_a_query(client, job):
    assert client.get("/api/jobs/good/search").status_code == 422


@pytest.mark.parametrize("kind", ["audio", "image"])
def test_match_reports_bad_reference_files_as_400(client, job, monkeypatch, kind):
    def bad(*args):
        raise ValueError("参考文件不可用")

    monkeypatch.setattr(pipeline, f"match_{kind}", bad)
    r = client.post(f"/api/jobs/good/match/{kind}", files={"file": ("ref.bin", b"x")})
    assert r.status_code == 400
    assert r.json()["detail"] == "参考文件不可用"


@pytest.mark.parametrize("kind", ["audio", "image"])
def test_match_turns_unexpected_failures_into_400(client, job, monkeypatch, kind):
    def broken(*args):
        raise RuntimeError("decoder died")

    monkeypatch.setattr(pipeline, f"match_{kind}", broken)
    r = client.post(f"/api/jobs/good/match/{kind}", files={"file": ("ref.bin", b"x")})
    assert r.status_code == 400
    assert "decoder died" in r.json()["detail"]


def test_match_returns_scores(client, job, monkeypatch):
    monkeypatch.setattr(pipeline, "match_audio", lambda job_dir, ref: [{"id": "S1", "score": 0.9}])
    r = client.post("/api/jobs/good/match/audio", files={"file": ("ref.wav", b"x")})
    assert r.json() == {"scores": [{"id": "S1", "score": 0.9}]}


# ------------------------------------------------------------------------- export


@pytest.fixture
def export_calls(monkeypatch):
    """Replaces the export thread body; a queue because it runs on another thread."""
    calls = queue.Queue()
    monkeypatch.setattr(server, "_run_export", lambda *args: calls.put(args))
    return calls


@pytest.mark.parametrize("speakers", [[], ["S9"], ["S1", "S9"]])
def test_export_needs_known_speakers(client, job, export_calls, speakers):
    assert client.post("/api/jobs/good/export", json={"speakers": speakers}).status_code == 400
    assert export_calls.empty()


@pytest.mark.parametrize("field, value", [("gap", 99), ("gap", -1), ("lufs", 0), ("lufs", -40), ("denoise_db", 1), ("denoise_db", 101)])
def test_export_validates_its_options(client, job, export_calls, field, value):
    r = client.post("/api/jobs/good/export", json={"speakers": ["S1"], field: value})
    assert r.status_code == 422
    assert export_calls.empty()


@pytest.mark.parametrize(
    "name, key",
    [
        (None, "S1"),
        ("", "S1"),
        ("..", "S1"),
        ("../..", "S1"),
        ("../../evil", "evil"),
        ("..\\..\\evil", "evil"),
        ("/etc/passwd", "etcpasswd"),
        ("C:\\evil", "Cevil"),
        ("a/b c.mp3", "abcmp3"),
        ("姜文", "姜文"),
        ("a-b_c", "a-b_c"),
    ],
)
def test_export_name_is_reduced_to_a_safe_stem(client, job, export_calls, name, key):
    assert client.post("/api/jobs/good/export", json={"speakers": ["S1"], "name": name}).json() == {"key": key}
    job_id, used_key, speakers, opts = export_calls.get(timeout=5)
    assert (job_id, used_key, speakers) == ("good", key, ["S1"])
    assert "name" not in opts and "speakers" not in opts


def test_export_defaults_to_the_speaker_names(client, job, export_calls):
    assert client.post("/api/jobs/good/export", json={"speakers": ["S1", "S2"]}).json() == {"key": "S1_S2"}


def test_export_records_its_queued_state(client, job, export_calls):
    client.post("/api/jobs/good/export", json={"speakers": ["S1"]})
    record = client.get("/api/jobs/good/exports").json()["S1"]
    assert record["status"] == "queued"
    assert record["speakers"] == ["S1"]


def test_a_running_export_cannot_be_started_twice(client, job, store, export_calls):
    store.update("good", exports={"S1": {"status": "running"}})
    assert client.post("/api/jobs/good/export", json={"speakers": ["S1"]}).status_code == 409
    assert export_calls.empty()


def test_exports_start_empty(client, job):
    assert client.get("/api/jobs/good/exports").json() == {}


def test_run_export_records_success(store, job, monkeypatch):
    def fake_export(job_dir, speakers, opts, report, name=None):
        assert (job_dir, speakers, opts, name) == (store.dir("good"), ["S1"], {"gap": 0.5}, "key")
        report(0.5, "halfway")
        assert store.get("good")["exports"]["key"]["message"] == "halfway"
        return {"files": {"mp3": "key.mp3"}}

    monkeypatch.setattr(pipeline, "export", fake_export)
    server._run_export("good", "key", ["S1"], {"gap": 0.5})
    record = store.get("good")["exports"]["key"]
    assert record["status"] == "done"
    assert record["progress"] == 1.0
    assert record["meta"] == {"files": {"mp3": "key.mp3"}}


def test_run_export_records_failure(store, job, monkeypatch):
    def fake_export(*args, **kwargs):
        raise ValueError("没有符合条件的片段")

    monkeypatch.setattr(pipeline, "export", fake_export)
    server._run_export("good", "key", ["S1"], {})
    record = store.get("good")["exports"]["key"]
    assert record["status"] == "error"
    assert record["message"] == "ValueError: 没有符合条件的片段"


# --------------------------------------------------------- restart and analysis queue


@pytest.fixture
def startup(monkeypatch):
    """Runs _startup without starting the real analysis thread."""
    monkeypatch.setattr(server, "_analysis_worker", lambda: None)
    monkeypatch.setattr(server, "pid_alive", lambda pid: pid == 111)
    return server._startup


def queued_ids() -> list[str]:
    out = []
    while not server._analysis_q.empty():
        out.append(server._analysis_q.get_nowait())
    return out


def test_startup_requeues_waiting_jobs_oldest_first(store, startup):
    for name, created in (("second", 2.0), ("first", 1.0)):
        add_job(store, name, status="queued")
        store.update(name, created=created)
    startup()
    assert queued_ids() == ["first", "second"]


def test_startup_requeues_jobs_whose_worker_died(store, startup):
    add_job(store, "orphan", status="running")
    store.update("orphan", pid=222)
    startup()
    assert queued_ids() == ["orphan"]
    assert store.get("orphan")["status"] == "queued"


def test_startup_leaves_jobs_of_a_live_process_alone(store, startup):
    add_job(store, "cli", status="running")
    store.update("cli", pid=111)
    startup()
    assert queued_ids() == []
    assert store.get("cli")["status"] == "running"


@pytest.mark.parametrize("status", ["done", "error"])
def test_startup_ignores_finished_jobs(store, startup, status):
    add_job(store, "old", status=status)
    startup()
    assert queued_ids() == []


def test_startup_skips_jobs_without_a_source(store, startup):
    store.create("a.mp4", {}, job_id="empty")
    startup()
    assert queued_ids() == []


def test_startup_fails_interrupted_exports_only(store, startup):
    add_job(store, "old")
    store.update("old", exports={"a": {"status": "running"}, "b": {"status": "done"}})
    startup()
    exports = store.get("old")["exports"]
    assert exports["a"]["status"] == "error"
    assert exports["b"]["status"] == "done"


class StopWorker(Exception):
    pass


class FiniteQueue:
    """Hands out the given job IDs, then ends the worker's infinite loop."""

    def __init__(self, *job_ids):
        self.job_ids = list(job_ids)

    def get(self):
        if not self.job_ids:
            raise StopWorker
        return self.job_ids.pop(0)


def run_worker(monkeypatch, *job_ids):
    monkeypatch.setattr(server, "_analysis_q", FiniteQueue(*job_ids))
    with pytest.raises(StopWorker):
        server._analysis_worker()


def test_worker_marks_a_finished_job_done(store, job, monkeypatch):
    def analyze(src, job_dir, options, report):
        assert src == Path(store.get("good")["source_path"])
        assert job_dir == store.dir("good")
        assert store.get("good")["status"] == "running"
        report("asr", 0.5, "转写中")
        assert store.get("good")["stage"] == "asr"

    monkeypatch.setattr(pipeline, "analyze", analyze)
    run_worker(monkeypatch, "good")
    record = store.get("good")
    assert record["status"] == "done"
    assert record["progress"] == 1.0
    assert record["pid"] == os.getpid()


def test_worker_records_a_failed_job_and_keeps_going(store, job, monkeypatch):
    def analyze(src, job_dir, options, report):
        if job_dir == store.dir("good"):
            raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "analyze", analyze)
    run_worker(monkeypatch, "good", "good2")
    assert store.get("good")["status"] == "error"
    assert store.get("good")["error"] == "RuntimeError: boom"
    assert store.get("good2")["status"] == "done"


def test_worker_survives_a_job_with_a_corrupt_record(store, job, monkeypatch):
    (store.dir("good") / "job.json").write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(pipeline, "analyze", lambda *args: None)
    run_worker(monkeypatch, "good", "good2")
    assert store.get("good2")["status"] == "done"


def test_worker_skips_a_job_that_no_longer_exists(store, monkeypatch):
    def analyze(*args):
        raise AssertionError("nothing to analyze")

    monkeypatch.setattr(pipeline, "analyze", analyze)
    run_worker(monkeypatch, "ghost")
