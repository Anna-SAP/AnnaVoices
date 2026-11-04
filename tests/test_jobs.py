"""JobStore and process liveness: the on-disk job record and its ID rules."""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from voiceextract.jobs import JobStore, pid_alive

ILLEGAL_IDS = ["", ".", "..", "../x", "..\\x", "a/b", "a\\b", "a b", "a.b", "a\n", "C:", "%2e%2e", "/etc", "C:\\Windows"]
ONLY_ON_WINDOWS = pytest.mark.skipif(os.name != "nt", reason="device names are only reserved on Windows")
DAMAGED_RECORDS = [b"", b"{not json", b"\xff\xfe", b"null", b"[]", b"123", b"{}"]


@pytest.fixture
def s(tmp_path):
    return JobStore(tmp_path / "jobs")


@pytest.mark.parametrize("job_id", ILLEGAL_IDS)
def test_dir_rejects_illegal_ids(s, job_id):
    with pytest.raises(KeyError):
        s.dir(job_id)


@pytest.mark.parametrize("job_id", ["demo", "20261104-120000-ab12", "任务_一", "a-b_c"])
def test_dir_keeps_legal_ids_directly_under_the_root(s, job_id):
    assert s.dir(job_id) == s.root / job_id


@ONLY_ON_WINDOWS
@pytest.mark.parametrize("job_id", ["CON", "NUL", "AUX", "COM1"])
def test_dir_rejects_windows_device_names(s, job_id):
    with pytest.raises(KeyError):
        s.dir(job_id)


@ONLY_ON_WINDOWS
@pytest.mark.parametrize("job_id", ["con", "Nul", "prn", "COM9", "lpt1", "LPT0", "COM¹", "lpt³"])
def test_dir_rejects_device_names_in_any_case(s, job_id):
    with pytest.raises(KeyError):
        s.dir(job_id)
    assert s.get(job_id) is None


@pytest.mark.parametrize("job_id", ["console", "nul1", "com", "com10", "auxiliary", "CON_1", "lpt-1"])
def test_dir_keeps_names_that_only_resemble_a_device(s, job_id):
    assert s.dir(job_id) == s.root / job_id


@pytest.mark.parametrize("job_id", ILLEGAL_IDS)
def test_get_returns_none_for_illegal_ids(s, job_id):
    assert s.get(job_id) is None


def test_get_returns_none_for_unknown_id(s):
    assert s.get("ghost") is None


def test_create_then_get_round_trips(s):
    job = s.create("采访.mp4", {"separate": False}, job_id="demo")
    assert s.get("demo") == job
    assert job["status"] == "queued"
    assert job["name"] == "采访.mp4"
    assert job["options"] == {"separate": False}
    assert job["source_path"] is None


def test_create_generates_unique_ids(s):
    a, b = s.create("a.mp4", {}), s.create("a.mp4", {})
    assert a["id"] != b["id"]
    assert s.dir(a["id"]).is_dir()


@pytest.mark.parametrize("job_id", ["..", "../x", "a/b"])
def test_create_rejects_illegal_ids_without_writing(s, job_id):
    with pytest.raises(KeyError):
        s.create("a.mp4", {}, job_id=job_id)
    assert list(s.root.iterdir()) == []
    assert sorted(p.name for p in s.root.parent.iterdir()) == ["jobs"]


def test_update_merges_fields_and_leaves_no_temp_file(s):
    s.create("a.mp4", {}, job_id="demo")
    job = s.update("demo", status="running", progress=0.5)
    assert job["status"] == "running"
    assert job["name"] == "a.mp4"
    assert s.get("demo") == job
    assert sorted(p.name for p in s.dir("demo").iterdir()) == ["job.json"]


@pytest.mark.parametrize("job_id", ["..", "../x"])
def test_update_rejects_illegal_ids_without_writing(s, job_id):
    with pytest.raises(KeyError):
        s.update(job_id, status="running")
    assert list(s.root.iterdir()) == []


def test_list_is_empty_for_a_new_store(s):
    assert s.list() == []


def test_list_is_newest_first(s):
    for name, created in (("old", 1.0), ("new", 3.0), ("mid", 2.0)):
        s.create("a.mp4", {}, job_id=name)
        s.update(name, created=created)
    assert [j["id"] for j in s.list()] == ["new", "mid", "old"]


def test_list_skips_corrupt_records_and_folders_without_one(s):
    s.create("a.mp4", {}, job_id="ok")
    s.create("b.mp4", {}, job_id="broken")
    (s.dir("broken") / "job.json").write_text("{not json", encoding="utf-8")
    s.dir("empty").mkdir()
    assert [j["id"] for j in s.list()] == ["ok"]


@ONLY_ON_WINDOWS
def test_create_refuses_a_device_name_without_writing(s):
    with pytest.raises(KeyError):
        s.create("a.mp4", {}, job_id="NUL")
    assert list(s.root.iterdir()) == []


def damage(s, job_id, data):
    s.create("a.mp4", {}, job_id=job_id)
    (s.dir(job_id) / "job.json").write_bytes(data)


@pytest.mark.parametrize("data", DAMAGED_RECORDS)
def test_get_treats_a_damaged_record_as_missing(s, data):
    damage(s, "demo", data)
    assert s.get("demo") is None


@pytest.mark.parametrize("data", DAMAGED_RECORDS)
def test_list_skips_a_damaged_record(s, data):
    damage(s, "broken", data)
    s.create("a.mp4", {}, job_id="ok")
    assert [j["id"] for j in s.list()] == ["ok"]


@pytest.mark.parametrize("data", DAMAGED_RECORDS)
def test_update_does_not_overwrite_a_damaged_record(s, data):
    damage(s, "demo", data)
    with pytest.raises(ValueError):
        s.update("demo", status="done")
    assert (s.dir("demo") / "job.json").read_bytes() == data


def test_records_are_stored_as_readable_utf8(s):
    s.create("采访.mp4", {}, job_id="demo")
    assert json.loads((s.dir("demo") / "job.json").read_text(encoding="utf-8"))["name"] == "采访.mp4"


@pytest.mark.parametrize("pid", [None, 0])
def test_pid_alive_is_false_without_a_pid(pid):
    assert pid_alive(pid) is False


def test_pid_alive_sees_this_process():
    assert pid_alive(os.getpid()) is True


@pytest.mark.allow_subprocess
def test_pid_alive_is_false_after_the_process_exits():
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    assert pid_alive(p.pid) is False
