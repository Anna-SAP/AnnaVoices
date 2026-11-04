"""Checks on the repository itself and on the test sandbox."""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import urllib.request
from pathlib import Path

import pytest

from voiceextract import jobs, models, server

ROOT = Path(__file__).resolve().parent.parent
HEAVY = {"demucs", "torch", "torchaudio", "opencv-python-headless"}
LOCAL_ONLY_DIRS = {"jobs", "models", "output"}
LOCAL_ONLY_SUFFIXES = {
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv",
    ".mp3", ".wav", ".m4a", ".flac", ".aac", ".srt", ".zip",
    ".onnx", ".pt", ".pth", ".npy",
}


def pins(name: str) -> dict[str, str]:
    """{package: version} from a requirements file, ignoring extras and comments."""
    found = {}
    for line in (ROOT / name).read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"([A-Za-z0-9._-]+)(?:\[[^\]]*\])?==(\S+)", line.split("#")[0].strip())
        if m:
            found[m[1].lower()] = m[2]
    return found


# ------------------------------------------------------------------ sandbox checks


def test_tests_never_use_the_real_jobs_folder():
    assert jobs.JOBS_DIR != models.ROOT / "jobs"
    assert models.ROOT not in Path(os.environ["AV_JOBS_DIR"]).parents


def test_server_fixture_replaces_the_store(store):
    assert server.store is store


def test_network_is_blocked():
    with socket.socket() as s, pytest.raises(AssertionError, match="network"):
        s.connect(("192.0.2.1", 80))


def test_downloads_are_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "MODEL_DIR", tmp_path / "models")
    with pytest.raises(AssertionError, match="download"):
        models.ensure_models(["vad"])
    with pytest.raises(AssertionError, match="download"):
        urllib.request.urlopen("https://example.invalid/")


def test_child_processes_are_blocked():
    with pytest.raises(AssertionError, match="process"):
        subprocess.run(["ffmpeg", "-version"])


# -------------------------------------------------------------------- repo content


def test_start_bat_is_crlf_ascii():
    """cmd.exe mangles batch files with LF line endings or non-ASCII bytes."""
    raw = (ROOT / "start.bat").read_bytes()
    assert raw.isascii()
    assert raw.count(b"\n") == raw.count(b"\r\n")


def test_requirements_list_the_same_packages():
    runtime, test = pins("requirements.txt"), pins("requirements-test.txt")
    shared = runtime.keys() & test.keys()
    assert shared >= {"numpy", "scipy", "scikit-learn", "soundfile", "sherpa-onnx", "fastapi", "python-multipart"}
    assert {k: test[k] for k in shared} == {k: runtime[k] for k in shared}


def test_test_requirements_stay_light():
    assert not HEAVY & pins("requirements-test.txt").keys()


def test_every_test_requirement_is_pinned():
    lines = [ln.strip() for ln in (ROOT / "requirements-test.txt").read_text(encoding="utf-8").splitlines()]
    assert len([ln for ln in lines if ln and not ln.startswith("#")]) == len(pins("requirements-test.txt"))


@pytest.mark.allow_subprocess
def test_frontend_script_parses():
    node = shutil.which("node")
    if node is None:
        if os.environ.get("CI"):
            pytest.fail("node is missing on CI, so web/app.js went unchecked")
        pytest.skip("node is not installed")
    subprocess.run([node, "--check", str(ROOT / "web" / "app.js")], check=True)


@pytest.mark.allow_subprocess
def test_no_media_or_model_files_are_tracked():
    """The repo is public: recordings, exports and model weights stay local."""
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True
    ).stdout.splitlines()
    assert tracked, "git ls-files returned nothing"
    assert [f for f in tracked if f.split("/")[0] in LOCAL_ONLY_DIRS] == []
    assert [f for f in tracked if Path(f).suffix.lower() in LOCAL_ONLY_SUFFIXES] == []
