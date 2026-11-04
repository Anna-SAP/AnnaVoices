"""Shared test setup: every test runs sealed off from real data and heavy tools.

Importing ``voiceextract.server`` creates its job store, so ``AV_JOBS_DIR`` must
point at a disposable folder before the first import; otherwise a test run could
touch the real ``jobs/`` folder.
"""
from __future__ import annotations

import os
import queue
import socket
import subprocess
import tempfile
import urllib.request

import pytest

_sandbox = tempfile.TemporaryDirectory(prefix="annavoices-tests-", ignore_cleanup_errors=True)
os.environ["AV_JOBS_DIR"] = _sandbox.name

from fastapi.testclient import TestClient  # noqa: E402

from voiceextract import jobs, server  # noqa: E402

_LOOPBACK = {"127.0.0.1", "::1", "localhost"}  # asyncio's self-pipe uses it on Windows


@pytest.fixture(autouse=True)
def sealed(request, monkeypatch):
    """No network, no model downloads and no child processes (so no ffmpeg)."""
    real_connect = socket.socket.connect

    def connect(self, address):
        if isinstance(address, tuple) and address[0] not in _LOOPBACK:
            raise AssertionError(f"test tried to reach the network: {address}")
        return real_connect(self, address)

    def urlopen(url, *args, **kwargs):
        raise AssertionError(f"test tried to download: {url}")

    def popen(*args, **kwargs):
        raise AssertionError(f"test tried to start a process: {args}")

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    if not request.node.get_closest_marker("allow_subprocess"):
        monkeypatch.setattr(subprocess, "Popen", popen)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private JobStore wired into the server, with an empty analysis queue."""
    s = jobs.JobStore(tmp_path / "jobs")
    monkeypatch.setattr(server, "store", s)
    monkeypatch.setattr(server, "_analysis_q", queue.Queue())
    return s


@pytest.fixture
def client(store):
    """HTTP client; lifespan is not entered, so no analysis worker is running."""
    return TestClient(server.app, raise_server_exceptions=False)
