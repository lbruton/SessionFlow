"""Shared fixtures for SessionFlow tests."""

import json
import os
import sys
import time
from pathlib import Path

import pytest

# Add project root to sys.path so `import http_server` works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def tmp_server_dir(tmp_path):
    """Create a temporary directory mimicking ~/.sessionflow/."""
    server_dir = tmp_path / ".sessionflow"
    server_dir.mkdir()
    return server_dir


@pytest.fixture
def mock_heartbeat_file(tmp_server_dir):
    """Write a fresh heartbeat JSON file."""
    hb_path = tmp_server_dir / "heartbeat"
    data = {
        "timestamp": time.time(),
        "pid": os.getpid(),
        "activity": "idle",
    }
    hb_path.write_text(json.dumps(data))
    return hb_path


@pytest.fixture
def stale_heartbeat_file(tmp_server_dir):
    """Write a heartbeat with timestamp 300 seconds in the past."""
    hb_path = tmp_server_dir / "heartbeat"
    data = {
        "timestamp": time.time() - 300,
        "pid": os.getpid(),
        "activity": "idle",
    }
    hb_path.write_text(json.dumps(data))
    return hb_path


@pytest.fixture
def mock_pid_file(tmp_server_dir):
    """Write a PID file with current PID."""
    pid_path = tmp_server_dir / "server.pid"
    pid_path.write_text(str(os.getpid()))
    return pid_path


@pytest.fixture
def script_path():
    """Return the absolute path to sessionflow-server.sh."""
    return str(Path(__file__).resolve().parent.parent / "sessionflow-server.sh")
