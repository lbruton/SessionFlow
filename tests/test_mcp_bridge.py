"""SESF-71 tests for the Claude Desktop stdio bridge (``mcp-remote-wait.sh``).

The bridge must wait for the server with no cap, hand off to ``mcp-remote``
once the port answers, and never touch the external volume at runtime.
"""

import http.server
import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
BRIDGE = REPO / "mcp-remote-wait.sh"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_npx(tmp_path):
    """A stand-in ``npx`` that records its argv and exits 0."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    record = tmp_path / "npx-args"
    npx = bin_dir / "npx"
    npx.write_text(f'#!/bin/bash\necho "$@" > "{record}"\n')
    npx.chmod(0o755)
    return bin_dir, record


def _start_bridge(port, bin_dir, **extra_env):
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": os.environ["HOME"],
        "SESSIONFLOW_PORT": str(port),
        # Never kickstart the real LaunchAgent from a test.
        "SESSIONFLOW_AGENT_LABEL": "cc.lbruton.sessionflow.test-nonexistent",
        **extra_env,
    }
    return subprocess.Popen(
        ["/bin/bash", str(BRIDGE)],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _serve(port):
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(503)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_bridge_keeps_waiting_past_old_cap(fake_npx):
    """A refused port never makes the bridge give up and run mcp-remote."""
    bin_dir, record = fake_npx
    proc = _start_bridge(_free_port(), bin_dir, SESSIONFLOW_CONNECT_MAX_WAIT="1")
    try:
        time.sleep(4)
        assert proc.poll() is None, "bridge exited instead of waiting"
        assert not record.exists(), "bridge ran mcp-remote before the port opened"
    finally:
        proc.kill()
        proc.wait()


def test_bridge_execs_mcp_remote_once_port_answers(fake_npx):
    """Once the server answers (even 503 starting), the bridge execs mcp-remote."""
    bin_dir, record = fake_npx
    port = _free_port()
    proc = _start_bridge(port, bin_dir)
    server = None
    try:
        time.sleep(2)
        assert proc.poll() is None
        server = _serve(port)
        assert proc.wait(timeout=15) == 0
        args = record.read_text().split()
        assert args[:3] == ["-y", "mcp-remote", f"http://127.0.0.1:{port}/mcp/"]
    finally:
        if server:
            server.shutdown()
        if proc.poll() is None:
            proc.kill()


def test_bridge_has_no_external_volume_paths():
    """Claude Desktop's bash cannot read /Volumes (TCC), so code must not use it."""
    code = [
        line for line in BRIDGE.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not [line for line in code if re.search(r"/Volumes/", line)]
