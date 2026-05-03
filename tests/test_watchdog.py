"""Tests for shell script changes (Task 2): is_running() and watchdog behavior.

These tests call sessionflow-server.sh functions via subprocess with HOME
overridden to a temp directory, so that SERVER_DIR resolves to tmp/.sessionflow.

Tests will fail because:
1. HEARTBEAT_FILE variable doesn't exist in the current script
2. is_heartbeat_fresh() function doesn't exist
3. is_running() still uses curl (not heartbeat-aware)
"""

import json
import os
import socket
import subprocess
import time

import pytest


def _run_status(script_path, home_dir, port=19999, timeout=10):
    """Run 'sessionflow-server.sh status' with overridden HOME and PORT.

    Returns (exit_code, stderr_output).
    """
    env = os.environ.copy()
    env["HOME"] = str(home_dir)
    env["SESSIONFLOW_PORT"] = str(port)
    result = subprocess.run(
        ["bash", script_path, "status"],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    return result.returncode, result.stderr


def _find_free_port():
    """Find a free TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestIsRunningFreshHeartbeat:
    """is_running() returns 0 when PID alive + heartbeat fresh."""

    def test_is_running_fresh_heartbeat(
        self, tmp_path, mock_pid_file, mock_heartbeat_file, script_path
    ):
        # mock_pid_file and mock_heartbeat_file both live under
        # tmp_path/.sessionflow/ and use the current PID (which is alive).
        exit_code, stderr = _run_status(script_path, tmp_path)
        assert exit_code == 0, (
            f"is_running should return 0 with valid PID + fresh heartbeat. stderr: {stderr}"
        )


class TestIsRunningNoPid:
    """is_running() returns 1 when PID file missing."""

    def test_is_running_no_pid(self, tmp_path, tmp_server_dir, script_path):
        # tmp_server_dir exists (mkdir by fixture) but no PID file written.
        exit_code, stderr = _run_status(script_path, tmp_path)
        assert exit_code == 1, (
            f"is_running should return 1 without PID file. stderr: {stderr}"
        )


class TestIsRunningStaleHeartbeatPortClosed:
    """is_running() returns 1 when heartbeat stale and port not listening."""

    def test_is_running_stale_heartbeat_port_closed(
        self, tmp_path, mock_pid_file, stale_heartbeat_file, script_path
    ):
        # Stale heartbeat (300s old), PID alive, but no port listener.
        # Use a port that nothing is listening on.
        free_port = _find_free_port()
        exit_code, stderr = _run_status(script_path, tmp_path, port=free_port)
        assert exit_code == 1, (
            f"is_running should return 1 with stale heartbeat and closed port. stderr: {stderr}"
        )


class TestIsRunningStaleHeartbeatPortOpen:
    """is_running() returns 0 when heartbeat stale but port still listening (TCP fallback)."""

    def test_is_running_stale_heartbeat_port_open(
        self, tmp_path, mock_pid_file, stale_heartbeat_file, script_path
    ):
        # Start a temporary TCP listener on a random port.
        port = _find_free_port()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        try:
            exit_code, stderr = _run_status(script_path, tmp_path, port=port)
            assert exit_code == 0, (
                f"is_running should return 0 with stale heartbeat but open port (TCP fallback). "
                f"stderr: {stderr}"
            )
        finally:
            sock.close()


class TestWatchdogNoRestartFreshHeartbeat:
    """Watchdog probe succeeds with fresh heartbeat (doesn't log restart)."""

    def test_watchdog_no_restart_fresh_heartbeat(
        self, tmp_path, mock_pid_file, mock_heartbeat_file, script_path, tmp_server_dir
    ):
        # Create the log file so we can check for restart messages.
        log_file = tmp_server_dir / "server.log"
        log_file.write_text("")

        # Run status -- with fresh heartbeat, the server is considered running.
        # The watchdog itself is a background loop, so we test the status check
        # that underpins the watchdog's probe logic.
        exit_code, stderr = _run_status(script_path, tmp_path)

        # If status returns 0 (running), the watchdog would NOT restart.
        assert exit_code == 0, (
            f"Status should return 0 with fresh heartbeat (watchdog would not restart). "
            f"stderr: {stderr}"
        )
        # Verify no restart message in log
        log_content = log_file.read_text()
        assert "restart" not in log_content.lower(), (
            "Log should not contain restart messages with fresh heartbeat"
        )


class TestWatchdogRestartsOnStaleHeartbeatDeadPid:
    """Watchdog restarts when heartbeat stale and PID dead."""

    def test_watchdog_restarts_on_stale_heartbeat_dead_pid(
        self, tmp_path, stale_heartbeat_file, script_path, tmp_server_dir
    ):
        # Write a PID file with a dead PID (PID 99999 is very unlikely to exist).
        pid_file = tmp_server_dir / "server.pid"
        pid_file.write_text("99999")

        # Run status -- with stale heartbeat and dead PID, the server is not running.
        exit_code, stderr = _run_status(script_path, tmp_path)

        # If status returns 1 (not running), the watchdog would detect this and restart.
        assert exit_code == 1, (
            f"Status should return 1 with stale heartbeat + dead PID "
            f"(watchdog would trigger restart). stderr: {stderr}"
        )
