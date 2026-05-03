"""Tests for the HeartbeatThread class (to be implemented in http_server.py).

All tests will fail with ImportError until Task 1 adds HeartbeatThread.
This is the expected TDD red-phase behavior.
"""

import json
import os
import time

import pytest


def _get_heartbeat_thread():
    """Import and return HeartbeatThread; raises ImportError until Task 1."""
    from http_server import HeartbeatThread
    return HeartbeatThread


class TestHeartbeatWritesFile:
    """HeartbeatThread creates heartbeat file after start."""

    def test_heartbeat_writes_file(self, tmp_server_dir):
        HeartbeatThread = _get_heartbeat_thread()
        hb_path = tmp_server_dir / "heartbeat"
        thread = HeartbeatThread(path=hb_path, interval=0.5)
        thread.start()
        try:
            time.sleep(1.0)
            assert hb_path.exists(), "Heartbeat file should exist after start"
        finally:
            thread.stop()


class TestHeartbeatUpdatesTimestamp:
    """Timestamp advances on each write."""

    def test_heartbeat_updates_timestamp(self, tmp_server_dir):
        HeartbeatThread = _get_heartbeat_thread()
        hb_path = tmp_server_dir / "heartbeat"
        thread = HeartbeatThread(path=hb_path, interval=0.5)
        thread.start()
        try:
            time.sleep(0.8)
            first = json.loads(hb_path.read_text())["timestamp"]
            time.sleep(1.0)
            second = json.loads(hb_path.read_text())["timestamp"]
            assert second > first, "Timestamp should advance between heartbeats"
        finally:
            thread.stop()


class TestHeartbeatAtomicWrite:
    """File is never partially written (check via concurrent reads)."""

    def test_heartbeat_atomic_write(self, tmp_server_dir):
        HeartbeatThread = _get_heartbeat_thread()
        hb_path = tmp_server_dir / "heartbeat"
        thread = HeartbeatThread(path=hb_path, interval=0.5)
        thread.start()
        try:
            time.sleep(0.8)
            for _ in range(10):
                content = hb_path.read_text()
                data = json.loads(content)  # Must not raise JSONDecodeError
                assert "timestamp" in data
        finally:
            thread.stop()


class TestHeartbeatIncludesPid:
    """Heartbeat JSON includes correct PID."""

    def test_heartbeat_includes_pid(self, tmp_server_dir):
        HeartbeatThread = _get_heartbeat_thread()
        hb_path = tmp_server_dir / "heartbeat"
        thread = HeartbeatThread(path=hb_path, interval=0.5)
        thread.start()
        try:
            time.sleep(1.0)
            data = json.loads(hb_path.read_text())
            assert data["pid"] == os.getpid(), "Heartbeat PID should match current process"
        finally:
            thread.stop()


class TestHeartbeatIncludesActivity:
    """Heartbeat JSON includes activity field."""

    def test_heartbeat_includes_activity(self, tmp_server_dir):
        HeartbeatThread = _get_heartbeat_thread()
        hb_path = tmp_server_dir / "heartbeat"
        thread = HeartbeatThread(path=hb_path, interval=0.5)
        thread.start()
        try:
            time.sleep(1.0)
            data = json.loads(hb_path.read_text())
            assert "activity" in data, "Heartbeat should include activity field"
            assert data["activity"] in ("idle", "processing"), (
                f"Activity should be 'idle' or 'processing', got {data['activity']!r}"
            )
        finally:
            thread.stop()


class TestHeartbeatStopsCleanly:
    """stop() causes thread to exit within 2 seconds."""

    def test_heartbeat_stops_cleanly(self, tmp_server_dir):
        HeartbeatThread = _get_heartbeat_thread()
        hb_path = tmp_server_dir / "heartbeat"
        thread = HeartbeatThread(path=hb_path, interval=0.5)
        thread.start()
        time.sleep(0.5)
        thread.stop()
        # The internal thread should no longer be alive
        time.sleep(0.5)
        assert not thread._thread.is_alive(), (
            "Thread should be dead within 2 seconds of stop()"
        )
