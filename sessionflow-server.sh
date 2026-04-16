#!/bin/bash
# Launcher for the persistent SessionFlow HTTP server.
# Starts the server if not running, verifies health.
# Safe to call multiple times (idempotent).
#
# Usage: ./sessionflow-server.sh [start|stop|status|restart]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$HOME/.sessionflow"
PID_FILE="$SERVER_DIR/server.pid"
WATCHDOG_PID_FILE="$SERVER_DIR/watchdog.pid"
LOG_FILE="$SERVER_DIR/server.log"
PYTHON="$SCRIPT_DIR/venv/bin/python"
PORT="${SESSIONFLOW_PORT:-7102}"
HEALTH_URL="http://127.0.0.1:$PORT/health"
MAX_WAIT=60
WATCHDOG_INTERVAL=30
WATCHDOG_MAX_FAILURES=3

mkdir -p "$SERVER_DIR"

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
                return 0
            fi
            return 1
        else
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

stop_watchdog() {
    if [ -f "$WATCHDOG_PID_FILE" ]; then
        local wpid
        wpid=$(cat "$WATCHDOG_PID_FILE")
        if kill -0 "$wpid" 2>/dev/null; then
            kill "$wpid" 2>/dev/null || true
        fi
        rm -f "$WATCHDOG_PID_FILE"
    fi
}

# start_watchdog starts a background watchdog that regularly checks HEALTH_URL, restarts the server after WATCHDOG_MAX_FAILURES consecutive failures, writes its background PID to WATCHDOG_PID_FILE, and logs actions to LOG_FILE.
start_watchdog() {
    stop_watchdog

    (
        failures=0
        while true; do
            sleep "$WATCHDOG_INTERVAL"

            if curl -sf --max-time 5 "$HEALTH_URL" >/dev/null 2>&1; then
                failures=0
            else
                failures=$((failures + 1))
                echo "[sessionflow-watchdog] Health check failed ($failures/$WATCHDOG_MAX_FAILURES)" >> "$LOG_FILE"
            fi

            if [ "$failures" -ge "$WATCHDOG_MAX_FAILURES" ]; then
                echo "[sessionflow-watchdog] Server unresponsive — restarting..." >> "$LOG_FILE"

                # Kill old server if still alive
                if [ -f "$PID_FILE" ]; then
                    local old_pid
                    old_pid=$(cat "$PID_FILE")
                    kill "$old_pid" 2>/dev/null || true
                    sleep 2
                    kill -9 "$old_pid" 2>/dev/null || true
                    rm -f "$PID_FILE"
                fi

                # Kill stale port holders
                local stale_pids
                stale_pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
                if [ -n "$stale_pids" ]; then
                    echo "$stale_pids" | xargs kill 2>/dev/null || true
                    sleep 1
                fi

                # Restart the server
                export PYTHONPATH="$SCRIPT_DIR"
                export HF_HUB_OFFLINE=1
                export TRANSFORMERS_OFFLINE=1
                [ -n "${SESSIONFLOW_MILVUS_URI:-}" ] && export SESSIONFLOW_MILVUS_URI
                nohup "$PYTHON" -u "$SCRIPT_DIR/http_server.py" >> "$LOG_FILE" 2>&1 &

                # Wait for it to be healthy
                local waited=0
                while [ $waited -lt $MAX_WAIT ]; do
                    sleep 1
                    waited=$((waited + 1))
                    if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
                        echo "[sessionflow-watchdog] Server restarted successfully (${waited}s)" >> "$LOG_FILE"
                        break
                    fi
                done
                failures=0
            fi
        done
    ) &
    echo $! > "$WATCHDOG_PID_FILE"
    echo "[sessionflow] Watchdog started (PID $!)" >&2
}

# do_start starts the SessionFlow HTTP server: it removes stale PID, kills any process holding the configured port, launches the server with required environment variables, waits up to MAX_WAIT seconds for the health endpoint to respond, starts the watchdog on success, and exits non‑zero if the server fails to become healthy.
do_start() {
    if is_running; then
        echo "[sessionflow] Already running (PID $(cat "$PID_FILE"))" >&2
        exit 0
    fi

    rm -f "$PID_FILE"

    # Kill any stale process holding our port
    local stale_pids
    stale_pids=$(lsof -ti :"$PORT" 2>/dev/null || true)
    if [ -n "$stale_pids" ]; then
        echo "[sessionflow] Killing stale process(es) on port $PORT: $stale_pids" >&2
        echo "$stale_pids" | xargs kill 2>/dev/null || true
        sleep 1
    fi

    echo "[sessionflow] Starting HTTP server on port $PORT..." >&2

    export PYTHONPATH="$SCRIPT_DIR"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    # Pass through Milvus URI for Standalone mode
    [ -n "${SESSIONFLOW_MILVUS_URI:-}" ] && export SESSIONFLOW_MILVUS_URI
    nohup "$PYTHON" -u "$SCRIPT_DIR/http_server.py" >> "$LOG_FILE" 2>&1 &
    local server_pid=$!

    local waited=0
    while [ $waited -lt $MAX_WAIT ]; do
        sleep 1
        waited=$((waited + 1))

        if ! kill -0 $server_pid 2>/dev/null; then
            echo "[sessionflow] Server process died. Check $LOG_FILE" >&2
            exit 1
        fi

        if curl -sf --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
            echo "[sessionflow] Server ready (PID $server_pid, ${waited}s)" >&2
            start_watchdog
            exit 0
        fi
    done

    echo "[sessionflow] Server failed to become healthy after ${MAX_WAIT}s. Check $LOG_FILE" >&2
    exit 1
}

# do_stop stops the SessionFlow server and its watchdog, then removes the PID file.
# If a server PID exists it requests termination, waits up to 10 seconds for exit, and sends SIGKILL if still alive; prints status messages to stderr.
do_stop() {
    stop_watchdog

    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            echo "[sessionflow] Stopping server (PID $pid)..." >&2
            kill "$pid"
            local waited=0
            while [ $waited -lt 10 ]; do
                if ! kill -0 "$pid" 2>/dev/null; then
                    break
                fi
                sleep 1
                waited=$((waited + 1))
            done
            if kill -0 "$pid" 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$PID_FILE"
        echo "[sessionflow] Stopped." >&2
    else
        echo "[sessionflow] Not running." >&2
    fi
}

# do_status prints whether the sessionflow server is running to stderr and exits with status 1 if it is not running.
do_status() {
    if is_running; then
        echo "[sessionflow] Running (PID $(cat "$PID_FILE"))" >&2
    else
        echo "[sessionflow] Not running." >&2
        exit 1
    fi
}

case "${1:-start}" in
    start)  do_start ;;
    stop)   do_stop ;;
    status) do_status ;;
    restart)
        do_stop
        do_start
        ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}" >&2
        exit 1
        ;;
esac
