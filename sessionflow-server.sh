#!/bin/bash
# Launcher for the persistent SessionFlow HTTP server.
# Starts the server if not running, verifies health.
# Safe to call multiple times (idempotent).
#
# Usage: ./sessionflow-server.sh [start|stop|status|restart|install-agent|uninstall-agent|agent-status]
#
# Optional macOS LaunchAgent (install-agent / uninstall-agent / agent-status)
# autostarts SessionFlow at login before any harness hooks run. The LaunchAgent
# is OPTIONAL — start/stop/status/restart continue to work standalone.

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
WATCHDOG_STALE_THRESHOLD=120
HEARTBEAT_FILE="$SERVER_DIR/heartbeat"

# Optional LaunchAgent (user scope) — autostart at login before harness hooks.
LAUNCH_AGENT_LABEL="cc.lbruton.sessionflow"
LAUNCH_AGENT_DIR="$HOME/Library/LaunchAgents"
LAUNCH_AGENT_PLIST="$LAUNCH_AGENT_DIR/${LAUNCH_AGENT_LABEL}.plist"
LAUNCH_AGENT_STDOUT="$SERVER_DIR/launchagent.out.log"
LAUNCH_AGENT_STDERR="$SERVER_DIR/launchagent.err.log"
LAUNCHER_SCRIPT="$SERVER_DIR/sessionflow-launcher.sh"

# SESF-7: hourly backfill agent — independent of the persistent server agent.
# Decouples indexing throughput from MCP session lifecycle.
BACKFILL_AGENT_LABEL="cc.lbruton.sessionflow-backfill"
BACKFILL_AGENT_PLIST="$LAUNCH_AGENT_DIR/${BACKFILL_AGENT_LABEL}.plist"
BACKFILL_AGENT_STDOUT="$SERVER_DIR/backfill-agent.out.log"
BACKFILL_AGENT_STDERR="$SERVER_DIR/backfill-agent.err.log"
BACKFILL_INTERVAL_SECONDS="${SESSIONFLOW_BACKFILL_INTERVAL_SECONDS:-3600}"

mkdir -p "$SERVER_DIR"

# is_heartbeat_fresh determines whether HEARTBEAT_FILE and PID_FILE exist and whether the heartbeat's `timestamp` is within WATCHDOG_STALE_THRESHOLD seconds and its `pid` matches the PID stored in PID_FILE. Exits with status 0 if the heartbeat is fresh and matches the expected PID, non-zero otherwise.
is_heartbeat_fresh() {
    if [ ! -f "$HEARTBEAT_FILE" ] || [ ! -f "$PID_FILE" ]; then
        return 1
    fi
    local expected_pid
    expected_pid=$(cat "$PID_FILE")
    "$PYTHON" -c "
import json, time, sys
try:
    h = json.load(open(sys.argv[1]))
    age_ok = (time.time() - h['timestamp']) < float(sys.argv[2])
    pid_ok = str(h.get('pid')) == sys.argv[3]
    sys.exit(0 if age_ok and pid_ok else 1)
except Exception:
    sys.exit(1)
" "$HEARTBEAT_FILE" "$WATCHDOG_STALE_THRESHOLD" "$expected_pid"
}

# is_running checks whether the server recorded in PID_FILE is alive and either has a fresh heartbeat or accepts TCP connections on the configured PORT.
is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid
        pid=$(cat "$PID_FILE")
        if kill -0 "$pid" 2>/dev/null; then
            # Process alive — check heartbeat or TCP fallback
            if is_heartbeat_fresh; then
                return 0
            fi
            # TCP fallback: prevents double-start during startup before first heartbeat.
            # Intentionally more lenient than the watchdog (which uses heartbeat-only).
            if (echo >/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
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

# start_watchdog launches a background watchdog that monitors heartbeat freshness and, after repeated stale detections, kills and restarts the server process and any stale port holders.
start_watchdog() {
    stop_watchdog

    (
        failures=0
        while true; do
            sleep "$WATCHDOG_INTERVAL"

            if is_heartbeat_fresh; then
                failures=0
            else
                # Heartbeat stale — check if process is actually dead
                if [ -f "$PID_FILE" ]; then
                    local pid
                    pid=$(cat "$PID_FILE")
                    if kill -0 "$pid" 2>/dev/null; then
                        # Process alive but heartbeat stale — suspect hang
                        failures=$((failures + 1))
                        echo "[sessionflow-watchdog] Heartbeat stale, process alive — suspect hang ($failures/$WATCHDOG_MAX_FAILURES)" >> "$LOG_FILE"
                    else
                        # Process dead
                        failures=$((failures + 1))
                        echo "[sessionflow-watchdog] Heartbeat stale, process dead ($failures/$WATCHDOG_MAX_FAILURES)" >> "$LOG_FILE"
                    fi
                else
                    failures=$((failures + 1))
                    echo "[sessionflow-watchdog] No PID file ($failures/$WATCHDOG_MAX_FAILURES)" >> "$LOG_FILE"
                fi
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

                # Wait for it to be healthy (HTTP check is fine during startup — no embedding load)
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

do_status() {
    if is_running; then
        echo "[sessionflow] Running (PID $(cat "$PID_FILE"))" >&2
    else
        echo "[sessionflow] Not running." >&2
        exit 1
    fi
}

# --- Optional LaunchAgent (user scope) ---

# write_user_launcher creates a home-directory launcher for hook/launchd
# contexts that cannot execute scripts directly from /Volumes.
write_user_launcher() {
    cat > "$LAUNCHER_SCRIPT" <<LAUNCHER
#!/bin/bash
set -euo pipefail

REPO_DIR="$SCRIPT_DIR"
SERVER_DIR="\$HOME/.sessionflow"
PID_FILE="\$SERVER_DIR/server.pid"
LOG_FILE="\$SERVER_DIR/server.log"
PYTHON="\$REPO_DIR/venv/bin/python"
PORT="\${SESSIONFLOW_PORT:-7102}"
HEALTH_URL="http://127.0.0.1:\$PORT/health"
MAX_WAIT="\${SESSIONFLOW_START_MAX_WAIT:-60}"
LAUNCH_AGENT_LABEL="$LAUNCH_AGENT_LABEL"
LAUNCH_AGENT_PLIST="\$HOME/Library/LaunchAgents/\${LAUNCH_AGENT_LABEL}.plist"
SESSIONFLOW_MILVUS_URI="\${SESSIONFLOW_MILVUS_URI:-http://192.168.1.83:19530}"

mkdir -p "\$SERVER_DIR"

is_healthy() {
    curl -sf --max-time 5 "\$HEALTH_URL" >/dev/null 2>&1
}

is_pid_alive() {
    [ -f "\$PID_FILE" ] && kill -0 "\$(cat "\$PID_FILE")" 2>/dev/null
}

wait_for_health() {
    local server_pid="\${1:-}"
    local waited=0
    while [ "\$waited" -lt "\$MAX_WAIT" ]; do
        sleep 1
        waited=\$((waited + 1))

        if [ -n "\$server_pid" ] && ! kill -0 "\$server_pid" 2>/dev/null; then
            echo "[sessionflow-launcher] Server process died. Check \$LOG_FILE" >&2
            exit 1
        fi

        if is_healthy; then
            echo "[sessionflow-launcher] Server ready (\${waited}s)" >&2
            exit 0
        fi
    done

    echo "[sessionflow-launcher] Server failed to become healthy after \${MAX_WAIT}s. Check \$LOG_FILE" >&2
    exit 1
}

start_server() {
    if is_healthy; then
        echo "[sessionflow-launcher] Already healthy on port \$PORT" >&2
        exit 0
    fi

    if is_pid_alive; then
        echo "[sessionflow-launcher] PID file exists but health failed; waiting for process \$(cat "\$PID_FILE") to become healthy..." >&2
        wait_for_health "\$(cat "\$PID_FILE")"
    fi

    if [ -f "\$LAUNCH_AGENT_PLIST" ] && launchctl print "gui/\$(id -u)/\${LAUNCH_AGENT_LABEL}" >/dev/null 2>&1; then
        launchctl kickstart -k "gui/\$(id -u)/\${LAUNCH_AGENT_LABEL}" >/dev/null 2>&1 || true
        wait_for_health
        exit \$?
    fi

    export PYTHONPATH="\$REPO_DIR"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export SESSIONFLOW_MILVUS_URI

    cd "\$HOME"
    nohup "\$PYTHON" -u "\$REPO_DIR/http_server.py" >> "\$LOG_FILE" 2>&1 &
    wait_for_health "\$!"
}

run_server() {
    export PYTHONPATH="\$REPO_DIR"
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export SESSIONFLOW_MILVUS_URI

    cd "\$HOME"
    exec "\$PYTHON" -u "\$REPO_DIR/http_server.py" >> "\$LOG_FILE" 2>&1
}

case "\${1:-start}" in
    start)
        start_server
        ;;
    run)
        run_server
        ;;
    status)
        if is_healthy; then
            echo "[sessionflow-launcher] Healthy on port \$PORT" >&2
        else
            echo "[sessionflow-launcher] Not healthy on port \$PORT" >&2
            exit 1
        fi
        ;;
    *)
        echo "Usage: \$0 [start|run|status]" >&2
        exit 2
        ;;
esac
LAUNCHER
    chmod +x "$LAUNCHER_SCRIPT"
}

# write_launch_agent_plist creates the user LaunchAgent plist that launches the
# home-directory launcher in foreground mode at login.
write_launch_agent_plist() {
    mkdir -p "$LAUNCH_AGENT_DIR"
    write_user_launcher
    cat > "$LAUNCH_AGENT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTD/PropertyLists-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${LAUNCH_AGENT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>${LAUNCHER_SCRIPT}</string>
        <string>run</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <!-- EnvironmentVariables: HOME + PATH only. Custom SessionFlow env
         (SESSIONFLOW_MILVUS_URI, SESSIONFLOW_PORT, etc.) is NOT propagated
         here because launchd does not inherit shell env. To set them under
         the LaunchAgent: either (a) run \`launchctl setenv <KEY> <VAL>\` once
         before \`install-agent\`, or (b) hand-edit this plist and add the
         keys below the PATH entry. -->
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>StandardOutPath</key>
    <string>${LAUNCH_AGENT_STDOUT}</string>
    <key>StandardErrorPath</key>
    <string>${LAUNCH_AGENT_STDERR}</string>
    <key>WorkingDirectory</key>
    <string>${HOME}</string>
</dict>
</plist>
PLIST
}

# launch_agent_target prints the launchctl gui/<uid> domain target for the
# current user, e.g. "gui/501". Used by bootstrap/bootout/print.
launch_agent_target() {
    echo "gui/$(id -u)"
}

do_install_agent() {
    if [ ! -x "$SCRIPT_DIR/sessionflow-server.sh" ]; then
        chmod +x "$SCRIPT_DIR/sessionflow-server.sh"
    fi
    write_user_launcher
    local target
    target=$(launch_agent_target)
    # If already bootstrapped, bootout first so the new plist takes effect.
    if launchctl print "${target}/${LAUNCH_AGENT_LABEL}" >/dev/null 2>&1; then
        launchctl bootout "$target" "$LAUNCH_AGENT_PLIST" 2>/dev/null || true
    fi
    write_launch_agent_plist
    if launchctl bootstrap "$target" "$LAUNCH_AGENT_PLIST" 2>/dev/null; then
        echo "[sessionflow] LaunchAgent installed at $LAUNCH_AGENT_PLIST" >&2
        echo "[sessionflow] Bootstrapped into $target" >&2
    else
        # bootstrap can fail with "Bootstrap failed: 5: Input/output error" if
        # the domain already has the label; fall back to enable + kickstart.
        launchctl enable "${target}/${LAUNCH_AGENT_LABEL}" 2>/dev/null || true
        launchctl kickstart -k "${target}/${LAUNCH_AGENT_LABEL}" 2>/dev/null || true
        echo "[sessionflow] LaunchAgent written at $LAUNCH_AGENT_PLIST" >&2
        echo "[sessionflow] launchctl bootstrap reported a soft failure; tried enable+kickstart." >&2
        echo "[sessionflow] Verify with: $0 agent-status" >&2
    fi
}

do_uninstall_agent() {
    local target
    target=$(launch_agent_target)
    if [ -f "$LAUNCH_AGENT_PLIST" ]; then
        launchctl bootout "$target" "$LAUNCH_AGENT_PLIST" 2>/dev/null || true
        rm -f "$LAUNCH_AGENT_PLIST"
        echo "[sessionflow] LaunchAgent removed ($LAUNCH_AGENT_PLIST)" >&2
    else
        # Best-effort bootout by label in case the file was deleted manually.
        launchctl bootout "${target}/${LAUNCH_AGENT_LABEL}" 2>/dev/null || true
        echo "[sessionflow] No LaunchAgent plist found at $LAUNCH_AGENT_PLIST" >&2
    fi
}

write_backfill_agent_plist() {
    mkdir -p "$LAUNCH_AGENT_DIR"
    local port="${SESSIONFLOW_PORT:-7102}"
    cat > "$BACKFILL_AGENT_PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTD/PropertyLists-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>${BACKFILL_AGENT_LABEL}</string>
    <!-- Calls the running server's /backfill endpoint so the drain reuses
         the server's warmed-up MLX executor. Avoids spawning a parallel
         MLX/Metal process every hour. If the server is down the curl
         simply fails (non-fatal) and the next tick retries. -->
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/curl</string>
        <string>--silent</string>
        <string>--show-error</string>
        <string>--fail</string>
        <string>--max-time</string>
        <string>1800</string>
        <string>-X</string>
        <string>POST</string>
        <string>-H</string>
        <string>Content-Type: application/json</string>
        <string>-d</string>
        <string>{"action":"run","mode":"incremental"}</string>
        <string>http://127.0.0.1:${port}/backfill</string>
    </array>
    <key>StartInterval</key>
    <integer>${BACKFILL_INTERVAL_SECONDS}</integer>
    <key>RunAtLoad</key>
    <false/>
    <key>EnvironmentVariables</key>
    <dict>
        <key>HOME</key>
        <string>${HOME}</string>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>StandardOutPath</key>
    <string>${BACKFILL_AGENT_STDOUT}</string>
    <key>StandardErrorPath</key>
    <string>${BACKFILL_AGENT_STDERR}</string>
    <key>WorkingDirectory</key>
    <string>${SCRIPT_DIR}</string>
</dict>
</plist>
PLIST
}

do_install_backfill_agent() {
    local target
    target=$(launch_agent_target)
    if launchctl print "${target}/${BACKFILL_AGENT_LABEL}" >/dev/null 2>&1; then
        launchctl bootout "$target" "$BACKFILL_AGENT_PLIST" 2>/dev/null || true
    fi
    write_backfill_agent_plist
    if launchctl bootstrap "$target" "$BACKFILL_AGENT_PLIST" 2>/dev/null; then
        echo "[sessionflow] Backfill LaunchAgent installed at $BACKFILL_AGENT_PLIST" >&2
        echo "[sessionflow] Bootstrapped into $target (interval=${BACKFILL_INTERVAL_SECONDS}s)" >&2
    else
        launchctl enable "${target}/${BACKFILL_AGENT_LABEL}" 2>/dev/null || true
        echo "[sessionflow] Backfill plist written at $BACKFILL_AGENT_PLIST" >&2
        echo "[sessionflow] launchctl bootstrap reported a soft failure; tried enable." >&2
    fi
}

do_uninstall_backfill_agent() {
    local target
    target=$(launch_agent_target)
    if [ -f "$BACKFILL_AGENT_PLIST" ]; then
        launchctl bootout "$target" "$BACKFILL_AGENT_PLIST" 2>/dev/null || true
        rm -f "$BACKFILL_AGENT_PLIST"
        echo "[sessionflow] Backfill LaunchAgent removed ($BACKFILL_AGENT_PLIST)" >&2
    else
        launchctl bootout "${target}/${BACKFILL_AGENT_LABEL}" 2>/dev/null || true
        echo "[sessionflow] No backfill plist found at $BACKFILL_AGENT_PLIST" >&2
    fi
}

do_backfill_agent_status() {
    local target
    target=$(launch_agent_target)
    if [ -f "$BACKFILL_AGENT_PLIST" ]; then
        echo "[sessionflow] Backfill plist: $BACKFILL_AGENT_PLIST (present)" >&2
    else
        echo "[sessionflow] Backfill plist: $BACKFILL_AGENT_PLIST (missing)" >&2
    fi
    if launchctl print "${target}/${BACKFILL_AGENT_LABEL}" >/dev/null 2>&1; then
        echo "[sessionflow] Backfill LaunchAgent: loaded in ${target}" >&2
        launchctl print "${target}/${BACKFILL_AGENT_LABEL}" \
            | grep -E '^[[:space:]]*(state|last exit code|pid)[[:space:]]*=' \
            | sed 's/^[[:space:]]*/  /' >&2 || true
        exit 0
    else
        echo "[sessionflow] Backfill LaunchAgent: not loaded in ${target}" >&2
        exit 1
    fi
}

do_agent_status() {
    local target
    target=$(launch_agent_target)
    if [ -f "$LAUNCH_AGENT_PLIST" ]; then
        echo "[sessionflow] Plist: $LAUNCH_AGENT_PLIST (present)" >&2
    else
        echo "[sessionflow] Plist: $LAUNCH_AGENT_PLIST (missing)" >&2
    fi
    if launchctl print "${target}/${LAUNCH_AGENT_LABEL}" >/dev/null 2>&1; then
        echo "[sessionflow] LaunchAgent: loaded in ${target}" >&2
        # Surface the most useful bits (state + last exit code) without dumping
        # the entire print payload.
        launchctl print "${target}/${LAUNCH_AGENT_LABEL}" \
            | grep -E '^[[:space:]]*(state|last exit code|pid)[[:space:]]*=' \
            | sed 's/^[[:space:]]*/  /' >&2 || true
        exit 0
    else
        echo "[sessionflow] LaunchAgent: not loaded in ${target}" >&2
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
    install-agent)   do_install_agent ;;
    uninstall-agent) do_uninstall_agent ;;
    agent-status)    do_agent_status ;;
    install-backfill-agent)   do_install_backfill_agent ;;
    uninstall-backfill-agent) do_uninstall_backfill_agent ;;
    backfill-agent-status)    do_backfill_agent_status ;;
    *)
        echo "Usage: $0 {start|stop|status|restart|install-agent|uninstall-agent|agent-status|install-backfill-agent|uninstall-backfill-agent|backfill-agent-status}" >&2
        exit 1
        ;;
esac
