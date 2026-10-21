#!/bin/bash
# Stdio bridge for MCP clients that cannot speak HTTP directly (Claude Desktop).
#
# Waits until the SessionFlow server answers on its port, then execs
# `npx -y mcp-remote <url>`. Without the wait, a client launched at login can
# start mcp-remote before the server listens; mcp-remote exits on the first
# ECONNREFUSED and the client does not retry (SESF-70).
#
# There is no wait cap: the bridge never gives up on its own (SESF-71). If the
# client's initialize timeout fires first, the client kills it and launches a
# fresh one, which keeps waiting. Progress goes to stderr only (the client's
# MCP log), rate-limited, so nothing reaches the user.
#
# Only a refused connection (curl exit 7) means "not up yet". Any HTTP
# response, including 503 "starting", or a slow /health that times out, means
# the port is open; the MCP handshake does not need the model loaded.
#
# While the port stays refused, the bridge nudges the SessionFlow LaunchAgent
# with `launchctl kickstart` (no -k, so a running server is left alone).
#
# INSTALL AS A COPY ON THE INTERNAL DISK, not a path into this repo:
#   ./sessionflow-server.sh install-mcp-bridge   # -> ~/.local/bin/sessionflow-mcp-remote
# Claude Desktop spawns MCP servers through Claude.app/Contents/Helpers/disclaimer,
# which makes the child its own TCC-responsible process; /bin/bash then cannot
# read anything on an external volume (/Volumes/DATA) and fails with
# "Operation not permitted" before the first line runs (SESF-71, same as
# DEVS-105 for mcp-launch). For the same reason this script must not touch the
# repo at runtime. Re-run the install after every change to this file.
#
# Claude Desktop config:
#   "sessionflow": {
#     "command": "/Users/<you>/.local/bin/sessionflow-mcp-remote"
#   }
#
# Env: SESSIONFLOW_PORT (default 7102), SESSIONFLOW_AGENT_LABEL (default
# cc.lbruton.sessionflow).

PORT="${SESSIONFLOW_PORT:-7102}"
AGENT_LABEL="${SESSIONFLOW_AGENT_LABEL:-cc.lbruton.sessionflow}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
MCP_URL="http://127.0.0.1:${PORT}/mcp/"
KICK_AFTER=10       # seconds refused before the first LaunchAgent nudge
KICK_EVERY=60       # seconds between nudges
LOG_EVERY=60        # seconds between "still waiting" lines

# Login-time launches get a minimal PATH; make sure npx resolves. Appended so
# an npx already on PATH wins.
export PATH="${PATH}:/opt/homebrew/bin:/usr/local/bin"

log() {
    echo "[sessionflow-mcp-remote] $*" >&2
}

port_open() {
    curl -s -o /dev/null --max-time 2 "$HEALTH_URL"
    [ "$?" -ne 7 ]
}

kick_agent() {
    local target
    target="gui/$(id -u)/${AGENT_LABEL}"
    if launchctl print "$target" >/dev/null 2>&1; then
        launchctl kickstart "$target" >/dev/null 2>&1 || true
        log "nudged LaunchAgent ${AGENT_LABEL}"
    fi
}

start=$SECONDS
next_log=$LOG_EVERY
next_kick=$KICK_AFTER
until port_open; do
    elapsed=$((SECONDS - start))
    if [ "$elapsed" -ge "$next_kick" ]; then
        kick_agent
        next_kick=$((elapsed + KICK_EVERY))
    fi
    if [ "$elapsed" -ge "$next_log" ]; then
        log "still waiting for SessionFlow on port ${PORT} (${elapsed}s)"
        next_log=$((elapsed + LOG_EVERY))
    fi
    # Poll every second for the first minute, then every 5s.
    if [ "$elapsed" -lt 60 ]; then sleep 1; else sleep 5; fi
done

elapsed=$((SECONDS - start))
if [ "$elapsed" -gt 0 ]; then
    log "SessionFlow answered after ${elapsed}s"
fi

exec npx -y mcp-remote "$MCP_URL" "$@"
