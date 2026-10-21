#!/bin/bash
# Stdio bridge for MCP clients that cannot speak HTTP directly (Claude Desktop).
#
# Waits until the SessionFlow server answers on its port, then execs
# `npx -y mcp-remote <url>`. Without the wait, a client launched at login can
# start mcp-remote before the server listens; mcp-remote exits on the first
# ECONNREFUSED and the client does not retry (SESF-70).
#
# Only a refused connection (curl exit 7) means "not up yet". Any HTTP
# response, including 503 "starting", or a slow /health that times out, means
# the port is open; the MCP handshake does not need the model loaded.
#
# Claude Desktop config:
#   "sessionflow": {
#     "command": "/Volumes/DATA/GitHub/SessionFlow/mcp-remote-wait.sh"
#   }
#
# Env: SESSIONFLOW_PORT (default 7102), SESSIONFLOW_CONNECT_MAX_WAIT seconds
# (default 45; keep it under the client's 60s initialize timeout).

PORT="${SESSIONFLOW_PORT:-7102}"
MAX_WAIT="${SESSIONFLOW_CONNECT_MAX_WAIT:-45}"
HEALTH_URL="http://127.0.0.1:${PORT}/health"
MCP_URL="http://127.0.0.1:${PORT}/mcp/"

# Login-time launches get a minimal PATH; make sure npx resolves.
export PATH="/opt/homebrew/bin:/usr/local/bin:${PATH}"

port_open() {
    curl -s -o /dev/null --max-time 2 "$HEALTH_URL"
    [ "$?" -ne 7 ]
}

waited=0
until port_open; do
    if [ "$waited" -ge "$MAX_WAIT" ]; then
        echo "[mcp-remote-wait] SessionFlow not answering on port $PORT after ${MAX_WAIT}s; starting mcp-remote anyway" >&2
        break
    fi
    sleep 1
    waited=$((waited + 1))
done

if [ "$waited" -gt 0 ] && [ "$waited" -lt "$MAX_WAIT" ]; then
    echo "[mcp-remote-wait] SessionFlow answered after ${waited}s" >&2
fi

exec npx -y mcp-remote "$MCP_URL" "$@"
