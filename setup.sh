#!/bin/bash
# Setup script for SessionFlow on Apple Silicon Mac.
# Single command to install everything: venv, deps, model, and verify.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "======================================================================"
echo "SessionFlow Setup for Apple Silicon Mac"
echo "======================================================================"
echo ""

# --- Check prerequisites ---
echo "Checking prerequisites..."

if [[ $(uname) != "Darwin" ]]; then
    echo "Error: This script is for macOS only"
    exit 1
fi

if [[ $(uname -m) != "arm64" ]]; then
    echo "Error: This script requires Apple Silicon (M1/M2/M3/M4)"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Install with: brew install python@3.12"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR_MINOR=$(echo "$PYTHON_VERSION" | cut -d'.' -f1,2)
echo "  Python $PYTHON_VERSION"

MACOS_VERSION=$(sw_vers -productVersion | cut -d'.' -f1)
echo "  macOS $MACOS_VERSION"
echo ""

# --- Create virtual environment ---
echo "Creating virtual environment..."
if [ -d "venv" ]; then
    echo "  venv already exists, using existing"
else
    python3 -m venv venv
    echo "  venv created"
fi

source venv/bin/activate

echo ""
echo "Installing dependencies..."
echo "  This may take 1-2 minutes on first run..."

pip install --quiet --upgrade pip

echo "  Installing Milvus Lite..."
pip install --quiet "setuptools>=70.0,<82.0" "pymilvus[milvus-lite]>=2.6.0"

echo "  Installing MLX embeddings and the reranker runtime..."
pip install --quiet "mlx>=0.30.0" mlx-embeddings "mlx-lm>=0.29" "transformers<5.0"

echo "  Installing MCP + HTTP server..."
pip install --quiet "mcp>=1.0.0,<2.0" starlette uvicorn httpx "watchdog>=4.0.0"

echo "  All dependencies installed"

# --- Download model ---
echo ""
SESSIONFLOW_MODEL="${SESSIONFLOW_MODEL:-embeddinggemma}"
echo "Downloading embedding model ($SESSIONFLOW_MODEL)..."
chmod +x "$SCRIPT_DIR/download-model.sh"
SESSIONFLOW_MODEL="$SESSIONFLOW_MODEL" "$SCRIPT_DIR/download-model.sh"

# Normalized like reranker.settings: case and surrounding spaces do not matter.
RERANKER=$(echo "${SESSIONFLOW_RERANKER:-qwen3}" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')
if [ "$RERANKER" != "off" ]; then
    echo "Downloading reranker ($RERANKER)..."
    "$SCRIPT_DIR/download-model.sh" "$RERANKER"
fi

# --- Test installation ---
echo ""
echo "Testing installation..."

SESSIONFLOW_MODEL="$SESSIONFLOW_MODEL" python3 << 'PYEOF'
import sys
sys.path.insert(0, '.')

try:
    import rag_engine
    import transcript_parser
    import file_watcher
    import tools
    print("  All imports successful")
except Exception as e:
    print(f"  Import failed: {e}")
    sys.exit(1)

try:
    emb = rag_engine.embed_texts(["test embedding"])
    dim = len(emb[0])
    expected = rag_engine._EMBED_DIM
    assert dim == expected, f"Expected {expected} dims, got {dim}"
    print(f"  Embedding works ({dim} dimensions, model: {rag_engine.get_model_name()})")
except Exception as e:
    print(f"  Embedding test failed: {e}")
    sys.exit(1)

print("")
print("  All tests passed!")
PYEOF

if [ $? -ne 0 ]; then
    echo "Installation test failed"
    exit 1
fi

# --- Make scripts executable ---
chmod +x "$SCRIPT_DIR/sessionflow-server.sh"
chmod +x "$SCRIPT_DIR/download-model.sh"
chmod +x "$SCRIPT_DIR/index_hook.py"

# --- Install hooks into ~/.claude/settings.json ---
echo ""
echo "Installing hooks into ~/.claude/settings.json..."

VENV_PYTHON="$SCRIPT_DIR/venv/bin/python"
SETTINGS_FILE="$HOME/.claude/settings.json"

python3 << PYEOF
import json
import os
import sys

settings_file = "$SETTINGS_FILE"
script_dir = "$SCRIPT_DIR"
venv_python = "$VENV_PYTHON"
launcher = os.path.expanduser("~/.sessionflow/sessionflow-launcher.sh")

# Our hooks to install
our_hooks = {
    "SessionStart": [
        {
            "type": "command",
            "command": f"{launcher} start",
            "timeout": 30000,
        },
        {
            "type": "command",
            "command": f"{script_dir}/session_start_hook.sh",
            "timeout": 5000,
        },
    ],
    "Stop": [
        {
            "type": "command",
            "command": f"{venv_python} {script_dir}/index_hook.py",
            "timeout": 15000,
        },
    ],
    "PreCompact": [
        {
            "type": "command",
            "command": f"{venv_python} {script_dir}/index_hook.py",
            "timeout": 30000,
        },
    ],
}

# Load existing settings
os.makedirs(os.path.dirname(settings_file), exist_ok=True)
if os.path.exists(settings_file):
    with open(settings_file) as f:
        settings = json.load(f)
else:
    settings = {}

hooks = settings.setdefault("hooks", {})

for event, new_hook_entries in our_hooks.items():
    # Get or create the event's hook groups array
    groups = hooks.setdefault(event, [])

    # Find or create a hook group (we use the first group)
    if not groups:
        groups.append({"hooks": []})
    group = groups[0]
    existing = group.setdefault("hooks", [])

    # Collect existing commands to avoid duplicates
    existing_commands = {h.get("command", "") for h in existing}

    added = 0
    for hook in new_hook_entries:
        # Check if already installed (match by command containing our script dir)
        cmd = hook["command"]
        if cmd not in existing_commands:
            # Also check if an older version exists (same script name, different path)
            script_name = os.path.basename(cmd.split()[-1])
            replaced = False
            for i, eh in enumerate(existing):
                if script_name in eh.get("command", ""):
                    existing[i] = hook
                    replaced = True
                    break
            if not replaced:
                existing.append(hook)
            added += 1

    if added:
        print(f"  {event}: {added} hook(s) installed")
    else:
        print(f"  {event}: already configured")

with open(settings_file, "w") as f:
    json.dump(settings, f, indent=2)
    f.write("\n")

print("  Settings saved")
PYEOF

# --- Memory session-start hook (Claude Code and Codex) ---
# One file serves both harnesses. Codex is registered here; the Claude Code
# entry is printed for the user to add, so this script never rewrites that
# hook list on its own.
echo ""
echo "Installing the memory session-start hook..."
MEMORY_HOOK="$SCRIPT_DIR/hooks/memory-session-start.py"
for harness_dir in "$HOME/.claude" "$HOME/.codex"; do
    if [ -d "$harness_dir" ]; then
        mkdir -p "$harness_dir/hooks"
        install -m 755 "$MEMORY_HOOK" "$harness_dir/hooks/memory-session-start.py"
        echo "  Copied to $harness_dir/hooks/memory-session-start.py"
    fi
done

if [ -d "$HOME/.codex" ]; then
    python3 - "$HOME/.codex/hooks.json" "$HOME/.codex/hooks/memory-session-start.py" << 'PYEOF'
import json
import os
import sys

path, hook = sys.argv[1], sys.argv[2]
data = {}
if os.path.exists(path):
    with open(path) as fh:
        data = json.load(fh)
groups = data.setdefault("hooks", {}).setdefault("SessionStart", [])
command = f"python3 '{hook}'"
if any(h.get("command") == command for g in groups for h in g.get("hooks", [])):
    print("  Codex SessionStart: already registered")
else:
    groups.append({"matcher": "", "hooks": [{"type": "command", "command": command}]})
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2)
        fh.write("\n")
    print("  Codex SessionStart: registered in ~/.codex/hooks.json")
PYEOF
fi

if [ -d "$HOME/.claude" ]; then
    echo "  Claude Code: add this entry to hooks.SessionStart in ~/.claude/settings.json:"
    echo "    {\"matcher\": \"\", \"hooks\": [{\"type\": \"command\", \"command\": \"python3 $HOME/.claude/hooks/memory-session-start.py\"}]}"
fi

# --- Install global MCP server (user scope) ---
echo ""
echo "Installing global MCP server..."

# Install the headersHelper script (see mcp-headers-helper.sh)
MCP_HELPERS_DIR="$HOME/.claude/mcp-helpers"
"$SCRIPT_DIR/sessionflow-server.sh" install-headers-helper

# Add MCP server to ~/.claude.json (user scope)
CLAUDE_JSON="$HOME/.claude.json"
python3 << PYEOF
import json
import os

claude_json = "$CLAUDE_JSON"
helpers_dir = "$MCP_HELPERS_DIR"

if os.path.exists(claude_json):
    with open(claude_json) as f:
        data = json.load(f)
else:
    data = {}

servers = data.setdefault("mcpServers", {})
servers["sessionflow"] = {
    "type": "http",
    "url": "http://127.0.0.1:7102/mcp/",
    "headersHelper": f"{helpers_dir}/sessionflow-headers.sh",
}

with open(claude_json, "w") as f:
    json.dump(data, f, indent=2)
    f.write("\n")

print("  MCP server added to ~/.claude.json (user scope, all projects)")
PYEOF

# --- Optional LaunchAgent (autostart at login) ---
# Off by default. Enable non-interactively with SESSIONFLOW_INSTALL_AGENT=1, or
# answer "y" when prompted in an interactive shell. The LaunchAgent runs
# ~/.sessionflow/sessionflow-launcher.sh run at login, before any harness hooks fire — this
# avoids multiple terminals racing on server start when several harnesses
# (Claude Code, Codex, OpenCode, Antigravity CLI) launch at once.
echo ""
INSTALL_AGENT="${SESSIONFLOW_INSTALL_AGENT:-}"
if [ -z "$INSTALL_AGENT" ] && [ -t 0 ]; then
    read -r -p "Install optional macOS LaunchAgent to autostart SessionFlow at login? [y/N] " INSTALL_AGENT || true
fi
case "${INSTALL_AGENT:-n}" in
    1|y|Y|yes|YES)
        echo "Installing LaunchAgent..."
        "$SCRIPT_DIR/sessionflow-server.sh" install-agent || \
            echo "  LaunchAgent install reported a soft failure — run './sessionflow-server.sh agent-status' to verify."
        ;;
    *)
        echo "Skipping LaunchAgent install (optional)."
        echo "  Install later with: ./sessionflow-server.sh install-agent"
        echo "  Uninstall with:     ./sessionflow-server.sh uninstall-agent"
        echo "  Inspect with:       ./sessionflow-server.sh agent-status"
        ;;
esac

# --- Done ---
echo ""
echo "======================================================================"
echo "Installation Complete!"
echo "======================================================================"
echo ""
echo "Embedding model: $SESSIONFLOW_MODEL (see rag_engine.py for details)"
echo ""
echo "Hooks installed in: ~/.claude/settings.json"
echo "  - SessionStart: auto-start server + register file watcher + backfill"
echo "  - Stop: index final turns when session ends"
echo "  - PreCompact: index turns before context compaction"
echo ""
echo "MCP server installed globally in: ~/.claude.json"
echo "  - Available in all projects automatically"
echo "  - Dynamic project root via headersHelper"
echo ""
echo "Next step: Restart Claude Code to activate"
echo ""
echo "Optional autostart (LaunchAgent) commands:"
echo "  ./sessionflow-server.sh install-agent     # autostart at login"
echo "  ./sessionflow-server.sh agent-status      # inspect LaunchAgent"
echo "  ./sessionflow-server.sh uninstall-agent   # remove LaunchAgent"
echo ""
echo "See README.md for full documentation."
echo "======================================================================"
