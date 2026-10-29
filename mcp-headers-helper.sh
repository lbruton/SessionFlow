#!/bin/bash
# Claude Code headersHelper for the SessionFlow MCP server.
# Prints {"X-Project-Root": "<repo root>"} for the session's project, or {} when
# the project is unknown. Claude Code runs this with its working directory set
# to its config folder, so the project comes from CLAUDE_PROJECT_DIR or from the
# working directory of the parent claude process.

parent_cwd() {
    if [ -r "/proc/$PPID/cwd" ]; then
        readlink "/proc/$PPID/cwd"
        return
    fi
    local lsof_bin name
    lsof_bin=$(command -v lsof || echo /usr/sbin/lsof)
    name=$("$lsof_bin" -a -p "$PPID" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p' | head -n 1)
    # lsof escapes backslashes and unprintable bytes in names.
    printf '%b' "$name"
}

claude_home="${SESSIONFLOW_CLAUDE_HOME:-${CLAUDE_CONFIG_DIR:-$HOME/.claude}}"
dir="${CLAUDE_PROJECT_DIR:-}"
[ -n "$dir" ] || dir=$(parent_cwd)
[ -n "$dir" ] || dir=$(pwd)

# The config folder is never a project, even inside a git repo such as a
# versioned home folder, so it is checked before the git lookup. Claude Desktop
# starts each session in a spares/ folder and enters the project after this runs.
case "$dir" in
    "$claude_home" | "$claude_home/spares" | "$claude_home/spares/"*)
        echo "{}"
        exit 0
        ;;
esac

root=$(git -C "$dir" rev-parse --show-toplevel 2>/dev/null || echo "$dir")

# "/" would scope searches to the bucket of sessions with no project.
if [ -z "$root" ] || [ "$root" = "/" ] || [ "$root" = "$claude_home" ]; then
    echo "{}"
    exit 0
fi

python3 -c 'import json, sys; print(json.dumps({"X-Project-Root": sys.argv[1]}))' "$root" 2>/dev/null \
    || echo "{}"
