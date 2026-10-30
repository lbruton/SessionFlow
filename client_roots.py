"""Take the current project from the MCP client's roots.

The server asks a client for its roots (``roots/list``) on the first scoped tool
call of an MCP session and keeps the first ``file://`` root as that session's
project. Asking needs a stateful session: the request travels on the tool call's
own response stream.
"""

from __future__ import annotations

import logging
import weakref
from typing import Iterable, Optional
from urllib.parse import unquote, urlsplit

import anyio
from mcp import types
from mcp.shared.message import ServerMessageMetadata

logger = logging.getLogger(__name__)

# A client that never answers costs one wait per session, not one per call.
ROOTS_TIMEOUT_S = 5.0

_ROOTS_CAPABILITY = types.ClientCapabilities(roots=types.RootsCapability())
_cache: "weakref.WeakKeyDictionary[object, Optional[str]]" = weakref.WeakKeyDictionary()


def root_path(uri: str) -> Optional[str]:
    """Return the local path of a ``file://`` root URI, or None for any other URI."""
    parts = urlsplit(uri or "")
    if parts.scheme != "file" or parts.netloc not in ("", "localhost"):
        return None
    path = unquote(parts.path)
    if not path.startswith("/"):
        return None
    return path.rstrip("/") or "/"


def first_root_path(roots: Iterable[types.Root]) -> Optional[str]:
    """Return the path of the first ``file://`` root, or None when there is none."""
    for root in roots:
        path = root_path(str(root.uri))
        if path:
            return path
    return None


async def project_root(session, request_id) -> Optional[str]:
    """Return the client's project folder for this MCP session, asking once.

    Args:
        session: The ``ServerSession`` of the calling client.
        request_id: The tool call's request ID; the roots request rides its stream.

    Returns:
        The first ``file://`` root as a path, or None when the client declares no
        roots, answers with none, fails or exceeds ``ROOTS_TIMEOUT_S``.
    """
    if session in _cache:
        return _cache[session]
    path = None
    if session.check_client_capability(_ROOTS_CAPABILITY):
        try:
            with anyio.fail_after(ROOTS_TIMEOUT_S):
                result = await session.send_request(
                    types.ServerRequest(types.ListRootsRequest()),
                    types.ListRootsResult,
                    metadata=ServerMessageMetadata(related_request_id=request_id),
                )
            path = first_root_path(result.roots)
        except Exception as exc:  # noqa: BLE001
            logger.warning("roots/list failed: %s: %s", type(exc).__name__, exc)
    _cache[session] = path
    return path


async def current_project_root(server) -> Optional[str]:
    """Return the project folder of the client behind the current request, if any."""
    try:
        ctx = server.request_context
    except (LookupError, AttributeError):
        return None
    return await project_root(ctx.session, ctx.request_id)


async def handle_list_changed(notification: types.RootsListChangedNotification) -> None:
    """Forget every cached answer; the notification does not say which session sent it."""
    forget_all()


def forget_all() -> None:
    """Drop every cached roots answer."""
    _cache.clear()


def register(server) -> None:
    """Install the ``notifications/roots/list_changed`` handler on ``server``."""
    server.notification_handlers[types.RootsListChangedNotification] = handle_list_changed
