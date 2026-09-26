"""
The opt-in tool audit log.

The one sanctioned exception to "no prompt content is written to disk" (CS-21, owner's decision,
2026-09-23): tool ARGUMENTS can contain the user's words - a search query - so they are
persisted only when the operator has actively asked for it by naming a file in the server
config. Unset or '' means nothing is written; there is no default location.

Results are deliberately NOT logged: that is where private data lives (grades). The file is
created owner-only (0600) and appended one JSON object per line.
"""
import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Optional


class ToolAuditLog:
    """
    Appends one JSON line per tool call to the configured file, or does nothing at all.

    Attributes:
        path (str | None): The file, or None when auditing is off.
    """

    def __init__(self, path: Optional[str]):
        self.path = path or None             # '' is off, exactly like unset
        self._lock = threading.Lock()        # several sessions may finish a call at once

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def record(self, session_id: str, tool: str, arguments: Dict[str, Any], outcome: str, duration_s: float):
        """
        Logs one call.

        Args:
            session_id: Which session made it.
            tool: The tool's name.
            arguments: What the model passed.
            outcome: 'ok' | 'failed' | 'refused' | 'denied'.
            duration_s: How long it took (0 for a call that never ran).
        """
        if not self.enabled:
            return
        line = json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "session_id": session_id,
                           "tool": tool, "arguments": arguments, "outcome": outcome,
                           "duration_s": round(duration_s, 3)}, ensure_ascii=False, default=str)
        with self._lock:
            # os.open with an explicit mode, so the file is born 0600 rather than created then chmod-ed.
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
