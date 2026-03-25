"""
Conversation memory manager with sliding window.

Stores chat history and resolves coreferences for follow-up queries.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings

Role = Literal["user", "assistant", "system", "tool"]


@dataclass
class Message:
    role: Role
    content: str
    metadata: dict = field(default_factory=dict)   # e.g. tool_name, tool_result


class ConversationMemory:
    """
    Sliding-window conversation memory.

    Keeps the last `window` full user+assistant turn pairs,
    plus a persistent system message at position 0.
    """

    def __init__(self, system_prompt: str = "", window: int | None = None):
        self._window = window or settings.memory_window
        self._system_prompt = system_prompt
        self._history: list[Message] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def add_user(self, text: str) -> None:
        self._history.append(Message(role="user", content=text))
        self._trim()

    def add_assistant(self, text: str) -> None:
        self._history.append(Message(role="assistant", content=text))

    def add_tool_result(self, tool_name: str, result: str) -> None:
        self._history.append(
            Message(role="tool", content=result, metadata={"tool_name": tool_name})
        )

    def get_messages(self) -> list[dict]:
        """Return message list in OpenAI chat format."""
        msgs: list[dict] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        for m in self._history:
            if m.role == "tool":
                # Represent as assistant note for compatibility
                msgs.append({
                    "role": "assistant",
                    "content": f"[Tool result — {m.metadata.get('tool_name', 'tool')}]: {m.content}",
                })
            else:
                msgs.append({"role": m.role, "content": m.content})
        return msgs

    def get_last_user_query(self) -> str:
        for m in reversed(self._history):
            if m.role == "user":
                return m.content
        return ""

    def get_context_string(self, last_n: int = 4) -> str:
        """Return last N messages as a plain text block for prompt injection."""
        recent = self._history[-last_n:]
        lines = []
        for m in recent:
            prefix = m.role.upper()
            lines.append(f"{prefix}: {m.content}")
        return "\n".join(lines)

    def resolve_coreference(self, query: str, candidates: list[str]) -> str | None:
        """
        Simple coreference resolution: if query contains pronouns like 'it', 'that',
        try to find the most recent entity mention in history.
        """
        pronouns = {"it", "that", "this", "they", "them", "those", "these"}
        tokens = set(query.lower().split())
        if not tokens.intersection(pronouns):
            return None

        # Look backwards through history for a candidate
        for m in reversed(self._history):
            for candidate in candidates:
                if candidate.lower() in m.content.lower():
                    return candidate
        return None

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _trim(self) -> None:
        """Keep only the last `window * 2` messages (window user+assistant pairs)."""
        max_msgs = self._window * 2
        if len(self._history) > max_msgs:
            self._history = self._history[-max_msgs:]
