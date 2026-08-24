"""Conversation memory: keeps a bounded, turn-based message history.

The Claude Messages API is stateless, so the caller resends history on every
turn. This module owns that history and trims it so long sessions don't grow
`max_tokens` costs unbounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass
class ConversationMemory:
    max_turns: int = 20
    messages: list[dict] = field(default_factory=list)

    def add(self, role: Role, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim()

    def _trim(self) -> None:
        # A "turn" is one user+assistant pair; keep the most recent N turns.
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def as_list(self) -> list[dict]:
        return list(self.messages)

    def reset(self) -> None:
        self.messages.clear()

    def truncate_to_turns(self, turns: int) -> None:
        """Keep only the first `turns` user+assistant pairs, dropping the rest.

        Used to rewind history when the user edits or retries an earlier turn.
        """
        self.messages = self.messages[: max(turns, 0) * 2]

    def __len__(self) -> int:
        return len(self.messages)
