"""Pause and resume whatever the GUI is running.

A graph cannot be frozen mid-thought, but every agent step starts with a model
call, so a callback that waits at the start of each call pauses a run at its
next step: the call in flight finishes, the next one waits. One gate serves
the whole server, since an analysis and a job never run at the same time.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from langchain_core.callbacks import BaseCallbackHandler


class PauseGate:
    def __init__(self):
        self._open = threading.Event()
        self._open.set()
        self.paused_at: float | None = None

    @property
    def paused(self) -> bool:
        return not self._open.is_set()

    def pause(self) -> None:
        if not self.paused:
            self.paused_at = time.time()
            self._open.clear()

    def resume(self) -> None:
        self.paused_at = None
        self._open.set()

    def wait(self, should_stop: Callable[[], bool] | None = None) -> None:
        """Block while paused; a stop request releases the wait."""
        while not self._open.wait(0.5):
            if should_stop and should_stop():
                return


GATE = PauseGate()


class PauseCallback(BaseCallbackHandler):
    """Holds each model call at its start while the gate is closed."""

    def __init__(self, gate: PauseGate = GATE,
                 on_hold: Callable[[], None] | None = None,
                 should_stop: Callable[[], bool] | None = None):
        self.gate = gate
        self.on_hold = on_hold
        self.should_stop = should_stop

    def _hold(self) -> None:
        if self.gate.paused:
            if self.on_hold:
                self.on_hold()
            self.gate.wait(self.should_stop)

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        self._hold()

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._hold()
