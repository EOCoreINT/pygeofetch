"""
Single-line progress display that is safe in Jupyter, terminals and logs.

Long network loops (for example the Sentinel-1 preflight burst check, which
fetches annotation and orbit files for every candidate date) used to show
nothing for minutes and then dump hundreds of log lines, so a notebook looked
frozen.  :class:`ProgressLine` shows ONE updating line instead:

* Jupyter: one output cell that is rewritten in place (no scrolling output);
* terminal: one line rewritten with a carriage return;
* anything else (log files, CI): a plain line every 25 %.

Updates are throttled, so calling :meth:`update` thousands of times is cheap,
and the class is thread-safe.
"""

from __future__ import annotations

import sys
import threading
import time
from typing import Any

_FILLED, _EMPTY = "█", "░"


def _fmt_time(seconds: float) -> str:
    seconds = int(max(0, seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _in_jupyter() -> bool:
    try:
        from IPython import get_ipython

        ip = get_ipython()
        return ip is not None and "ZMQInteractiveShell" in type(ip).__name__
    except Exception:
        return False


class ProgressLine:
    """One self-updating progress line.

    Example::

        with ProgressLine(total=60, desc="Preflight: annotations + orbits") as bar:
            for scene in scenes:
                fetch(scene)
                bar.update(note=scene_date)
    """

    def __init__(
        self,
        total: int,
        desc: str = "",
        enabled: bool = True,
        min_interval: float = 0.4,
        width: int = 24,
        stream: Any = None,
    ) -> None:
        self.total = max(int(total), 0)
        self.desc = desc
        self.enabled = enabled
        self.min_interval = min_interval
        self.width = width
        self.stream = stream or sys.stdout
        self.n = 0
        self._start = time.monotonic()
        self._last_draw = 0.0
        self._last_milestone = -1
        self._lock = threading.Lock()
        self._handle: Any = None
        self._mode = "off"
        self._closed = False
        if enabled:
            if stream is None and _in_jupyter():
                self._mode = "jupyter"
            elif getattr(self.stream, "isatty", lambda: False)():
                self._mode = "tty"
            else:
                self._mode = "plain"

    # ── context manager ──────────────────────────────────────────────────
    def __enter__(self) -> "ProgressLine":
        self._draw(force=True)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close(failed=exc_type is not None)

    # ── API ──────────────────────────────────────────────────────────────
    def update(self, n: int = 1, note: str = "") -> None:
        with self._lock:
            self.n = min(self.n + n, self.total) if self.total else self.n + n
            self._note = note
        self._draw()

    def close(self, failed: bool = False, final_note: str = "") -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._draw(force=True, final=True, failed=failed, final_note=final_note)
        if self._mode == "tty":
            self.stream.write("\n")
            self.stream.flush()

    # ── rendering ────────────────────────────────────────────────────────
    _note = ""

    def _text(
        self, final: bool = False, failed: bool = False, final_note: str = ""
    ) -> str:
        elapsed = time.monotonic() - self._start
        frac = (self.n / self.total) if self.total else 0.0
        filled = int(self.width * frac)
        bar = _FILLED * filled + _EMPTY * (self.width - filled)
        eta = ""
        if not final and self.n and self.total and self.n < self.total:
            eta = f" · ETA {_fmt_time(elapsed / self.n * (self.total - self.n))}"
        status = ""
        if final:
            status = "  ✗ stopped" if failed else "  ✓ done"
        note = final_note or (self._note if not final else "")
        note = f" · {note}" if note else ""
        return (
            f"{self.desc} {bar} {self.n}/{self.total} ({frac * 100:.0f}%)"
            f" · {_fmt_time(elapsed)}{eta}{note}{status}"
        )

    def _draw(
        self,
        force: bool = False,
        final: bool = False,
        failed: bool = False,
        final_note: str = "",
    ) -> None:
        if self._mode == "off":
            return
        now = time.monotonic()
        with self._lock:
            if not force and now - self._last_draw < self.min_interval:
                return
            self._last_draw = now
            text = self._text(final, failed, final_note)
            if self._mode == "plain":
                milestone = int((self.n / self.total) * 4) if self.total else 0
                if not final and milestone == self._last_milestone:
                    return
                self._last_milestone = milestone
        try:
            if self._mode == "jupyter":
                from IPython.display import display

                payload = {"text/plain": text}
                if self._handle is None:
                    self._handle = display(payload, raw=True, display_id=True)
                else:
                    self._handle.update(payload, raw=True)
            elif self._mode == "tty":
                self.stream.write("\r" + text + "\033[K")
                self.stream.flush()
            else:
                self.stream.write(text + "\n")
                self.stream.flush()
        except Exception:
            # Progress display must never break the real work.
            self._mode = "off"
