"""
PyGeoFetch logging & display layer.

Usage::

    from pygeofetch.core.logging import configure_logging, get_logger
    from pygeofetch.core.logging import print_search_results, DownloadProgress

    configure_logging(level="INFO")
    logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import re
import sys
import threading
import time
from typing import Any

# ── ANSI codes ────────────────────────────────────────────────────────────────

_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "blue": "\033[34m",
    "white": "\033[97m",
    "grey": "\033[90m",
    "bg_blue": "\033[44m",
    "bg_green": "\033[42m",
}

# Detect execution environment
_TTY = sys.stdout.isatty()


def _in_jupyter() -> bool:
    """Return True when running inside a Jupyter / IPython kernel."""
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None:
            return False
        return "ZMQInteractiveShell" in type(ip).__name__  # Jupyter kernel
    except ImportError:
        return False


_JUPYTER = _in_jupyter()


def _c(code: str, text: str) -> str:
    """Apply an ANSI code only when writing to a real terminal."""
    if not _TTY:
        return text
    return f"{_C.get(code, '')}{text}{_C['reset']}"


def _cb(text: str) -> str:
    return _c("bold", text)


def _cg(text: str) -> str:
    return _c("green", text)


def _cy(text: str) -> str:
    return _c("yellow", text)


def _cd(text: str) -> str:
    return _c("dim", text)


def _cr(text: str) -> str:
    return _c("red", text)


def _cc(text: str) -> str:
    return _c("cyan", text)


def _cbl(text: str) -> str:
    return _c("blue", text)


# ── Credential redaction ──────────────────────────────────────────────────────

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "api_key",
        "apikey",
        "token",
        "access_key",
        "secret_key",
        "client_secret",
        "credentials",
        "authorization",
        "auth",
    }
)


def _redact(msg: str) -> str:
    for key in _SENSITIVE_KEYS:
        msg = re.sub(
            r"(?i)(" + re.escape(key) + r"[\s]*[=:][\s]*)[^\s,}\]\n\"']+",
            r"\1***REDACTED***",
            msg,
        )
    return msg


# ── Text alignment helper (ANSI-aware) ────────────────────────────────────────


_ANSI_RE = re.compile(r"\033\[[\d;]*m")


def _visible_len(text: str) -> int:
    """Length of *text* ignoring ANSI escape codes."""
    return len(_ANSI_RE.sub("", text))


def _pad(text: str, width: int, align: str = "<") -> str:
    """
    Pad *text* to *width* using visible character count (strips ANSI codes first).
    This ensures columns align correctly even when text contains colour escapes,
    which Python's built-in ``:<N`` format spec does NOT handle correctly.
    """
    visible = _visible_len(text)
    padding = max(0, width - visible)
    if align == "<":
        return text + " " * padding
    if align == ">":
        return " " * padding + text
    # centre
    left = padding // 2
    right = padding - left
    return " " * left + text + " " * right


def _truncate(text: str, width: int) -> str:
    """Truncate *text* to *width* visible characters, adding an ellipsis if cut."""
    if _visible_len(text) <= width:
        return text
    if width <= 1:
        return text[:width]
    return text[: width - 1] + "…"


# ── Log formatter ─────────────────────────────────────────────────────────────


class _PGFFormatter(logging.Formatter):
    _LEVEL_STYLE = {
        "DEBUG": ("dim", "DEBG"),
        "INFO": ("cyan", "INFO"),
        "WARNING": ("yellow", "WARN"),
        "ERROR": ("red", "ERRR"),
        "CRITICAL": ("magenta", "CRIT"),
    }

    def __init__(self, use_colour: bool = True, show_module: bool = True):
        super().__init__()
        self._use_colour = use_colour and _TTY
        self._show_module = show_module

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, datefmt="%H:%M:%S")
        style, badge = self._LEVEL_STYLE.get(
            record.levelname, ("", record.levelname[:4])
        )
        if self._use_colour:
            badge = _c(style, badge)
            ts = _cd(ts)

        module = record.name.split(".")[-1] if self._show_module else ""
        mod_str = (
            _cd(f"[{module:>12}]")
            if (module and self._use_colour)
            else (f"[{module:>12}]" if module else "")
        )

        msg = _redact(record.getMessage())
        if record.exc_info:
            msg = f"{msg}\n{self.formatException(record.exc_info)}"

        return f"{ts} {badge} {mod_str} {msg}"


# ── configure_logging / get_logger ───────────────────────────────────────────


def configure_logging(
    level: str = "INFO",
    log_file: str | None = None,
    use_colour: bool = True,
    show_module: bool = True,
) -> None:
    """Configure PyGeoFetch logging. Call once at startup."""
    root = logging.getLogger("pygeofetch")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_PGFFormatter(use_colour=use_colour, show_module=show_module))
    root.addHandler(ch)

    if log_file:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(fh)

    for lib in (
        "urllib3",
        "requests",
        "httpx",
        "botocore",
        "azure",
        "pystac",
        "planetary_computer",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    if not name.startswith("pygeofetch"):
        name = f"pygeofetch.{name}"
    return logging.getLogger(name)


# ══════════════════════════════════════════════════════════════════════════════
#  SEARCH RESULTS TABLE
# ══════════════════════════════════════════════════════════════════════════════

_BOX_WIDTH = 76  # total width of the search-parameters box, including borders


def print_search_header(
    providers: list[str],
    bbox: Any,
    start_date: Any,
    end_date: Any,
    cloud_max: float,
    product: str = "any",
) -> None:
    """Print a clean, bordered search-parameters header box."""
    w = _BOX_WIDTH
    inner = w - 4  # content width inside "│ " ... " │"

    bbox_str = (
        f"[{bbox.min_lon:.3f}, {bbox.min_lat:.3f}, "
        f"{bbox.max_lon:.3f}, {bbox.max_lat:.3f}]"
        if bbox
        else "—"
    )
    prov_str = ", ".join(providers) if providers else "—"

    title = " SEARCH PARAMETERS "
    dashes = max(0, w - 2 - len(title))
    print("┌" + title + "─" * dashes + "┐")

    def row(label: str, value: str) -> None:
        text = f"{_cc(f'{label:<11}')}: {_truncate(value, inner - 13)}"
        print("│ " + _pad(text, inner) + " │")

    row("Providers", prov_str)
    row("BBox", bbox_str)
    row("Date range", f"{start_date}  →  {end_date}")
    row("Cloud max", f"{cloud_max}%")
    row("Product", product)

    print("└" + "─" * (w - 2) + "┘")


def print_provider_progress(
    provider: str, status: str, count: int = 0, duration: float = 0.0, error: str = ""
) -> None:
    """Print a single provider search result line."""
    provider_str = _pad(_cb(provider), 28)

    if error:
        icon = _cr("✗")
        line = f"  {icon}  {provider_str}  {_cr(_truncate(error, 45))}"
    elif count == 0:
        icon = _cy("○")
        line = f"  {icon}  {provider_str}  {_cy('no results')}"
    else:
        icon = _cg("✓")
        dur_str = _cd(f"{duration:.1f}s")
        count_str = _cb(_cg(f"{count:>4} scenes"))
        line = f"  {icon}  {provider_str}  {count_str}   {dur_str}"

    print(line)


def print_search_results(results: list[Any], elapsed: float = 0.0) -> None:
    """Print a perfectly aligned, bordered table of search results."""
    if not results:
        print(_cy("  ○  No scenes matched the search criteria."))
        return

    n = len(results)

    # ── column definitions: (header, data_width) ─────────────────────────────
    COLS = [
        ("SCENE ID", 42),
        ("DATE", 10),
        ("SATELLITE", 14),
        ("CLOUD", 6),
        ("PRODUCT", 7),
        ("POLARISATION", 12),
        ("PASS", 11),
        ("ORBIT", 5),
        ("SCORE", 5),
        ("PROVIDER", 20),
    ]

    def hline(left: str, mid: str, right: str) -> str:
        return left + mid.join("─" * (cw + 2) for _, cw in COLS) + right

    def row_line(cells: list[str]) -> str:
        parts = [" " + _pad(text, cw) + " " for text, (_, cw) in zip(cells, COLS)]
        return "│" + "│".join(parts) + "│"

    # ── header ───────────────────────────────────────────────────────────────
    print(hline("┌", "┬", "┐"))
    header_cells = [_cb(h) for h, _ in COLS]
    print(
        "│"
        + "│".join(
            " " + _pad(text, cw, align="^") + " "
            for text, (_, cw) in zip(header_cells, COLS)
        )
        + "│"
    )
    print(hline("├", "┼", "┤"))

    # ── rows ─────────────────────────────────────────────────────────────────
    for r in results[:30]:
        scene_id = _truncate(r.id or "—", COLS[0][1])
        date = str(r.datetime)[:10] if r.datetime else "—"
        sat = _truncate(r.satellite or "—", COLS[2][1])
        ptype = _truncate(r.product_type or "—", COLS[4][1])
        pol = _truncate(r.polarisation or "—", COLS[5][1])
        pdir = _truncate(r.pass_direction or "—", COLS[6][1])
        orbit = _truncate(
            str(r.relative_orbit) if r.relative_orbit else "—", COLS[7][1]
        )
        score = f"{r.score:.2f}" if r.score else "—"
        provider = _truncate(r.provider or "—", COLS[9][1])

        # Cloud cover with colour
        if r.cloud_cover is not None:
            raw = f"{r.cloud_cover:.0f}%"
            if r.cloud_cover <= 10:
                cloud = _cg(raw)
            elif r.cloud_cover <= 30:
                cloud = _cy(raw)
            else:
                cloud = _cr(raw)
        else:
            cloud = "—"

        cells = [
            _cd(scene_id),
            date,
            _cb(sat),
            cloud,
            ptype,
            pol,
            pdir,
            orbit,
            score,
            _cd(provider),
        ]
        print(row_line(cells))

    print(hline("└", "┴", "┘"))

    # ── footer ────────────────────────────────────────────────────────────────
    overflow = f"  (+{n - 30} more not shown)" if n > 30 else ""
    elapsed_str = f"  ·  {elapsed:.1f}s" if elapsed else ""
    print(
        f"  {_cb(_cg(str(n)))} scene{'s' if n != 1 else ''} found{elapsed_str}{overflow}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  DOWNLOAD PROGRESS
# ══════════════════════════════════════════════════════════════════════════════

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# ANSI "clear to end of line" — used instead of manual space-padding so a
# shorter line never leaves stray characters from a previous, longer render.
_CLEAR_EOL = "\033[K"


class DownloadProgress:
    """
    Live download progress — real-time, works correctly in both terminals
    and Jupyter notebooks, and correctly supports multiple CONCURRENT
    downloads (parallel > 1) without state corruption between them.

    Built on tqdm (tqdm.auto), the standard, battle-tested progress bar
    library, which auto-detects the execution environment (real terminal
    vs Jupyter/IPython kernel vs plain non-TTY stream) and renders
    correctly in each — this is the single implementation that "works
    everywhere" rather than hand-rolled ANSI/HTML dual-mode rendering.

    Each concurrently-downloading item gets its own named progress bar
    (terminal: stacked via cursor positioning; Jupyter: separate widgets),
    identified by an explicit item_id passed to every call — this is what
    makes it safe to call from multiple ThreadPoolExecutor worker threads
    at once, unlike a design that tracks a single shared "current item".

    Usage::

        with DownloadProgress(total=3, destination="./downloads") as dp:
            dp.start_item("scene_1", expected_bytes=2_000_000_000)
            dp.update("scene_1", bytes_done=500_000_000, speed_bps=18_000_000)
            dp.complete_item("scene_1", success=True, bytes_total=2_000_000_000, duration=17.8)
    """

    def __init__(self, total: int, destination: str = "") -> None:
        self.total = total
        self.destination = destination
        self._completed = 0
        self._failed = 0
        self._start = time.time()
        self._lock = threading.Lock()

        self._bars: dict[str, Any] = {}  # item_id -> tqdm instance
        self._positions: dict[str, int] = {}  # item_id -> stable position slot
        self._next_position = 1  # 0 is reserved for the overall bar
        self._overall_bar: Any = None
        self._tqdm_cls: Any = None

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "DownloadProgress":
        try:
            from tqdm.auto import tqdm as tqdm_cls
        except ImportError:
            # tqdm is a core dependency; this only triggers on a stale
            # install. Degrade gracefully to plain logging rather than
            # crashing the whole download.
            logging.getLogger("pygeofetch.core.logging").warning(
                "tqdm not installed — falling back to plain log lines for "
                "download progress. Run: pip install tqdm"
            )
            self._tqdm_cls = None
            return self

        self._tqdm_cls = tqdm_cls
        dest = f"  →  {self.destination}" if self.destination else ""
        self._overall_bar = tqdm_cls(
            total=self.total,
            position=0,
            desc=f"⬇ {self.total} scene{'s' if self.total != 1 else ''}{dest}",
            unit="scene",
            dynamic_ncols=True,
            leave=True,
            bar_format="{desc}  {bar}  {n_fmt}/{total_fmt}  [{elapsed}]",
        )
        return self

    def __exit__(self, *_exc) -> None:
        with self._lock:
            for bar in list(self._bars.values()):
                bar.close()
            self._bars.clear()
            if self._overall_bar is not None:
                self._overall_bar.close()

        if self._tqdm_cls is None:
            elapsed = time.time() - self._start
            if self._failed == 0:
                print(f"✓ All {self._completed} scenes downloaded  ({elapsed:.1f}s)")
            else:
                print(
                    f"✓ {self._completed} succeeded, ✗ {self._failed} failed  "
                    f"({elapsed:.1f}s)"
                )

    # ── public API ───────────────────────────────────────────────────────────

    def start_item(self, item_id: str, expected_bytes: int = 0) -> None:
        """Begin tracking a new download item. Safe to call concurrently
        for different item_ids from multiple threads."""
        if self._tqdm_cls is None:
            logging.getLogger("pygeofetch.core.logging").info(
                "Downloading %s", _truncate(item_id, 60)
            )
            return

        with self._lock:
            existing = self._bars.get(item_id)
            if existing is not None:
                existing.close()

            position = self._positions.get(item_id)
            if position is None:
                position = self._next_position
                self._positions[item_id] = position
                self._next_position += 1

            bar = self._tqdm_cls(
                total=expected_bytes or None,
                position=position,
                desc=_truncate(item_id, 45),
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
                leave=False,
            )
            self._bars[item_id] = bar

    def update(self, item_id: str, bytes_done: int, speed_bps: float = 0.0) -> None:
        """
        Report incremental progress for item_id. Call this repeatedly
        (e.g. once per HTTP chunk) with the CUMULATIVE bytes downloaded so
        far — this is what makes the bar move in real time rather than
        jumping straight from 0% to 100% when the download finishes.
        """
        if self._tqdm_cls is None:
            return

        with self._lock:
            bar = self._bars.get(item_id)
            if bar is None:
                return
            delta = bytes_done - bar.n
            if delta > 0:
                bar.update(delta)
            if speed_bps > 0:
                bar.set_postfix_str(self._fmt_speed(speed_bps), refresh=False)

    def complete_item(
        self, item_id: str, success: bool, bytes_total: int = 0, duration: float = 0.0
    ) -> None:
        """Mark item_id as finished (success or failure)."""
        with self._lock:
            if success:
                self._completed += 1
            else:
                self._failed += 1

            if self._tqdm_cls is not None:
                bar = self._bars.pop(item_id, None)
                if bar is not None:
                    if bytes_total and bar.total is None:
                        bar.total = bytes_total
                    if bytes_total and bar.n < bytes_total:
                        bar.update(bytes_total - bar.n)
                    icon = "✓" if success else "✗"
                    bar.set_description_str(f"{icon} {_truncate(item_id, 43)}")
                    bar.refresh()
                    bar.close()
                if self._overall_bar is not None:
                    self._overall_bar.update(1)
            else:
                icon = "✓" if success else "✗"
                size = self._fmt_size(bytes_total)
                logging.getLogger("pygeofetch.core.logging").info(
                    "  %s %-45s %10s  %5.1fs",
                    icon,
                    _truncate(item_id, 45),
                    size,
                    duration,
                )

    # ── formatting helpers ────────────────────────────────────────────────────

    def _fmt_size(self, b: int) -> str:
        if b <= 0:
            return "—"
        if b < 1 << 20:
            return f"{b / 1024:.0f} KB"
        if b < 1 << 30:
            return f"{b / 1048576:.1f} MB"
        return f"{b / 1073741824:.2f} GB"

    def _fmt_speed(self, bps: float) -> str:
        if bps <= 0:
            return ""
        if bps < 1 << 20:
            return f"{bps / 1024:.0f} KB/s"
        return f"{bps / 1048576:.1f} MB/s"


# ══════════════════════════════════════════════════════════════════════════════
#  PER-CHUNK PROGRESS BRIDGE (used by provider download() implementations)
# ══════════════════════════════════════════════════════════════════════════════

_progress_local = threading.local()


def _set_active_progress(
    progress: "DownloadProgress | None", item_id: str | None
) -> None:
    """
    Internal: called by AdaptiveDownloader around each provider.download()
    call, so that provider code (running on its own worker thread) can
    report real-time chunk progress back to the right bar via
    report_download_progress() below, without every provider needing an
    explicit progress-callback parameter threaded through its signature.

    Uses thread-local storage — safe for ThreadPoolExecutor, since each
    worker thread gets its own isolated (progress, item_id) pair and
    parallel downloads never cross-report into each other's bars.
    """
    _progress_local.progress = progress
    _progress_local.item_id = item_id


def report_download_progress(
    bytes_done: int, bytes_total: int = 0, speed_bps: float = 0.0
) -> None:
    """
    Call this from inside a provider's download() streaming loop — once
    per chunk written — to drive real-time progress bar updates.

    Safe to call unconditionally, including when there is no active
    progress bar (e.g. provider.download() called directly outside
    AdaptiveDownloader, or in a context where progress display is off):
    becomes a no-op in that case rather than raising.

    Example (inside a provider's streaming download loop)::

        from pygeofetch.core.logging import report_download_progress

        bytes_written = 0
        with open(output_file, "wb") as f:
            for chunk in response.iter_bytes(chunk_size=chunk_size):
                f.write(chunk)
                bytes_written += len(chunk)
                report_download_progress(bytes_written, total_bytes)
    """
    progress = getattr(_progress_local, "progress", None)
    item_id = getattr(_progress_local, "item_id", None)
    if progress is not None and item_id is not None:
        progress.update(item_id, bytes_done, speed_bps)


# ── Backward compat ───────────────────────────────────────────────────────────


def _render_progress_bar(
    completed, total, filename, bytes_done, bytes_total, speed_bps, bar_width=20
) -> str:
    """Legacy helper kept for backward compatibility."""
    filled = int(bar_width * min(completed, total) / max(total, 1))
    bar = "█" * filled + "░" * (bar_width - filled)
    done_gb = bytes_done / 1e9
    tot_gb = bytes_total / 1e9 if bytes_total else 0.0
    spd_mbs = speed_bps / 1e6
    name = filename[:40] + "…" if len(filename) > 40 else filename
    size_str = (
        f"{done_gb:.1f} GB / {tot_gb:.1f} GB"
        if bytes_total > 0
        else f"{done_gb:.1f} GB"
    )
    return (
        f"\r  [{bar}]  {completed}/{total}  {name:<42}  {size_str}  {spd_mbs:.1f} MB/s"
    )
