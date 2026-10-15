"""
PyGeoFetch logging setup — delegates to pygeofetch.core.logging.

This module is kept for backward compatibility with code that imports from here.
New code should import from pygeofetch.core.logging directly.
"""

from pygeofetch.core.logging import (  # noqa: F401 — re-exported for compat
    _PGFFormatter,
    _redact,
    _render_progress_bar,
    configure_logging,
    get_logger,
)


def setup_logging(
    level: str = "INFO",
    log_file=None,
    use_rich: bool = True,  # accepted but ignored — we use _PGFFormatter
    use_colour: bool = True,
    show_module: bool = True,
    **kwargs,  # absorb any future/unknown keyword arguments
) -> None:
    """
    Configure PyGeoFetch logging.
    Alias for configure_logging(). Kept for backward compatibility.
    Accepts (and ignores) use_rich — the new formatter replaces Rich.
    """
    configure_logging(
        level=level,
        log_file=str(log_file) if log_file else None,
        use_colour=use_colour,
        show_module=show_module,
    )


__all__ = [
    "configure_logging",
    "setup_logging",
    "get_logger",
    "_render_progress_bar",
    "_PGFFormatter",
    "_redact",
]
