"""
File handling utilities for PyGeoFetch.

Provides checksum computation, file verification, extraction,
and other file system operations used during downloads.

Example::

    from pygeofetch.utils.file_utils import compute_checksum, safe_extract

    checksum = compute_checksum(Path("/data/scene.tif"), algorithm="sha256")
    safe_extract(Path("/data/scene.zip"), Path("/data/extracted/"))
"""

from __future__ import annotations

import hashlib
import json
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Generator


def compute_checksum(path: Path, algorithm: str = "md5") -> str:
    """
    Compute the checksum of a file.

    Args:
        path: Path to the file.
        algorithm: Hash algorithm ('md5', 'sha256', 'sha512').

    Returns:
        Lowercase hex digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If algorithm is unsupported.
    """
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    algo = algorithm.lower().replace("-", "")
    try:
        hasher = hashlib.new(algo)
    except ValueError:
        msg = f"Unsupported checksum algorithm: {algorithm}"
        raise ValueError(msg)

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_checksum(path: Path, expected: str, algorithm: str = "md5") -> bool:
    """
    Verify a file's checksum against an expected value.

    Args:
        path: Path to the file.
        expected: Expected checksum hex string.
        algorithm: Hash algorithm to use.

    Returns:
        True if checksums match, False otherwise.
    """
    actual = compute_checksum(path, algorithm)
    return actual.lower() == expected.lower()


def safe_extract(
    archive_path: Path,
    destination: Path,
    remove_archive: bool = False,
) -> list[Path]:
    """
    Safely extract a zip or tar archive, preventing path traversal.

    Args:
        archive_path: Path to the archive file.
        destination: Directory to extract into.
        remove_archive: If True, delete the archive after extraction.

    Returns:
        List of extracted file paths.

    Raises:
        ValueError: If archive format is unsupported or path traversal detected.
    """
    destination.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    def is_safe_path(base: Path, target: Path) -> bool:
        try:
            target.resolve().relative_to(base.resolve())
            return True
        except ValueError:
            return False

    suffix = archive_path.suffix.lower()
    if archive_path.name.lower().endswith(".tar.gz") or suffix in (
        ".gz",
        ".bz2",
        ".xz",
    ):
        with tarfile.open(archive_path) as tar:
            for member in tar.getmembers():
                target = destination / member.name
                if not is_safe_path(destination, target):
                    msg = f"Path traversal detected in archive: {member.name}"
                    raise ValueError(msg)
            tar.extractall(destination)
            extracted = [destination / m.name for m in tar.getmembers() if m.isfile()]
    elif suffix == ".zip":
        with zipfile.ZipFile(archive_path) as zf:
            for name in zf.namelist():
                target = destination / name
                if not is_safe_path(destination, target):
                    msg = f"Path traversal detected in archive: {name}"
                    raise ValueError(msg)
            zf.extractall(destination)
            extracted = [destination / n for n in zf.namelist() if not n.endswith("/")]
    else:
        msg = f"Unsupported archive format: {archive_path.suffix}"
        raise ValueError(msg)

    if remove_archive:
        archive_path.unlink(missing_ok=True)

    return extracted


def ensure_directory(path: Path) -> Path:
    """
    Create directory and all parents if they don't exist.

    Args:
        path: Directory path to create.

    Returns:
        The created/existing directory path.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(path: Path) -> int:
    """Return file size in bytes, or 0 if file does not exist."""
    try:
        return path.stat().st_size
    except (FileNotFoundError, OSError):
        return 0


def human_readable_size(size_bytes: int) -> str:
    """
    Format byte count as human-readable string.

    Args:
        size_bytes: File size in bytes.

    Returns:
        String like '1.23 GB', '456 MB', '789 KB', '123 B'.
    """
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} {unit}"
        size_bytes /= 1024  # type: ignore
    return f"{size_bytes:.1f} PB"


def write_json(data: Any, path: Path, indent: int = 2) -> None:
    """Write data as formatted JSON to a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)


def read_json(path: Path) -> Any:
    """Read and parse JSON from a file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_write(path: Path, content: bytes) -> None:
    """
    Write content to a file atomically using a temporary file.

    Prevents partial writes if the process is interrupted.

    Args:
        path: Target file path.
        content: Binary content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            tmp.write(content)
            tmp.flush()
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
    tmp_path.replace(path)


def chunk_file_reader(
    path: Path, chunk_size: int = 1024 * 1024
) -> Generator[bytes, None, None]:
    """
    Yield file content in chunks.

    Args:
        path: File to read.
        chunk_size: Chunk size in bytes (default: 1 MB).

    Yields:
        Binary chunks.
    """
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk


def find_files(
    directory: Path, pattern: str = "*", recursive: bool = True
) -> list[Path]:
    """
    Find files matching a glob pattern in a directory.

    Args:
        directory: Root directory to search.
        pattern: Glob pattern (e.g., '*.tif', '**/*.json').
        recursive: If True, search subdirectories.

    Returns:
        Sorted list of matching Path objects.
    """
    if recursive:
        return sorted(directory.rglob(pattern))
    return sorted(directory.glob(pattern))


def clean_directory(directory: Path, older_than_days: int | None = None) -> int:
    """
    Remove files from a directory, optionally only those older than N days.

    Args:
        directory: Directory to clean.
        older_than_days: Only remove files older than this many days.

    Returns:
        Number of files removed.
    """
    import time

    if not directory.exists():
        return 0

    removed = 0
    cutoff = time.time() - (older_than_days * 86400) if older_than_days else None

    for item in directory.rglob("*"):
        if item.is_file():
            if cutoff is None or item.stat().st_mtime < cutoff:
                item.unlink()
                removed += 1

    # Remove empty directories
    for item in sorted(directory.rglob("*"), reverse=True):
        if item.is_dir():
            try:
                item.rmdir()
            except OSError:
                pass

    return removed
