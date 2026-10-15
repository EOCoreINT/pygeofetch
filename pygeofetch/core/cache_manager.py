"""
Disk-backed cache manager for PyGeoFetch.

Provides a persistent key-value store using JSON files in a configurable
cache directory (~/.pygeofetch/cache by default).  Each entry is stored
as a separate JSON file with an expiry timestamp so stale entries can be
pruned on startup or on demand.

Example::

    from pygeofetch.core.cache_manager import CacheManager

    cache = CacheManager(ttl_seconds=86400)
    cache.set("my_key", {"data": [1, 2, 3]})
    value = cache.get("my_key")
    cache.purge_expired()
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


def _default_cache_dir() -> Path:
    return Path.home() / ".pygeofetch" / "cache"


class CacheManager:
    """
    Persistent disk cache with TTL-based expiry.

    Each cached item is stored as a JSON file named by the MD5 hash of
    the cache key under *cache_dir*.  The JSON envelope includes the
    original key, expiry timestamp, and payload.

    Attributes:
        cache_dir: Path to the cache directory.
        ttl: Default time-to-live in seconds.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_seconds: int = 86_400,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache_dir()
        self.ttl = ttl_seconds
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get(self, key: str) -> Any | None:
        """
        Retrieve a cached value.

        Args:
            key: Cache key string.

        Returns:
            Cached value, or ``None`` if missing or expired.
        """
        path = self._path(key)
        if not path.exists():
            return None
        try:
            envelope = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            path.unlink(missing_ok=True)
            return None

        if time.time() > envelope.get("expires_at", 0):
            path.unlink(missing_ok=True)
            return None

        return envelope.get("value")

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Store a value in the cache.

        Args:
            key: Cache key string.
            value: JSON-serialisable value to store.
            ttl: Override the default TTL in seconds.
        """
        expires_at = time.time() + (ttl if ttl is not None else self.ttl)
        envelope = {
            "key": key,
            "expires_at": expires_at,
            "stored_at": time.time(),
            "value": value,
        }
        path = self._path(key)
        tmp = path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(envelope, default=str))
            tmp.replace(path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            msg = f"Cache write failed for key {key!r}: {exc}"
            raise RuntimeError(msg) from exc

    def delete(self, key: str) -> bool:
        """
        Remove an entry from the cache.

        Args:
            key: Cache key to remove.

        Returns:
            True if the entry existed and was removed, False otherwise.
        """
        path = self._path(key)
        if path.exists():
            path.unlink()
            return True
        return False

    def exists(self, key: str) -> bool:
        """Return True if *key* is present and not expired."""
        return self.get(key) is not None

    def clear(self) -> int:
        """
        Remove all entries from the cache directory.

        Returns:
            Number of files removed.
        """
        count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                f.unlink()
                count += 1
            except OSError:
                pass
        return count

    def purge_expired(self) -> int:
        """
        Remove expired cache entries.

        Returns:
            Number of stale entries removed.
        """
        now = time.time()
        count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                envelope = json.loads(f.read_text())
                if now > envelope.get("expires_at", 0):
                    f.unlink()
                    count += 1
            except (json.JSONDecodeError, OSError):
                f.unlink(missing_ok=True)
                count += 1
        return count

    def stats(self) -> dict[str, Any]:
        """
        Return cache statistics.

        Returns:
            Dict with ``total``, ``expired``, ``valid``, ``size_bytes`` keys.
        """
        now = time.time()
        total = 0
        expired = 0
        size_bytes = 0
        for f in self.cache_dir.glob("*.json"):
            total += 1
            size_bytes += f.stat().st_size
            try:
                envelope = json.loads(f.read_text())
                if now > envelope.get("expires_at", 0):
                    expired += 1
            except Exception:
                expired += 1
        return {
            "total": total,
            "expired": expired,
            "valid": total - expired,
            "size_bytes": size_bytes,
            "cache_dir": str(self.cache_dir),
        }

    def list_keys(self) -> list[tuple[str, float]]:
        """
        List all non-expired cache keys and their expiry timestamps.

        Returns:
            List of (key, expires_at) tuples.
        """
        now = time.time()
        result = []
        for f in self.cache_dir.glob("*.json"):
            try:
                envelope = json.loads(f.read_text())
                if now <= envelope.get("expires_at", 0):
                    result.append((envelope.get("key", f.stem), envelope["expires_at"]))
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        """Return the file path for a given cache key."""
        digest = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"CacheManager(dir={self.cache_dir}, "
            f"valid={stats['valid']}, expired={stats['expired']}, "
            f"ttl={self.ttl}s)"
        )

    def prune_to_size(self, max_bytes: int) -> int:
        """
        Remove oldest entries until cache is below max_bytes.

        Args:
            max_bytes: Target maximum cache size in bytes.

        Returns:
            Number of entries removed.
        """
        import json as _json

        files = sorted(
            [
                (f, _json.loads(f.read_text()).get("stored_at", 0))
                for f in self.cache_dir.glob("*.json")
                if f.exists()
            ],
            key=lambda x: x[1],
        )
        removed = 0
        for f, _ in files:
            stats = self.stats()
            if stats["size_bytes"] <= max_bytes:
                break
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
        return removed

    def clear_filtered(
        self,
        provider_filter: str | None = None,
        max_age_seconds: float | None = None,
    ) -> int:
        """
        Remove cache entries with optional filters.

        Args:
            provider_filter: Remove only entries whose key contains this provider ID.
            max_age_seconds: Remove entries older than this many seconds.

        Returns:
            Number of entries removed.
        """
        import json as _json

        now = time.time()
        count = 0
        for f in self.cache_dir.glob("*.json"):
            try:
                envelope = _json.loads(f.read_text())
                key = envelope.get("key", "")
                stored_at = envelope.get("stored_at", 0)

                if provider_filter and provider_filter not in key:
                    continue
                if max_age_seconds and (now - stored_at) < max_age_seconds:
                    continue
                f.unlink()
                count += 1
            except (OSError, Exception):
                f.unlink(missing_ok=True)
                count += 1
        return count
