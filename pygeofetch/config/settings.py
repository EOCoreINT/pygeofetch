"""
Configuration management for PyGeoFetch.

Implements a hierarchical configuration system:
  1. Built-in defaults (defaults.yaml)
  2. Global config (~/.pygeofetch/config.yaml)
  3. Environment variables (PYGEOFETCH_*)
  4. Project config (.pygeofetch.yaml)
  5. CLI arguments (applied at runtime)

Example::

    from pygeofetch.config.settings import get_settings

    settings = get_settings()
    print(settings.download.parallel)
    print(settings.providers.usgs.base_url)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DownloadSettings(BaseModel):
    """Download-related configuration."""

    parallel: int = Field(default=2, ge=1, le=32)
    chunk_size_mb: float = Field(default=8.0, gt=0)
    retry_attempts: int = Field(default=3, ge=0)
    retry_strategy: str = "exponential_jitter"
    retry_delay_seconds: float = 1.0
    verify_checksum: bool = True
    checksum_algorithm: str = "md5"
    bandwidth_limit_mbps: float = 0
    timeout_seconds: int = 300
    overwrite: bool = False
    keep_original: bool = False


class SearchSettings(BaseModel):
    """Search-related configuration."""

    max_results: int = 100
    page_size: int = 100
    cache_ttl_seconds: int = 3600
    deduplicate: bool = True
    sort_by: str = "datetime"
    sort_ascending: bool = False


class CacheSettings(BaseModel):
    """Cache configuration."""

    enabled: bool = True
    directory: Path = Path("~/.pygeofetch/cache").expanduser()
    max_size_gb: float = 5.0
    ttl_seconds: int = 86400
    provider_ttl: dict[str, int] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Per-provider configuration."""

    base_url: str = ""
    auth_url: str | None = None
    timeout: int = 60
    rate_limit_per_minute: int | None = None
    region: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class ProxySettings(BaseModel):
    """HTTP proxy configuration."""

    http: str | None = None
    https: str | None = None
    no_proxy: list[str] = Field(default_factory=list)


class SecuritySettings(BaseModel):
    """Security and credential configuration."""

    verify_ssl: bool = True
    credential_storage: str = "keyring"
    credential_file: Path = Path("~/.pygeofetch/credentials.enc").expanduser()


class NotificationSettings(BaseModel):
    """Notification configuration."""

    webhook: str | None = None
    email: str | None = None
    slack: str | None = None


class GeneralSettings(BaseModel):
    """General application settings."""

    output_dir: Path = Path("./satellite_data")
    temp_dir: Path | None = None
    log_level: str = "INFO"
    log_file: Path | None = None


class Settings(BaseSettings):
    """
    Main PyGeoFetch settings container.

    Reads from environment variables prefixed with PYGEOFETCH_
    and config files in standard locations.

    Environment variable examples:
        PYGEOFETCH_DOWNLOAD__PARALLEL=4
        PYGEOFETCH_GENERAL__LOG_LEVEL=DEBUG
        PYGEOFETCH_GENERAL__OUTPUT_DIR=/data/satellite
    """

    model_config = SettingsConfigDict(
        env_prefix="PYGEOFETCH_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="allow",
    )

    general: GeneralSettings = Field(default_factory=GeneralSettings)
    download: DownloadSettings = Field(default_factory=DownloadSettings)
    search: SearchSettings = Field(default_factory=SearchSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    proxy: ProxySettings = Field(default_factory=ProxySettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> Settings:
        """Load settings from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)


def _load_defaults() -> dict[str, Any]:
    """Load the bundled defaults.yaml."""
    defaults_path = Path(__file__).parent / "defaults.yaml"
    with open(defaults_path) as f:
        return yaml.safe_load(f) or {}


def _load_user_config() -> dict[str, Any]:
    """Load user config from ~/.pygeofetch/config.yaml if it exists."""
    config_path = Path.home() / ".pygeofetch" / "config.yaml"
    if config_path.exists():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _load_project_config() -> dict[str, Any]:
    """Load project config from .pygeofetch.yaml in CWD if it exists."""
    project_path = Path.cwd() / ".pygeofetch.yaml"
    if project_path.exists():
        with open(project_path) as f:
            return yaml.safe_load(f) or {}
    return {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Build and cache the application settings.

    Merges defaults → user config → project config → environment variables.
    Call reset_settings() to invalidate the cache.

    Returns:
        Settings instance with all configuration layers applied.
    """
    data = _load_defaults()
    data = _deep_merge(data, _load_user_config())
    data = _deep_merge(data, _load_project_config())
    return Settings(**data)


def reset_settings() -> None:
    """Invalidate the cached settings (useful for testing)."""
    get_settings.cache_clear()


def get_config_dir() -> Path:
    """Return the PyGeoFetch configuration directory."""
    config_dir = Path.home() / ".pygeofetch"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def save_user_config(updates: dict[str, Any]) -> Path:
    """
    Merge updates into the user config file and save.

    Args:
        updates: Settings to update (nested dict).

    Returns:
        Path to the saved config file.
    """
    config_dir = get_config_dir()
    config_path = config_dir / "config.yaml"
    existing: dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            existing = yaml.safe_load(f) or {}
    merged = _deep_merge(existing, updates)
    with open(config_path, "w") as f:
        yaml.dump(merged, f, default_flow_style=False, allow_unicode=True)
    reset_settings()
    return config_path
