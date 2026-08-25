"""
Abstract base provider for PyGeoFetch.

All satellite data providers must inherit from AbstractBaseProvider and
implement its abstract methods. This ensures a consistent interface
regardless of the underlying data source.

Example - implementing a new provider::

    from pygeofetch.providers.base import AbstractBaseProvider

    class MyProvider(AbstractBaseProvider):
        PROVIDER_ID = "myprovider"
        DISPLAY_NAME = "My Satellite Provider"
        REQUIRES_AUTH = True

        def authenticate(self, credentials):
            ...

        def search(self, query):
            ...

        def download(self, data, destination, options):
            ...

        def get_capabilities(self):
            ...

        def validate_credentials(self, credentials):
            ...

        def get_quota_info(self):
            ...
"""

from __future__ import annotations

import abc
from pathlib import Path
from typing import Any, Dict, List, Optional  # noqa: F401

from pygeofetch.models.download_task import DownloadOptions, DownloadResult
from pygeofetch.models.satellite_data import (
    ProviderCapabilities,
    QuotaInfo,
    SatelliteData,
)
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.user_auth import AuthSession, Credentials
from pygeofetch.utils.logging_setup import get_logger
from pygeofetch.utils.retry_handler import CircuitBreaker


class ProviderError(Exception):
    """Base exception for all provider errors."""


class AuthenticationError(ProviderError):
    """Raised when authentication fails."""


class SearchError(ProviderError):
    """Raised when a search operation fails."""


class DownloadError(ProviderError):
    """Raised when a download operation fails."""


class QuotaExceededError(ProviderError):
    """Raised when provider quota is exceeded."""


class ProviderUnavailableError(ProviderError):
    """Raised when a provider is temporarily unavailable."""


class AbstractBaseProvider(abc.ABC):
    """
    Abstract base class for all satellite data providers.

    Subclasses must define class attributes PROVIDER_ID, DISPLAY_NAME,
    and REQUIRES_AUTH, and implement all abstract methods.

    Class Attributes:
        PROVIDER_ID: Unique string identifier (e.g., 'usgs').
        DISPLAY_NAME: Human-readable name (e.g., 'USGS Earth Explorer').
        REQUIRES_AUTH: Whether authentication is required.
        DESCRIPTION: Short description of the data source.
        DATA_TYPES: List of data type strings this provider offers.
        BASE_URL: Default API base URL.
    """

    PROVIDER_ID: str = ""
    DISPLAY_NAME: str = ""
    REQUIRES_AUTH: bool = True
    DESCRIPTION: str = ""
    DATA_TYPES: list[str] = []
    BASE_URL: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """
        Initialize the provider.

        Args:
            config: Provider-specific configuration dict overriding defaults.
        """
        self.config = config or {}
        self._session: AuthSession | None = None
        self._logger = get_logger(f"provider.{self.PROVIDER_ID}")
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0,
            name=self.PROVIDER_ID,
        )

    # ------------------------------------------------------------------
    # Abstract methods (must be implemented by all providers)
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def authenticate(self, credentials: Credentials) -> AuthSession:
        """
        Authenticate with the provider using the given credentials.

        Args:
            credentials: Provider credentials (API key, username/password, etc.).

        Returns:
            Active AuthSession.

        Raises:
            AuthenticationError: If authentication fails.
        """

    @abc.abstractmethod
    def search(self, query: SearchQuery) -> list[SatelliteData]:
        """
        Search for satellite data matching the query.

        Args:
            query: Search parameters including spatial, temporal, and
                   quality filters.

        Returns:
            List of SatelliteData matching the query.

        Raises:
            SearchError: If the search operation fails.
            AuthenticationError: If the provider requires auth and none is set.
        """

    @abc.abstractmethod
    def download(
        self,
        data: SatelliteData,
        destination: Path,  # noqa: F821
        options: DownloadOptions,
    ) -> DownloadResult:
        """
        Download a satellite data product.

        Args:
            data: SatelliteData record to download.
            destination: Directory or file path to download to.
            options: Download configuration (retry, checksum, etc.).

        Returns:
            DownloadResult with status and output paths.

        Raises:
            DownloadError: If the download fails after retries.
            AuthenticationError: If session has expired.
        """

    @abc.abstractmethod
    def get_capabilities(self) -> ProviderCapabilities:
        """
        Return what this provider supports.

        Returns:
            ProviderCapabilities instance.
        """

    @abc.abstractmethod
    def validate_credentials(self, credentials: Credentials) -> bool:
        """
        Check whether the given credentials are valid without fully authenticating.

        Args:
            credentials: Credentials to validate.

        Returns:
            True if credentials appear valid.
        """

    @abc.abstractmethod
    def get_quota_info(self) -> QuotaInfo:
        """
        Return current quota/rate-limit usage information.

        Returns:
            QuotaInfo instance.

        Raises:
            AuthenticationError: If the provider requires auth.
        """

    # ------------------------------------------------------------------
    # Concrete helpers available to all providers
    # ------------------------------------------------------------------

    @property
    def is_authenticated(self) -> bool:
        """Return True if there is an active, non-expired session."""
        return self._session is not None and self._session.is_valid

    @property
    def session(self) -> AuthSession | None:
        """Return the current auth session, or None."""
        return self._session

    def set_session(self, session: AuthSession) -> None:
        """Manually set an auth session (e.g. from cached credentials)."""
        self._session = session
        self._logger.debug(f"Session set for {self.PROVIDER_ID}")

    def require_auth(self) -> None:
        """
        Raise AuthenticationError if the provider is not authenticated.

        Raises:
            AuthenticationError: When not authenticated.
        """
        if self.REQUIRES_AUTH and not self.is_authenticated:
            msg = (
                f"Provider '{self.PROVIDER_ID}' requires authentication. "
                f"Run: pygeofetch auth add {self.PROVIDER_ID}"
            )
            raise AuthenticationError(msg)

    def _make_http_client(self, **kwargs: Any) -> Any:
        """
        Create a configured httpx Client for this provider.

        Args:
            **kwargs: Additional arguments passed to httpx.Client.

        Returns:
            Configured httpx.Client.
        """
        import httpx  # lazy import — httpx is a required dep but not imported at module level

        timeout = self.config.get("timeout", 60)
        headers = {"User-Agent": "PyGeoFetch/0.1.0"}

        if self._session and self._session.access_token:
            headers["Authorization"] = f"Bearer {self._session.access_token}"

        return httpx.Client(
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            **kwargs,
        )

    def _handle_http_error(self, response: Any) -> None:
        """
        Raise appropriate exceptions based on HTTP status codes.

        Args:
            response: httpx Response object.

        Raises:
            AuthenticationError: On 401/403.
            QuotaExceededError: On 429.
            ProviderError: On other 4xx/5xx errors.
        """
        if response.status_code == 401:
            msg = (
                f"Authentication failed for {self.PROVIDER_ID}. "
                "Please re-authenticate with: pygeofetch auth login"
            )
            raise AuthenticationError(msg)
        if response.status_code == 403:
            msg = f"Access denied for {self.PROVIDER_ID}. Check your subscription/permissions."
            raise AuthenticationError(msg)
        if response.status_code == 429:
            msg = (
                f"Rate limit exceeded for {self.PROVIDER_ID}. "
                "Please wait before making more requests."
            )
            raise QuotaExceededError(msg)
        if response.status_code >= 500:
            msg = f"{self.PROVIDER_ID} server error: {response.status_code}"
            raise ProviderUnavailableError(msg)
        if response.status_code >= 400:
            msg = (
                f"{self.PROVIDER_ID} request failed: HTTP {response.status_code} - "
                f"{response.text[:200]}"
            )
            raise ProviderError(msg)

    def __repr__(self) -> str:
        auth_status = "authenticated" if self.is_authenticated else "unauthenticated"
        return f"<{self.__class__.__name__} provider_id={self.PROVIDER_ID!r} {auth_status}>"


def _bbox4(v) -> tuple[float, float, float, float] | None:
    """Normalise any bbox to a 4-float tuple or None."""
    if v is None:
        return None
    try:
        t = tuple(float(x) for x in list(v)[:4])
        return t if len(t) == 4 else None
    except Exception:
        return None
