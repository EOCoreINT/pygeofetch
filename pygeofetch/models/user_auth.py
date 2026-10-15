"""
Authentication models for PyGeoFetch.

Defines credential types, auth sessions, and authentication configuration
structures used across all providers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, SecretStr


class AuthType(str, Enum):
    """Supported authentication mechanisms."""

    NONE = "none"
    API_KEY = "api_key"
    USERNAME_PASSWORD = "username_password"
    OAUTH2 = "oauth2"
    TOKEN = "token"
    AWS_CREDENTIALS = "aws_credentials"
    SERVICE_ACCOUNT = "service_account"
    EARTHDATA_LOGIN = "earthdata_login"


class Credentials(BaseModel):
    """
    Provider credentials container.

    Attributes:
        provider: Provider name these credentials belong to.
        auth_type: Authentication mechanism.
        username: Username for username/password auth.
        password: Password (stored as SecretStr).
        api_key: API key for API key auth.
        token: Bearer/access token.
        client_id: OAuth2 client ID.
        client_secret: OAuth2 client secret.
        access_key: AWS access key ID.
        secret_key: AWS secret access key.
        service_account_json: JSON path or content for service accounts.
        extra: Additional provider-specific credential fields.
    """

    provider: str
    auth_type: AuthType = AuthType.USERNAME_PASSWORD
    username: str | None = None
    password: SecretStr | None = None
    api_key: SecretStr | None = None
    token: SecretStr | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    access_key: str | None = None
    secret_key: SecretStr | None = None
    service_account_json: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {SecretStr: lambda v: "***"}

    def get_password(self) -> str | None:
        """Return password as plain string."""
        return self.password.get_secret_value() if self.password else None

    def get_api_key(self) -> str | None:
        """Return API key as plain string."""
        return self.api_key.get_secret_value() if self.api_key else None

    def get_token(self) -> str | None:
        """Return token as plain string."""
        return self.token.get_secret_value() if self.token else None

    def get_secret_key(self) -> str | None:
        """Return AWS secret key as plain string."""
        return self.secret_key.get_secret_value() if self.secret_key else None


class AuthSession(BaseModel):
    """
    Represents an active authentication session.

    Attributes:
        provider: Provider this session belongs to.
        access_token: Current access/session token.
        refresh_token: Token for refreshing the session.
        expires_at: Session expiration datetime.
        session_data: Provider-specific session state.
        created_at: When the session was created.
    """

    provider: str
    access_token: (
        Any | None
    )  # str or SecretStr — use .get_secret_value() if SecretStr = None
    refresh_token: str | None = None
    expires_at: datetime | None = None
    session_data: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_expired(self) -> bool:
        """Return True if the session has expired."""
        if self.expires_at is None:
            return False
        from datetime import timezone

        if self.expires_at.tzinfo is not None:
            return datetime.now(timezone.utc) >= self.expires_at
        return datetime.utcnow() >= self.expires_at

    @property
    def is_valid(self) -> bool:
        """Return True if session exists and is not expired."""
        return self.access_token is not None and not self.is_expired

    @property
    def minutes_until_expiry(self) -> float | None:
        """Return minutes until session expires, or None if no expiry set."""
        if self.expires_at is None:
            return None
        delta = self.expires_at - datetime.utcnow()
        return delta.total_seconds() / 60
