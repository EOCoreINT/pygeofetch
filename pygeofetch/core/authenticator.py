"""
Authentication manager for PyGeoFetch.

Handles secure credential storage, session caching, and authentication
flows for all providers. Supports system keyring, encrypted file storage,
and in-memory session caching.

Example::

    from pygeofetch.core.authenticator import AuthManager

    auth = AuthManager()
    auth.add("usgs", username="user", password="pass")
    session = auth.authenticate("usgs")  # Returns cached or new session

    auth.list()   # All saved providers
    auth.remove("usgs")  # Remove credentials
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pygeofetch.core.logging import get_logger
from pygeofetch.models.user_auth import AuthSession, AuthType, Credentials

if TYPE_CHECKING:
    import builtins

    from cryptography.fernet import Fernet

logger = get_logger(__name__)


class CredentialStore:
    """
    Manages secure credential storage.

    Attempts to use the system keyring; falls back to encrypted file storage.
    """

    SERVICE_NAME = "pygeofetch"

    def __init__(self, storage_backend: str = "keyring") -> None:
        self.storage_backend = storage_backend
        self._config_dir = Path.home() / ".pygeofetch"
        self._cred_file = self._config_dir / "credentials.enc"
        self._legacy_cred_file = self._config_dir / "credentials.json"
        self._key_file = self._config_dir / "credentials.key"
        self._in_memory: dict[str, dict[str, Any]] = {}
        self._fernet: Fernet | None = None  # lazily initialized — see _get_fernet()
        self._migrating = False  # re-entrancy guard for _migrate_legacy_file_if_needed

    def save(self, provider: str, credentials: dict[str, Any]) -> None:
        """Persist credentials for a provider."""
        if self.storage_backend == "keyring":
            self._save_keyring(provider, credentials)
        else:
            self._save_file(provider, credentials)
        self._in_memory[provider] = credentials

    def load(self, provider: str) -> dict[str, Any] | None:
        """Load credentials for a provider — keyring first, then file fallback."""
        if provider in self._in_memory:
            return self._in_memory[provider]
        if self.storage_backend == "keyring":
            result = self._load_keyring(provider)
            if result:
                return result
            # Fall back to file store so credentials saved via --store file
            # are always visible regardless of which backend is active
        result = self._load_file(provider)
        if result:
            self._in_memory[provider] = result  # cache it
        return result

    def delete(self, provider: str) -> bool:
        """Remove stored credentials for a provider."""
        self._in_memory.pop(provider, None)
        if self.storage_backend == "keyring":
            return self._delete_keyring(provider)
        return self._delete_file(provider)

    def list_providers(self) -> list[str]:
        """Return list of providers with stored credentials."""
        providers = set(self._in_memory.keys())
        if self._cred_file.exists():
            try:
                data = json.loads(self._cred_file.read_text())
                providers.update(data.keys())
            except Exception:
                pass
        return sorted(providers)

    def _save_keyring(self, provider: str, credentials: dict[str, Any]) -> None:
        try:
            import keyring

            keyring.set_password(self.SERVICE_NAME, provider, json.dumps(credentials))
            logger.debug(f"Credentials for {provider!r} saved to system keyring")
        except ImportError:
            logger.warning(
                "keyring package not available; falling back to file storage"
            )
            self._save_file(provider, credentials)
        except Exception as exc:
            logger.warning(f"Keyring save failed: {exc}; falling back to file storage")
            self._save_file(provider, credentials)

    def _load_keyring(self, provider: str) -> dict[str, Any] | None:
        try:
            import keyring

            data = keyring.get_password(self.SERVICE_NAME, provider)
            if data:
                return json.loads(data)
        except ImportError:
            return self._load_file(provider)
        except Exception:
            return self._load_file(provider)
        return None

    def _delete_keyring(self, provider: str) -> bool:
        try:
            import keyring

            keyring.delete_password(self.SERVICE_NAME, provider)
            return True
        except Exception:
            return self._delete_file(provider)

    _SENSITIVE_FIELDS = ("password", "api_key", "token", "secret_key", "client_secret")

    def _get_fernet(self) -> Fernet:
        """
        Return a cached Fernet instance, generating and persisting a new
        local encryption key on first use.

        REAL BUG FIXED: this file backend previously used
        ``base64.b64encode`` on sensitive fields with a comment stating
        outright "Basic obfuscation (not encryption)" -- base64 is
        trivially reversible by anyone who can read the file, with no
        key required, and the docs previously (and inaccurately) claimed
        this path used Fernet AES-128-CBC. It now genuinely does.
        """
        if self._fernet is not None:
            return self._fernet
        from cryptography.fernet import Fernet

        self._config_dir.mkdir(parents=True, exist_ok=True)
        if self._key_file.exists():
            key = self._key_file.read_bytes()
        else:
            key = Fernet.generate_key()
            self._key_file.write_bytes(key)
            self._key_file.chmod(0o600)
        self._fernet = Fernet(key)
        return self._fernet

    def _migrate_legacy_file_if_needed(self) -> None:
        """
        One-time transparent migration from the old base64-"obfuscated"
        ``credentials.json`` to the new Fernet-encrypted
        ``credentials.enc``, so existing users don't lose stored
        credentials when upgrading. Runs at most once: after migration,
        the legacy file is left in place but unused (never read again
        once ``credentials.enc`` exists) rather than deleted, in case a
        user needs to roll back.
        """
        if self._cred_file.exists() or not self._legacy_cred_file.exists() or self._migrating:
            return
        self._migrating = True
        try:
            try:
                legacy_data = json.loads(self._legacy_cred_file.read_text())
            except Exception as exc:
                logger.warning(f"Could not read legacy credentials file for migration: {exc}")
                return

            migrated = 0
            for provider, creds in legacy_data.items():
                if not isinstance(creds, dict):
                    continue
                plain = dict(creds)
                if plain.pop("__obfuscated", False):
                    for k in self._SENSITIVE_FIELDS:
                        if k in plain:
                            try:
                                plain[k] = base64.b64decode(str(plain[k]).encode()).decode()
                            except Exception:
                                pass
                self._save_file(provider, plain)
                migrated += 1
            if migrated:
                logger.info(
                    f"Migrated {migrated} provider credential(s) from the old "
                    f"{self._legacy_cred_file.name} (base64) to the new "
                    f"{self._cred_file.name} (Fernet-encrypted). The old file "
                    "was left in place, unused, and can be deleted manually."
                )
        finally:
            self._migrating = False

    def _save_file(self, provider: str, credentials: dict[str, Any]) -> None:
        self._migrate_legacy_file_if_needed()
        self._config_dir.mkdir(parents=True, exist_ok=True)
        existing: dict[str, Any] = {}
        if self._cred_file.exists():
            try:
                existing = json.loads(self._cred_file.read_text())
            except Exception:
                pass

        fernet = self._get_fernet()
        encrypted = {}
        for k, v in credentials.items():
            if k in self._SENSITIVE_FIELDS:
                encrypted[k] = fernet.encrypt(str(v).encode()).decode()
            else:
                encrypted[k] = v
        existing[provider] = {"__encrypted": True, **encrypted}
        self._cred_file.write_text(json.dumps(existing, indent=2))
        self._cred_file.chmod(0o600)
        logger.debug(f"Credentials for {provider!r} saved to {self._cred_file} (encrypted)")

    def _load_file(self, provider: str) -> dict[str, Any] | None:
        self._migrate_legacy_file_if_needed()
        if not self._cred_file.exists():
            return None
        try:
            data = json.loads(self._cred_file.read_text())
            creds = data.get(provider)
            if not creds:
                return None
            if creds.get("__encrypted"):
                creds = {k: v for k, v in creds.items() if k != "__encrypted"}
                fernet = self._get_fernet()
                decoded = {}
                for k, v in creds.items():
                    if k in self._SENSITIVE_FIELDS:
                        try:
                            decoded[k] = fernet.decrypt(str(v).encode()).decode()
                        except Exception:
                            decoded[k] = v
                    else:
                        decoded[k] = v
                return decoded
            # Legacy plain/base64 entries encountered directly in
            # credentials.enc (shouldn't normally happen, but handled
            # rather than silently failing).
            if creds.get("__obfuscated"):
                creds = {k: v for k, v in creds.items() if k != "__obfuscated"}
                decoded = {}
                for k, v in creds.items():
                    if k in self._SENSITIVE_FIELDS:
                        try:
                            decoded[k] = base64.b64decode(str(v).encode()).decode()
                        except Exception:
                            decoded[k] = v
                    else:
                        decoded[k] = v
                return decoded
            return creds
        except Exception as exc:
            logger.warning(f"Failed to load credentials from file: {exc}")
            return None

    def _delete_file(self, provider: str) -> bool:
        if not self._cred_file.exists():
            return False
        try:
            data = json.loads(self._cred_file.read_text())
            if provider in data:
                del data[provider]
                self._cred_file.write_text(json.dumps(data, indent=2))
                return True
        except Exception:
            pass
        return False


class AuthManager:
    """
    Manages authentication across all satellite data providers.

    Handles credential storage, session caching, and authentication flows.
    Sessions are cached in memory and reused until expiry.

    Attributes:
        store: Underlying credential store (keyring or file).
        _sessions: In-memory session cache keyed by provider ID.

    Example::

        auth = AuthManager()

        # Store credentials
        auth.add("usgs", username="user", password="secret")
        auth.add("planet", api_key="pk.abc123")

        # Authenticate (returns cached session if valid)
        session = auth.authenticate("usgs")

        # Check what's stored
        for p in auth.list():
            print(p)

        # Remove credentials
        auth.remove("copernicus")
    """

    def __init__(self, storage_backend: str = "file") -> None:
        self.store = CredentialStore(storage_backend=storage_backend)
        self._sessions: dict[str, AuthSession] = {}

    def add(
        self,
        provider: str,
        *,
        username: str | None = None,
        password: str | None = None,
        api_key: str | None = None,
        token: str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        auth_type: str | None = None,
        **extra: Any,
    ) -> None:
        """
        Add or update credentials for a provider.

        Args:
            provider: Provider ID (e.g., 'usgs', 'copernicus').
            username: Username for username/password auth.
            password: Password.
            api_key: API key.
            token: Bearer/access token.
            client_id: OAuth2 client ID.
            client_secret: OAuth2 client secret.
            access_key: AWS access key.
            secret_key: AWS secret key.
            auth_type: Override auth type string.
            **extra: Additional provider-specific credentials, stored under
                     Credentials.extra and passed through to the provider's
                     authenticate() method. For Copernicus accounts with
                     Two-Factor Authentication enabled, pass the current
                     6-digit code as totp="123456" — CDSE requires this on
                     every login when 2FA is active on the account.

        Raises:
            ValueError: If no credentials are provided.

        Example::

            auth.add("usgs", username="user", password="pass")
            auth.add("planet", api_key="pk.abc123")
            auth.add("copernicus", username="user@example.com", password="pass")
            # Copernicus account with 2FA enabled:
            auth.add("copernicus", username="user@example.com", password="pass",
                     totp="123456")
        """
        cred_data: dict[str, Any] = {"provider": provider}
        if username:
            cred_data["username"] = username
        if password:
            cred_data["password"] = password
        if api_key:
            cred_data["api_key"] = api_key
        if token:
            cred_data["token"] = token
        if client_id:
            cred_data["client_id"] = client_id
        if client_secret:
            cred_data["client_secret"] = client_secret
        if access_key:
            cred_data["access_key"] = access_key
        if secret_key:
            cred_data["secret_key"] = secret_key
        if auth_type:
            cred_data["auth_type"] = auth_type
        # IMPORTANT: nest remaining kwargs under the `extra` field rather than
        # spreading them as top-level keys. Credentials does not declare
        # arbitrary fields (pydantic default extra="ignore"), so anything
        # passed here as a bare top-level key — e.g. totp="123456" for
        # Copernicus 2FA-enabled accounts — would previously be silently
        # dropped before ever reaching the provider's authenticate() call.
        if extra:
            cred_data["extra"] = extra

        if len(cred_data) <= 1:  # Only 'provider' key
            msg = f"No credentials provided for {provider!r}"
            raise ValueError(msg)

        # Infer auth type if not set
        if "auth_type" not in cred_data:
            if api_key:
                cred_data["auth_type"] = AuthType.API_KEY.value
            elif token:
                cred_data["auth_type"] = AuthType.TOKEN.value
            elif access_key:
                cred_data["auth_type"] = AuthType.AWS_CREDENTIALS.value
            else:
                cred_data["auth_type"] = AuthType.USERNAME_PASSWORD.value

        self.store.save(provider, cred_data)
        # Invalidate cached session
        self._sessions.pop(provider, None)
        logger.info(f"Credentials saved for provider {provider!r}")

    def add_credentials(
        self,
        provider: str,
        creds: dict[str, Any],
    ) -> None:
        """
        Store credentials for the given provider.

        Accepts a dict (not keyword arguments — the engine passes a dict).
        Idempotent: calling twice with same provider updates, does not duplicate.
        Not raise for unknown provider names.

        Args:
            provider: Provider identifier e.g. "usgs", "planetary_computer"
            creds:    Dict with any of: username, password, api_key,
                      client_id, client_secret, token, access_key, secret_key.
                      Only non-None values are stored.
        """
        # Filter out None values and the redundant "provider" key
        clean_creds = {
            k: v for k, v in creds.items() if k != "provider" and v is not None
        }
        if not clean_creds:
            logger.warning(f"No credentials provided for {provider!r} — skipping")
            return
        # Delegate to add() which handles storage, session invalidation, type inference
        self.add(provider, **clean_creds)
        logger.debug(f"Stored credentials for provider: {provider}")

    def get_credentials(self, provider: str) -> Credentials | None:
        """
        Load stored credentials for a provider.

        Args:
            provider: Provider ID.

        Returns:
            Credentials instance, or None if not found.
        """
        data = self.store.load(provider)
        if not data:
            return None
        try:
            from pydantic import SecretStr

            # Always inject the provider field — it may not be persisted on disk
            data = dict(data)
            data.setdefault("provider", provider)
            if "password" in data and isinstance(data["password"], str):
                data["password"] = SecretStr(data["password"])
            if "api_key" in data and isinstance(data["api_key"], str):
                data["api_key"] = SecretStr(data["api_key"])
            if "token" in data and isinstance(data["token"], str):
                data["token"] = SecretStr(data["token"])
            if "secret_key" in data and isinstance(data["secret_key"], str):
                data["secret_key"] = SecretStr(data["secret_key"])
            if "client_secret" in data and isinstance(data["client_secret"], str):
                data["client_secret"] = SecretStr(data["client_secret"])
            return Credentials(**data)
        except Exception as exc:
            logger.warning(f"Failed to parse credentials for {provider!r}: {exc}")
            return None

    def authenticate(self, provider: str, force_refresh: bool = False) -> AuthSession:
        """
        Authenticate with a provider, using cached session if valid.

        Args:
            provider: Provider ID to authenticate with.
            force_refresh: If True, always re-authenticate even if session is valid.

        Returns:
            Valid AuthSession.

        Raises:
            ValueError: If no credentials are stored for this provider.
            AuthenticationError: If authentication fails.
        """
        from pygeofetch.providers import get_provider

        # Return cached session if still valid
        if not force_refresh and provider in self._sessions:
            session = self._sessions[provider]
            if session.is_valid:
                logger.debug(f"Using cached session for {provider!r}")
                return session

        credentials = self.get_credentials(provider)
        if not credentials:
            msg = (
                f"No credentials found for {provider!r}. "
                f"Add them with: pygeofetch auth add {provider}"
            )
            raise ValueError(msg)

        prov = get_provider(provider)
        session = prov.authenticate(credentials)
        self._sessions[provider] = session
        return session

    def set_session(self, provider: str, session: AuthSession) -> None:
        """Manually set a session (e.g. from OAuth2 callback)."""
        self._sessions[provider] = session

    def get_session(self, provider: str) -> AuthSession | None:
        """Return cached session for a provider, or None."""
        session = self._sessions.get(provider)
        if session and session.is_valid:
            return session
        return None

    def list(self) -> builtins.list[dict[str, Any]]:
        """
        List all providers with stored credentials.

        Returns:
            List of dicts with provider info and session status.
        """
        providers = self.store.list_providers()
        result = []
        for provider in providers:
            session = self._sessions.get(provider)
            result.append(
                {
                    "provider": provider,
                    "has_session": session is not None and session.is_valid,
                    "session_expires": session.expires_at.isoformat()
                    if session and session.expires_at
                    else None,
                }
            )
        return result

    def remove(self, provider: str) -> bool:
        """
        Remove stored credentials and session for a provider.

        Args:
            provider: Provider ID to remove.

        Returns:
            True if credentials were found and removed.
        """
        self._sessions.pop(provider, None)
        removed = self.store.delete(provider)
        if removed:
            logger.info(f"Credentials removed for {provider!r}")
        return removed

    def export_credentials(
        self, providers: builtins.list[str] | None = None
    ) -> dict[str, Any]:
        """
        Export credentials for backup (passwords are obfuscated).

        Args:
            providers: List of provider IDs to export. None = export all.

        Returns:
            Dict mapping provider IDs to credential data.
        """
        all_providers = providers or self.store.list_providers()
        result = {}
        for provider in all_providers:
            creds = self.get_credentials(provider)
            if creds:
                result[provider] = {
                    "provider": creds.provider,
                    "auth_type": creds.auth_type.value,
                    "username": creds.username,
                }
        return result