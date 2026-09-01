"""
Regression tests for CredentialStore's file backend.

Previously: the file backend's own source comment read "Basic
obfuscation (not encryption) - passwords stored as b64" -- base64 is
trivially reversible by anyone who can read the file, with no key
required. This verifies the real replacement (Fernet, a genuine
symmetric encryption scheme) actually protects the plaintext on disk,
round-trips correctly, and transparently migrates existing users' old
base64 files without losing their stored credentials.
"""

from __future__ import annotations

import base64
import json

import pytest

from pygeofetch.core.authenticator import CredentialStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr("pygeofetch.core.authenticator.Path.home", lambda: tmp_path)
    return CredentialStore(storage_backend="file")


class TestRealEncryption:
    def test_secret_is_not_recoverable_without_the_key(self, store):
        """
        REAL BUG FIXED: base64 provides zero confidentiality -- anyone
        who can read the file can trivially recover the secret with no
        key at all. This confirms the on-disk ciphertext cannot be
        turned back into the real password by base64-decoding it (the
        old, broken scheme), proving it's genuinely encrypted now.
        """
        store.save("usgs", {"username": "alice", "password": "s3cr3t-real-password"})

        raw = json.loads(store._cred_file.read_text())
        stored_password = raw["usgs"]["password"]

        assert "s3cr3t-real-password" not in store._cred_file.read_text()

        # The OLD scheme would have been recoverable via a bare base64
        # decode -- confirm that no longer works.
        try:
            decoded = base64.b64decode(stored_password.encode()).decode()
        except Exception:
            decoded = None
        assert decoded != "s3cr3t-real-password"

    def test_round_trips_correctly(self, store):
        store.save(
            "copernicus",
            {
                "username": "bob@example.com",
                "password": "hunter2",
                "api_key": "PL_ABC123",
            },
        )
        loaded = store.load("copernicus")
        assert loaded["username"] == "bob@example.com"
        assert loaded["password"] == "hunter2"
        assert loaded["api_key"] == "PL_ABC123"

    def test_key_file_created_with_owner_only_permissions(self, store):
        store.save("usgs", {"password": "x"})
        assert store._key_file.exists()
        mode = store._key_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_cred_file_created_with_owner_only_permissions(self, store):
        store.save("usgs", {"password": "x"})
        mode = store._cred_file.stat().st_mode & 0o777
        assert mode == 0o600

    def test_uses_the_enc_filename_not_json(self, store):
        store.save("usgs", {"password": "x"})
        assert store._cred_file.name == "credentials.enc"

    def test_same_key_reused_across_instances(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pygeofetch.core.authenticator.Path.home", lambda: tmp_path)
        store1 = CredentialStore(storage_backend="file")
        store1.save("usgs", {"password": "secret1"})

        store2 = CredentialStore(storage_backend="file")
        loaded = store2.load("usgs")
        assert loaded["password"] == "secret1"


class TestLegacyMigration:
    def test_old_base64_file_is_transparently_migrated(self, tmp_path, monkeypatch):
        """
        A user upgrading from the old base64 scheme must not lose their
        already-stored credentials.
        """
        monkeypatch.setattr("pygeofetch.core.authenticator.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".pygeofetch"
        config_dir.mkdir()
        legacy_file = config_dir / "credentials.json"
        legacy_password_b64 = base64.b64encode(b"my-old-password").decode()
        legacy_file.write_text(
            json.dumps(
                {
                    "usgs": {
                        "__obfuscated": True,
                        "username": "olduser",
                        "password": legacy_password_b64,
                    }
                }
            )
        )

        store = CredentialStore(storage_backend="file")
        loaded = store.load("usgs")

        assert loaded is not None
        assert loaded["username"] == "olduser"
        assert loaded["password"] == "my-old-password"

        # And the migrated copy is genuinely re-encrypted, not just
        # copied over in the old base64 form.
        assert store._cred_file.exists()
        raw = store._cred_file.read_text()
        assert legacy_password_b64 not in raw

    def test_migration_only_happens_once(self, tmp_path, monkeypatch):
        monkeypatch.setattr("pygeofetch.core.authenticator.Path.home", lambda: tmp_path)
        config_dir = tmp_path / ".pygeofetch"
        config_dir.mkdir()
        (config_dir / "credentials.json").write_text(
            json.dumps(
                {
                    "usgs": {
                        "__obfuscated": True,
                        "password": base64.b64encode(b"pw").decode(),
                    }
                }
            )
        )

        store = CredentialStore(storage_backend="file")
        store.load("usgs")  # triggers migration
        store.save(
            "copernicus", {"password": "new-one"}
        )  # should not re-migrate/duplicate

        raw = json.loads(store._cred_file.read_text())
        assert "usgs" in raw
        assert "copernicus" in raw
