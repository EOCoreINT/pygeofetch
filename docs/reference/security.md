# Security Model

pygeofetch is designed with credential safety, network security, and
data integrity as first-class concerns.

```{note}
**Previously base64, now genuinely encrypted.** The file-backend
credential store used to only base64-encode sensitive fields (its own
source comment read `# Basic obfuscation (not encryption)`) — trivially
reversible by anyone who could read the file, with no key required.
It now uses real Fernet symmetric encryption: a random key is
generated on first use and stored separately at
`~/.pygeofetch/credentials.key` (`chmod 0600`), and credentials are
written to `~/.pygeofetch/credentials.enc` (`chmod 0600`) with
sensitive fields (`password`, `api_key`, `token`, `secret_key`,
`client_secret`) genuinely encrypted, not just encoded. Existing users
upgrading from the old base64 file are migrated transparently on
first use — the old `credentials.json` is left in place, unused, and
can be deleted manually.

**`PyGeoFetch()`'s default backend is still `auth_backend="file"`**
(not keyring) — that default itself wasn't changed here, only what
"file" storage actually does to protect the data. Prefer
`PyGeoFetch(auth_backend="keyring")` when a real OS keyring is
available, since it hands storage off to the OS rather than a
locally-generated key file.
```

## Credential handling

- System keyring storage is available and works correctly when
  explicitly selected: macOS Keychain, Windows Credential Manager,
  Linux Secret Service (via the `keyring` package)
- The file-backend fallback (used by default, and for headless
  environments with no keyring daemon) now genuinely encrypts
  sensitive fields with Fernet — see the note above
- Environment variable support for CI/CD and Docker (`PYGEOFETCH_*`,
  see {doc}`/reference/configuration`)

## Network security

- TLS is used for all outbound provider connections via `httpx`/`requests`
- `security.verify_ssl: true` is the real, verified default in
  `pygeofetch/config/settings.py` — no code path was found that sets
  `verify=False`
- HTTP proxy support via `proxy.http` / `proxy.https` config, or the
  standard `HTTP_PROXY` / `HTTPS_PROXY` environment variables
- Per-provider connection timeouts (`timeout_seconds`, default 300 for
  downloads, 60 for provider API calls — see
  {doc}`/reference/configuration`)

## Data integrity

- SHA256 (or MD5/SHA512 — see `ChecksumAlgorithm` in
  {doc}`/reference/python-api`) checksum verification on downloads
  when `verify_checksum=True`
- Resume support via HTTP range requests, tracked per scene
- `overwrite: false` by default — existing files are skipped, not
  silently clobbered

## Reporting vulnerabilities

Do not open public GitHub issues for security vulnerabilities. See the
repository's `SECURITY.md` (or `CONTRIBUTING.md` if no separate
security policy exists) for the current disclosure contact and
response-time commitment.
