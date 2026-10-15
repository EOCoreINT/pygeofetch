# Security

## Credential storage

pygeofetch stores provider credentials with one of two backends, selected by the
`auth_backend` argument of `PyGeoFetch()` (not by configuration; see
[Configuration](configuration.md)):

| `auth_backend` | Storage | Notes |
|---|---|---|
| `"file"` (**default**) | `~/.pygeofetch/credentials.enc` | Sensitive fields (`password`, `api_key`, `token`, `secret_key`, `client_secret`) are encrypted with Fernet. The key is generated on first use and stored separately in `~/.pygeofetch/credentials.key`. Both files are created with mode `0600`. |
| `"keyring"` | The operating-system keyring | macOS Keychain, Windows Credential Manager, or the Linux Secret Service, through the `keyring` package. |

Prefer `PyGeoFetch(auth_backend="keyring")` when a keyring is available, because it hands
storage to the operating system instead of a key file sitting next to the encrypted data.
Anyone who can read both files in `~/.pygeofetch/` can decrypt the file backend.

Earlier releases only base64-encoded the file backend. Existing users are migrated on first
use; the old `credentials.json` is left in place, unused, and can be deleted.

```python
from pygeofetch import PyGeoFetch

client = PyGeoFetch(auth_backend="keyring")
client.add_credentials("planet", api_key="PL_KEY")
```

### Credentials and environment variables

Credentials are **not** loaded from environment variables. In CI or a container, read your own
secrets and pass them to `add_credentials()`. See
[Authentication](../data-access/authentication.md).

### Exporting credentials

`pygeofetch auth export` writes stored credentials to a JSON file for backup. That file
contains secrets in plain text. Store it securely and delete it when done.

## Network security

- Provider connections use TLS through `httpx` or `requests`. No code path sets `verify=False`.
- Standard `HTTP_PROXY` and `HTTPS_PROXY` environment variables are honoured by the HTTP clients.
- Provider requests have per-request timeouts (60 seconds by default for searches; configurable with `search run --timeout`).

The `security.verify_ssl` and `proxy.*` entries in the settings model are not read by the
search and download code. See [Configuration](configuration.md#what-controls-behaviour).

## Data integrity

- With `verify_checksum=True`, a downloaded file is compared with the checksum the provider supplied, when there is one. A mismatch is logged and recorded on the result. See [Error handling](error-handling.md#download-resilience).
- Files that fail a post-download validation (for example truncation) are retried.
- Existing files are skipped, not overwritten, unless `overwrite=True`.
- Archive extraction (`unzip`) rejects paths that would escape the destination directory.

## Reporting vulnerabilities

Do not open public GitHub issues for security vulnerabilities. See the repository's
`SECURITY.md`, or `CONTRIBUTING.md` if there is no separate policy, for the disclosure
contact.
