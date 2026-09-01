# Doctor / Diagnostics

```bash
pygeofetch doctor
```

Runs a full diagnostic pass:

- Python version compatibility
- Which optional dependency groups (`geo`, `insar`, `sar`, `viz`, ...)
  are actually importable in this environment
- System keyring availability (flags headless environments where auth
  storage needs an environment-variable fallback — see
  {doc}`/core-features/authentication`)
- Live connectivity check against a couple of open, no-auth providers

## Status dashboard

```bash
pygeofetch status
pygeofetch status --json
```

A snapshot of registered providers, authentication status per
provider, and cache statistics — useful for confirming what a given
environment can actually reach before kicking off a real pipeline run.
