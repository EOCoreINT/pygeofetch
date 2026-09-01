# Full CLI Reference

## Global options

```
pygeofetch [OPTIONS] COMMAND [ARGS]

Options:
  --log-level TEXT   DEBUG, INFO, WARNING, ERROR  [default: INFO]
  --log-file TEXT    Write logs to file path
  --log-format TEXT  console or json  [default: console]
  --config FILE      Path to config file
  --version          Show version and exit
  --help, -h         Show help message and exit
```

## All command groups

| Group | Subcommands | Description |
|---|---|---|
| `auth` | add, login, list, test, remove, export | Manage provider credentials |
| `providers` | list, info, search | Browse and inspect providers |
| `search` | run | Search for satellite scenes |
| `download` | run | Download scenes to disk |
| `cache` | stats, clear | Manage search result cache |
| `pipeline` | run, validate, schedule, list-scheduled, unschedule, logs, history, retry | Pipeline orchestration |
| `config` | show, get, set, path, reset | Read and modify configuration |
| `status` | — | System status dashboard |
| `doctor` | — | Diagnose installation and connectivity |
| `version` | — | Show version info |

## Cache commands

```bash
pygeofetch cache stats [--json]
pygeofetch cache clear [--provider PROVIDER]
```

## Providers commands

```bash
pygeofetch providers list
pygeofetch providers list --no-auth
pygeofetch providers list --capabilities sar
pygeofetch providers info planetary_computer
pygeofetch providers search "landsat"
```

## Config commands

```bash
pygeofetch config show
pygeofetch config get download.parallel
pygeofetch config set download.parallel 8
pygeofetch config path
pygeofetch config reset
```

## Shell completion

```bash
# Bash
pygeofetch --install-completion bash

# Zsh / Fish also supported
pygeofetch --install-completion zsh
pygeofetch --install-completion fish
```

See {doc}`/core-features/search`, {doc}`/core-features/download`,
{doc}`/core-features/authentication`, and {doc}`/reference/pipelines`
for each group's full flag reference and examples.
