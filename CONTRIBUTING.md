# Contributing to PyGeoFetch

Thank you for your interest in contributing! PyGeoFetch is an open-source
project and we welcome all contributions — bug fixes, new provider implementations,
documentation improvements, and test coverage.

---

## Getting Started

### 1. Fork and clone

```bash
git clone git@github.com:EOCoreINT/pygeofetch.git
cd pygeofetch
git remote add upstream git@github.com:EOCoreINT/pygeofetch.git
```

### 2. Set up development environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[dev,all]"
```

### 3. Run the tests

```bash
pytest tests/ -v --cov=pygeofetch --cov-report=term-missing
```

All tests must pass before submitting a pull request.

---

## How to Add a New Provider

Adding a provider is the most valuable contribution. Here's how:

### 1. Create `pygeofetch/providers/my_provider.py`

Use an existing stub (e.g. `opentopography.py`) as a template:

```python
from pygeofetch.providers.base import AbstractBaseProvider, ProviderCapabilities
from pygeofetch.models.satellite_data import SatelliteData
from pygeofetch.models.search_query import SearchQuery
from pygeofetch.models.download_task import DownloadOptions, DownloadResult
from pygeofetch.models.user_auth import AuthSession

class MyProvider(AbstractBaseProvider):
    PROVIDER_ID = "my_provider"
    REQUIRES_AUTH = True   # or False

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            provider_id=self.PROVIDER_ID,
            name="My Provider",
            auth_type="api_key",
            satellites=["MySat-1", "MySat-2"],
            max_cloud_cover=100,
            supports_preview=False,
        )

    def authenticate(self, credentials: dict) -> AuthSession:
        # Validate and exchange credentials for a session
        ...

    def search(self, query: SearchQuery) -> list[SatelliteData]:
        # Call provider API and return SatelliteData items
        ...

    def download(self, data: SatelliteData, destination, options: DownloadOptions) -> DownloadResult:
        # Download and return DownloadResult
        ...
```

### 2. Register the provider

In `pygeofetch/providers/__init__.py`, add your provider to the `PROVIDER_REGISTRY`:

```python
"my_provider": ("pygeofetch.providers.my_provider", "MyProvider"),
```

Also add it to `PROVIDER_INFO` with metadata, and to `FREE_PROVIDERS` if it requires no auth.

### 3. Populate real geometry, not just bbox

**A real, systematic issue was found and fixed across the provider layer**:
18 of 24 existing providers were populating `SatelliteData.bbox` but never
`SatelliteData.geometry` — even when their own real API responses already
carried the real, precise footprint shape needed to build it. A rectangular
bbox is always an approximation; if your provider's real search response
includes an actual polygon/footprint field, populate `geometry` with it
directly, not just its bounding box:

```python
return SatelliteData(
    id=item_id,
    provider=self.PROVIDER_ID,
    bbox=bbox,
    geometry=item.get("geometry"),  # real, precise footprint — don't skip this
    ...
)
```

If the provider genuinely has no footprint field (some APIs only ever
return a bbox), leave `geometry=None` — `MapViewer.add_search_results()`
already falls back to a bbox-derived rectangle automatically, so this is
safe to omit when there's truly nothing better available. What isn't safe
is silently discarding a real, precise footprint the API already gave you.

**Verify the real field name and structure against the provider's own live
API response or official documentation — never guess.** Two real,
representative examples from this project's own history: a Copernicus fix
that had to correct a guessed field name (`Footprint`, a WKT string that
never existed) against the real, current field (`GeoFootprint`, a
structured GeoJSON object, confirmed only by checking Copernicus's actual
API documentation directly); and a NASA Earthdata fix that added support
for a real, confirmed `polygons` field that had simply never been read at
all, found only by checking two independent, real working examples of the
live API response. Both bugs would have been caught immediately by testing
against a real or format-accurate response instead of assuming a
reasonable-sounding field name.

### 4. Write tests

Add tests in `tests/unit/test_providers.py` and an integration test with VCR
cassettes in `tests/integration/`. If your provider parses `geometry`, add
a real or format-accurate test case confirming it's actually extracted —
`tests/unit/test_provider_geometry_audit.py` is a real, working reference
for this pattern, including how to test the graceful fallback when a
result has no geometry at all.

### 5. Document

Add a row to the provider table in `README.md` and create a doc page at
`docs/providers/my_provider.md` with authentication instructions.

---

## Code Style

- **PEP 8** compliance enforced by `ruff`
- **Type hints** on all public functions and methods
- **Docstrings** on all classes and public methods (Google style)
- **Black** for formatting (`black pygeofetch/ tests/`)
- **mypy** for static type checking

Run all checks:

```bash
ruff check pygeofetch/
black --check pygeofetch/ tests/
mypy pygeofetch/
```

---

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/my-provider
   ```

2. **Write the code** with tests and docstrings.

3. **Run the full test suite** and ensure >80% coverage for new code.

4. **Update CHANGELOG.md** with a brief description of your change.

5. **Submit a PR** with:
   - A clear description of what was added/changed
   - Links to any relevant provider API docs
   - Test results (paste `pytest --tb=short` output)

---

## Reporting Bugs

Please open an issue with:
- PyGeoFetch version (`pygeofetch --version`)
- Python version
- Exact command run or code snippet
- Full error traceback
- Expected vs. actual behaviour

---

## Security Issues

Do **not** open public issues for security vulnerabilities.
Please email `security@example.com` with details.

---

## Code of Conduct

Be respectful and constructive. We follow the
[Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct.

---

Thank you for contributing! 🛰️