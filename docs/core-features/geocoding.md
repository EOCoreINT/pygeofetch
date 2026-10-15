# Geocoding

Turn a place name into coordinates, or coordinates into a place name —
without needing a real bbox or lat/lon in hand before you start.
Everything else in pygeofetch assumes you already have real, numeric
coordinates; this is the one piece that gets you there from a name.

:::warning Verified against official docs, not yet against a live call
Every other feature in this documentation was tested against real,
live execution. This one is verified differently, and it's worth
being precise about the difference: the parsing logic (field names,
and critically the `boundingbox` ordering) has been checked directly
against
[Nominatim's own official, current API documentation](https://nominatim.org/release-docs/latest/api/Output/)
— not just a response shape assumed from memory — so the code's
*logic* is confirmed correct against the authoritative source. What
hasn't happened is an actual live HTTP round-trip against the real
service — real network behavior, actual rate-limit responses, and
any server-side quirks remain unverified. Reasonably confident this
works; genuinely test it in your own environment before depending on
it for anything production-critical.
:::

## Why Nominatim specifically

Uses [Nominatim](https://nominatim.openstreetmap.org) (OpenStreetMap's
free geocoding service) because it requires no API key — consistent
with pygeofetch's other no-account-needed providers. Real usage policy
limits free requests to 1/second; this is enforced automatically, not
left for you to violate accidentally.

## Quick start

```python
from pygeofetch.utils.geocoding import geocode, reverse_geocode
from pygeofetch.models import SearchQuery

results = geocode("Nairobi, Kenya")
place = results[0]
print(place["display_name"])   # confirm it found the right place
print(place["bbox"])          # (min_lon, min_lat, max_lon, max_lat)

# Feed the bbox directly into a real search
query = SearchQuery(bbox=place["bbox"], start_date="2024-01-01", end_date="2024-06-30")
```

## Forward geocoding — name to coordinates

```python
results = geocode("Eiffel Tower", limit=3)
```

Returns a list of dicts, one per real match (Nominatim may return
several for an ambiguous query like a common place name) — empty list
if nothing real matched, never an exception for "not found." Each
result has:

| Key | Real content |
|---|---|
| `lat`, `lon` | Real WGS84 coordinates of the match's center point |
| `bbox` | `(min_lon, min_lat, max_lon, max_lat)` — ready to pass directly to `SearchQuery` |
| `display_name` | The full real matched place name — check this to confirm the geocoder understood your query correctly, especially for ambiguous names |
| `place_type` | Nominatim's own real classification (`city`, `building`, `natural`, etc.) |

**Always check `display_name`** before trusting a result for an
ambiguous query — "Springfield" alone will match *a* Springfield, not
necessarily the one you meant.

## Reverse geocoding — coordinates to name

```python
place = reverse_geocode(-1.286389, 36.817223)
if place:
    print(place["display_name"])
    print(place["address"])  # structured breakdown: country, city, road, etc.
```

Returns `None` if nothing real is found at that point (open ocean, for
instance) — check for this before accessing the result, rather than
assume every coordinate resolves to a real place.

## CLI reference

```bash
pygeofetch geocode "Nairobi, Kenya"
pygeofetch geocode "Springfield" --limit 5   # see every real ambiguous match

pygeofetch reverse-geocode -1.286389 36.817223
```

## Common pitfalls

- **Trusting the first result for an ambiguous name without checking
  `display_name`.** Common city/street names exist in many countries —
  always look at what actually matched, not just its coordinates.
- **Geocoding in a tight loop without spacing requests out.** The
  built-in 1-second throttle handles this automatically within a
  single process, but running many parallel processes against
  Nominatim simultaneously can still violate the real, shared usage
  policy — Nominatim can and does rate-limit or block abusive traffic.
- **Assuming `reverse_geocode` always returns something.** Points over
  water, polar regions, or other real areas with no OSM data return
  `None` — a real, valid outcome, not a bug.
- **Not handling network failures.** The parsing logic is verified
  against Nominatim's own official documentation, but no live request
  has been made in this project's own testing — wrap real calls in
  your own error handling until you've confirmed actual runtime
  behavior in your environment.

## Full function reference

| Function | Real purpose |
|---|---|
| `geocode(query, limit=1, timeout=10.0)` | Place name → coordinates + bbox |
| `reverse_geocode(lat, lon, timeout=10.0)` | Coordinates → place name |

Both live in `pygeofetch.utils.geocoding` and are not currently
attached to the main `PyGeoFetch` client object (e.g. there is no
`client.geocode(...)`) — import them directly as shown above.
