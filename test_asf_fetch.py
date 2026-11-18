"""Check that pygeofetch can find (and optionally download and read) NISAR data from ASF.

Steps, each printed with OK / FAIL:

1. the pygeofetch ASF provider in use has the NISAR patch;
2. ASF itself returns NISAR products for the area (direct query, no pygeofetch);
3. pygeofetch finds the same products;
4. (optional, --download N) pygeofetch downloads N products with your Earthdata login;
5. (optional, after 4) the first file opens, its layers are listed, and PhysInSAR's reader reads it.

Usage (from the physinsar folder, inside the venv):

    python case_studies/01_mexico_city/test_asf_fetch.py
    python case_studies/01_mexico_city/test_asf_fetch.py --download 1
    python case_studies/01_mexico_city/test_asf_fetch.py --level GSLC --no-pr

Earthdata credentials for --download: already stored in pygeofetch, or in the
environment variables EARTHDATA_USERNAME / EARTHDATA_PASSWORD.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

ASF_SEARCH = "https://api.daac.asf.alaska.edu/services/search/param"
PROVIDER = "alaska_satellite_facility"


def ok(msg: str) -> None:
    print(f"  OK    {msg}")


def fail(msg: str) -> None:
    print(f"  FAIL  {msg}")


def asf_params(args: argparse.Namespace) -> dict:
    lo_lon, lo_lat, hi_lon, hi_lat = args.bbox
    params = {"dataset": "NISAR", "processingLevel": args.level}
    if not args.no_pr:
        params["productionConfiguration"] = "PR"
    return params | {
        "intersectsWith": f"POLYGON(({lo_lon} {lo_lat},{hi_lon} {lo_lat},{hi_lon} {hi_lat},{lo_lon} {hi_lat},{lo_lon} {lo_lat}))",
        "start": f"{args.start}T00:00:00UTC",
        "end": f"{args.end}T23:59:59UTC",
        "output": "geojson",
        "maxResults": args.max_results,
    }


def describe(props: dict) -> str:
    keys = ("startTime", "flightDirection", "pathNumber", "frameNumber", "polarization")
    parts = []
    for k in keys:
        v = props.get(k)
        if v is not None:
            parts.append(f"{k}={str(v)[:19] if k == 'startTime' else v}")
    size = props.get("bytes")
    if isinstance(size, dict):  # NISAR: one entry per file of the product; report the main .h5
        size = sum(v.get("bytes", 0) for k, v in size.items() if k.endswith(".h5") and "QA" not in k)
    if isinstance(size, (int, float)) and size:
        parts.append(f"size={size / 1e9:.1f} GB")
    return ", ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox", type=float, nargs=4, default=[-99.183, 19.278, -99.003, 19.438],
                        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"))
    parser.add_argument("--start", default="2026-06-17")
    parser.add_argument("--end", default="2026-10-31")
    parser.add_argument("--level", default="GUNW", help="NISAR product level: GUNW, GSLC, RIFG, GCOV, ...")
    parser.add_argument("--no-pr", action="store_true", help="do not restrict to productionConfiguration=PR")
    parser.add_argument("--max-results", type=int, default=200)
    parser.add_argument("--download", type=int, default=0, metavar="N", help="download the first N products")
    parser.add_argument("--out", type=Path, default=Path("data/asf_test"))
    parser.add_argument("--direction", choices=["ASCENDING", "DESCENDING"], help="keep only this flight direction")
    args = parser.parse_args()
    all_ok = True

    # 1 -------------------------------------------------------------- patched provider?
    print("\n1. pygeofetch ASF provider")
    try:
        import pygeofetch
        import pygeofetch.providers.alaska_satellite_facility as asf_module
    except Exception as exc:  # e.g. pygeofetch 2.7.2.0 needs shapely at import
        fail(f"cannot import pygeofetch: {exc!r}  (try: pip install shapely)")
        return 1
    source = inspect.getsource(asf_module)
    print(f"        pygeofetch {getattr(pygeofetch, '__version__', '?')} from {asf_module.__file__}")
    if "api_params" in source and "unparseable" in source:
        ok("NISAR patch and robust parsing are active")
    else:
        fail("the provider is not patched: apply pygeofetch_asf_nisar.patch to the file above, then re-run")
        all_ok = False

    # 2 -------------------------------------------------------------- ASF directly
    print("\n2. ASF Search API, direct (no pygeofetch)")
    import httpx

    params = asf_params(args)
    shown = {k: v for k, v in params.items() if k != "intersectsWith"}
    print(f"        query: {shown}")
    try:
        resp = httpx.get(ASF_SEARCH, params=params, timeout=90)
        data = resp.json() if resp.status_code == 200 else None
    except Exception as exc:
        fail(f"request failed: {exc!r}")
        return 1
    feats = (data or {}).get("features") or []
    if resp.status_code != 200:
        fail(f"HTTP {resp.status_code}: {resp.text[:300]}")
        return 1
    if feats:
        ok(f"{len(feats)} {args.level} products")
        for f in feats[:5]:
            props = f.get("properties") or {}
            print(f"          {props.get('sceneName') or props.get('fileID')}: {describe(props)}")
        print("        property names of the first product:")
        print("          " + ", ".join(sorted((feats[0].get("properties") or {}).keys())))
    else:
        fail("ASF returned no products. Try --no-pr, a wider --start/--end, or another --level")
        print("        raw answer: " + json.dumps(data)[:500])
        all_ok = False

    # 3 -------------------------------------------------------------- through pygeofetch
    print("\n3. pygeofetch search")
    from pygeofetch import PyGeoFetch
    from pygeofetch.models.search_query import BoundingBox, SearchQuery

    api_params = {k: params[k] for k in ("dataset", "processingLevel", "productionConfiguration") if k in params}
    query = SearchQuery(
        bbox=BoundingBox.from_string(",".join(map(str, args.bbox))),
        start_date=args.start,
        end_date=args.end,
        max_results=args.max_results,
        provider_filters={PROVIDER: {"api_params": api_params}},
    )
    client = PyGeoFetch()
    items = client.search(query, providers=[PROVIDER])
    if args.direction:
        items = [i for i in items if str((i.properties or {}).get("flightDirection", "")).upper() == args.direction]
        feats = [f for f in feats if str((f.get("properties") or {}).get("flightDirection", "")).upper() == args.direction]
        print(f"        keeping {args.direction} only")
    if items and len(items) == len(feats):
        ok(f"{len(items)} products, same as ASF directly")
    elif items:
        ok(f"{len(items)} products (ASF directly: {len(feats)}; the difference is skipped or capped results)")
    else:
        fail("pygeofetch found nothing although ASF did" if feats else "pygeofetch found nothing (nor did ASF)")
        all_ok = False
    for it in items[:3]:
        href = next((a.href for a in (it.assets or {}).values() if getattr(a, "href", None)), None)
        print(f"          {it.id}  {str(it.datetime)[:19]}  {href}")

    # 4 -------------------------------------------------------------- download
    if args.download and items:
        print(f"\n4. download {args.download} product(s) to {args.out}")
        from pygeofetch.models.download_task import DownloadOptions

        user, password = os.environ.get("EARTHDATA_USERNAME"), os.environ.get("EARTHDATA_PASSWORD")
        if user and password:
            client.add_credentials(PROVIDER, username=user, password=password)
        args.out.mkdir(parents=True, exist_ok=True)
        results = client.download(items[: args.download], destination=args.out, options=DownloadOptions(parallel=2, resume=True))
        files = []
        for r in results:
            if getattr(r, "output_path", None):
                paths = getattr(r, "output_paths", None) or [r.output_path]
                files += [Path(p) for p in paths]
                ok(f"{r.data_id}: {', '.join(f'{Path(p).name} ({Path(p).stat().st_size / 1e6:.0f} MB)' for p in paths)}")
            else:
                fail(f"{getattr(r, 'data_id', '?')}: {getattr(r, 'status', '')} {getattr(r, 'error', '')}")
                all_ok = False

        # 5 ---------------------------------------------------------- open and read
        h5_files = [p for p in files if p.suffix == ".h5"]
        if h5_files:
            print(f"\n5. inspect {h5_files[0].name}")
            try:
                import h5py
            except ImportError:
                fail("h5py not installed: pip install h5py pyproj")
                return 1
            with h5py.File(h5_files[0], "r") as h5:
                names: list[str] = []
                h5.visititems(lambda n, o: names.append(f"{n} {getattr(o, 'shape', '')} {getattr(o, 'dtype', '')}")
                              if hasattr(o, "shape") and o.ndim >= 2 else None)
                print("        2-D/3-D layers:")
                for n in names[:40]:
                    print(f"          {n}")
                wrapped = [n for n in names if "wrappedInterferogram" in n]
                (ok if wrapped else fail)(f"wrapped interferogram layers found: {len(wrapped)}")
            if args.level.upper() == "GUNW":
                try:
                    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
                    from physinsar.realdata import read_nisar_gunw

                    stack = read_nisar_gunw(h5_files[:1], args.bbox)
                    rec = stack.interferograms[0]
                    up = rec.los_enu[:, 2]
                    ok(f"PhysInSAR read it: {rec.reference_date} -> {rec.secondary_date}, {rec.phase.size} pixels, "
                       f"wavelength {stack.wavelength_m * 100:.1f} cm, mean coherence {rec.coherence.mean():.2f}, "
                       f"LOS east {rec.los_enu[:, 0].mean():+.2f} north {rec.los_enu[:, 1].mean():+.2f} up {up.mean():.2f}, "
                       f"ionosphere prior {'yes' if rec.iono_phase is not None else 'no'}")
                except Exception as exc:
                    fail(f"PhysInSAR could not read it: {exc!r}")
                    all_ok = False

    print("\nAll checks passed." if all_ok else "\nSome checks failed (see FAIL above).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())