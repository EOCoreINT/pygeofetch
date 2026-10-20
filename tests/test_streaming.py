"""
Tests for pygeofetch.streaming.

The key claim -- "only the AOI's bytes cross the network" -- is measured, not
assumed: a local HTTP server that implements RFC 7233 range requests serves
real COGs and counts every byte it sends, and the tests compare that to the
file size.
"""

from __future__ import annotations

import http.server
import os
import re
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds

from pygeofetch.models.download_task import DownloadStatus
from pygeofetch.models.satellite_data import SatelliteAsset, SatelliteData
from pygeofetch.streaming import (
    AOIOutsideRasterError,
    StreamingUnsupportedError,
    compute_window,
    read_scene_bands,
    read_window,
    stream_many,
    stream_scene,
    stream_window_to_geotiff,
)

CRS = "EPSG:32630"  # UTM 30N, covers Ghana
ORIGIN_X, ORIGIN_Y = 620000.0, 710000.0


# --------------------------------------------------------------------------
# A real range-request HTTP server that counts bytes served
# --------------------------------------------------------------------------
class _Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.bytes_served = 0
        self.full_gets = 0
        self.range_gets = 0
        self.queries: list[str] = []


def _make_handler(root: Path, stats: _Stats):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def _resolve(self):
            path = self.path.split("?", 1)[0].lstrip("/")
            f = root / path
            return f if f.is_file() else None

        def do_HEAD(self):
            f = self._resolve()
            if f is None:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(f.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()

        def do_GET(self):
            f = self._resolve()
            if f is None:
                self.send_response(404)
                self.end_headers()
                return
            size = f.stat().st_size
            with stats.lock:
                stats.queries.append(self.path.partition("?")[2])
            rng = self.headers.get("Range")
            if rng:
                m = re.match(r"bytes=(\d*)-(\d*)", rng)
                a = int(m.group(1)) if m.group(1) else 0
                b = int(m.group(2)) if m.group(2) else size - 1
                b = min(b, size - 1)
                with open(f, "rb") as fh:
                    fh.seek(a)
                    body = fh.read(b - a + 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
                with stats.lock:
                    stats.range_gets += 1
                    stats.bytes_served += len(body)
            else:
                body = f.read_bytes()
                self.send_response(200)
                with stats.lock:
                    stats.full_gets += 1
                    stats.bytes_served += len(body)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

    return H


def _write_cog(path: Path, size: int, res: float, seed: int, *, dtype="uint16"):
    """A real tiled, deflate-compressed COG with overviews, noisy enough that
    compression does not collapse it to nothing."""
    rng = np.random.default_rng(seed)
    r, c = np.mgrid[0:size, 0:size]
    base = (r * 3 + c * 5) % 20000
    data = (base + rng.integers(0, 900, size=(size, size))).astype(dtype)
    tmp = path.with_suffix(".tmp.tif")
    prof = dict(
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype=dtype,
        crs=CRS,
        transform=from_origin(ORIGIN_X, ORIGIN_Y, res, res),
        nodata=0,
    )
    with rasterio.open(tmp, "w", **prof) as dst:
        dst.write(data, 1)
    rio_copy(
        tmp,
        path,
        driver="COG",
        COMPRESS="DEFLATE",
        BLOCKSIZE=256,
        OVERVIEW_RESAMPLING="NEAREST",
    )
    tmp.unlink()
    return data


@pytest.fixture(scope="module")
def served(tmp_path_factory):
    root = tmp_path_factory.mktemp("cogs")
    d10 = _write_cog(root / "B04.tif", 4096, 10.0, seed=1)  # 40.96 km
    d20 = _write_cog(root / "B11.tif", 2048, 20.0, seed=2)  # same footprint
    (root / "thumb.jpg").write_bytes(b"\xff\xd8not-a-raster")
    (root / "S2.SAFE.zip").write_bytes(b"PK\x03\x04archive")
    stats = _Stats()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(root, stats))
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    yield {"base": base, "root": root, "stats": stats, "d10": d10, "d20": d20}
    srv.shutdown()


# An AOI ~2.5 km square near the middle of the scene, in WGS84.
def _aoi_wgs84():
    cx, cy = ORIGIN_X + 20000, ORIGIN_Y - 20000
    l, b, r, t = transform_bounds(CRS, "EPSG:4326", cx, cy, cx + 2500, cy + 2500)
    return (l, b, r, t)


def _local_window(path: Path, bbox):
    with rasterio.open(path) as src:
        win = compute_window(src, bbox)
        return src.read(1, window=win), src.window_transform(win), win


# --------------------------------------------------------------------------
class TestWindowCorrectness:
    def test_remote_window_is_pixel_identical_to_local_window(self, served):
        bbox = _aoi_wgs84()
        want, want_t, _ = _local_window(served["root"] / "B04.tif", bbox)
        got = read_window(f"{served['base']}/B04.tif", bbox)
        assert got.data.shape == (1, *want.shape)
        assert np.array_equal(got.data[0], want)
        assert got.transform.almost_equals(want_t)
        assert got.decimation == 1

    def test_window_matches_known_pattern_independently(self, served):
        """Not just remote==local: also check against the generating array."""
        bbox = _aoi_wgs84()
        got = read_window(f"{served['base']}/B04.tif", bbox)
        w = got.window
        ref = served["d10"][
            int(w.row_off) : int(w.row_off + w.height),
            int(w.col_off) : int(w.col_off + w.width),
        ]
        assert np.array_equal(got.data[0], ref)

    def test_bbox_in_native_crs_gives_same_window(self, served):
        with rasterio.open(served["root"] / "B04.tif") as src:
            cx, cy = ORIGIN_X + 20000, ORIGIN_Y - 20000
            w_native = compute_window(src, (cx, cy, cx + 2500, cy + 2500), bbox_crs=CRS)
            w_wgs = compute_window(src, _aoi_wgs84())
        # WGS84 round trip may widen by at most one pixel each side
        assert abs(w_native.col_off - w_wgs.col_off) <= 1
        assert abs(w_native.width - w_wgs.width) <= 2

    def test_aoi_is_fully_covered_after_rounding(self, served):
        with rasterio.open(served["root"] / "B04.tif") as src:
            cx, cy = ORIGIN_X + 20003.3, ORIGIN_Y - 19996.7  # off-grid on purpose
            win = compute_window(src, (cx, cy - 2500, cx + 2500, cy), bbox_crs=CRS)
            b = rasterio.windows.bounds(win, src.transform)
        assert b[0] <= cx and b[2] >= cx + 2500 and b[1] <= cy - 2500 and b[3] >= cy

    def test_outside_raises(self, served):
        with pytest.raises(AOIOutsideRasterError):
            read_window(f"{served['base']}/B04.tif", (10.0, 50.0, 10.1, 50.1))

    def test_bad_bbox_raises(self, served):
        with pytest.raises(ValueError):
            read_window(f"{served['base']}/B04.tif", (1.0, 1.0, 0.0, 2.0))


class TestBytesActuallySaved:
    def test_aoi_read_transfers_a_small_fraction_of_the_file(self, served):
        st, size = served["stats"], (served["root"] / "B04.tif").stat().st_size
        st.reset()
        got = read_window(f"{served['base']}/B04.tif", _aoi_wgs84())
        assert st.full_gets == 0, "a full-file GET means range reads are not working"
        assert st.range_gets > 0
        frac = st.bytes_served / size
        assert (
            frac < 0.10
        ), f"transferred {frac:.1%} of the file for a {got.area_fraction:.2%} AOI"

    def test_decimation_uses_overviews_and_moves_fewer_bytes(self, served):
        st = served["stats"]
        st.reset()
        full = read_window(f"{served['base']}/B04.tif", _aoi_wgs84(), decimation=1)
        b_full = st.bytes_served
        st.reset()
        dec = read_window(f"{served['base']}/B04.tif", _aoi_wgs84(), decimation=4)
        b_dec = st.bytes_served
        assert dec.data.shape[1] == int(np.ceil(full.data.shape[1] / 4))
        assert dec.transform.a == pytest.approx(
            full.transform.a * full.data.shape[2] / dec.data.shape[2], rel=1e-6
        )
        assert b_dec < b_full

    def test_max_pixels_guard_auto_decimates(self, served):
        r = read_window(f"{served['base']}/B04.tif", _aoi_wgs84(), max_pixels=10_000)
        assert r.decimation > 1
        assert r.data.size <= 10_000 * 1.5


class TestBlockwiseGeoTiff:
    def test_streamed_file_equals_in_memory_window(self, served, tmp_path):
        bbox = _aoi_wgs84()
        want, want_t, win = _local_window(served["root"] / "B04.tif", bbox)
        info = stream_window_to_geotiff(
            f"{served['base']}/B04.tif",
            bbox,
            tmp_path / "out" / "aoi.tif",
            block_size=64,
        )
        with rasterio.open(info["path"]) as dst:
            assert dst.crs.to_epsg() == 32630
            assert dst.transform.almost_equals(want_t)
            assert np.array_equal(dst.read(1), want)
            assert dst.nodata == 0
        assert info["bytes_written"] < (served["root"] / "B04.tif").stat().st_size / 10
        assert not list((tmp_path / "out").glob("*.part")), "temp file left behind"

    def test_failure_leaves_no_partial_output(self, served, tmp_path):
        with pytest.raises(Exception):
            stream_window_to_geotiff(
                f"{served['base']}/missing.tif", _aoi_wgs84(), tmp_path / "x.tif"
            )
        assert not (tmp_path / "x.tif").exists()
        assert not list(tmp_path.glob("*.part"))


class TestArchivesRejected:
    def test_zip_href_rejected_without_any_request(self, served):
        st = served["stats"]
        st.reset()
        with pytest.raises(StreamingUnsupportedError):
            read_window(f"{served['base']}/S2.SAFE.zip", _aoi_wgs84())
        assert st.range_gets == 0 and st.full_gets == 0

    def test_tar_with_signing_query_rejected(self):
        with pytest.raises(StreamingUnsupportedError):
            read_window("https://x/y/LC08.tar?sig=1", (0, 0, 1, 1))


# --------------------------------------------------------------------------
class _Signer:
    """Stand-in provider whose sign_href appends a query, like a SAS token."""

    def __init__(self):
        self.calls = 0

    def sign_href(self, href, data=None):
        self.calls += 1
        return f"{href}?sig=TOKEN123"


def _scene(base, *, extra_assets=None, provider="stub"):
    def a(k, name, **kw):
        return SatelliteAsset(key=k, href=f"{base}/{name}", **kw)

    assets = {
        "B04": a("B04", "B04.tif", roles=["data"], media_type="image/tiff"),
        "B11": a("B11", "B11.tif", roles=["data"], media_type="image/tiff"),
        "thumbnail": a("thumbnail", "thumb.jpg", roles=["thumbnail"]),
    }
    assets.update(extra_assets or {})
    return SatelliteData(
        id="S2A_TEST_20240101",
        provider=provider,
        collection="sentinel-2-l2a",
        datetime=datetime(2024, 1, 1),
        bbox=(-1.8, 6.1, -1.5, 6.3),
        assets=assets,
    )


class TestSceneStreaming:
    def test_streams_only_requested_bands_signed_and_nested(self, served, tmp_path):
        st = served["stats"]
        st.reset()
        sig = _Signer()
        res = stream_scene(
            _scene(served["base"]), _aoi_wgs84(), tmp_path, bands=["B04"], provider=sig
        )
        assert res.status == DownloadStatus.COMPLETED, res.error
        assert [p.name for p in res.output_paths] == ["S2A_TEST_20240101_B04.tif"]
        assert res.output_paths[0].parent == tmp_path / "stub"  # provider nesting
        assert sig.calls == 1 and all(q == "sig=TOKEN123" for q in st.queries)
        want, _, _ = _local_window(served["root"] / "B04.tif", _aoi_wgs84())
        with rasterio.open(res.output_paths[0]) as d:
            assert np.array_equal(d.read(1), want)

    def test_second_call_resumes_without_network(self, served, tmp_path):
        d = _scene(served["base"])
        stream_scene(d, _aoi_wgs84(), tmp_path, bands=["B04"], provider=_Signer())
        st = served["stats"]
        st.reset()
        res = stream_scene(d, _aoi_wgs84(), tmp_path, bands=["B04"], provider=_Signer())
        assert res.status == DownloadStatus.COMPLETED
        assert st.range_gets == 0 and st.full_gets == 0

    def test_unmatched_band_is_an_error_not_all_bands(self, served, tmp_path):
        st = served["stats"]
        st.reset()
        res = stream_scene(
            _scene(served["base"]),
            _aoi_wgs84(),
            tmp_path,
            bands=["B99"],
            provider=_Signer(),
        )
        assert res.status == DownloadStatus.FAILED
        assert "B99" in res.error
        assert st.range_gets == 0 and not list(tmp_path.rglob("*.tif"))

    def test_non_raster_assets_are_never_streamed(self, served, tmp_path):
        res = stream_scene(
            _scene(served["base"]), _aoi_wgs84(), tmp_path, provider=_Signer()
        )
        names = sorted(p.name for p in res.output_paths)
        assert names == ["S2A_TEST_20240101_B04.tif", "S2A_TEST_20240101_B11.tif"]

    def test_archive_only_scene_fails_cleanly(self, served, tmp_path):
        d = SatelliteData(
            id="S2_ARCHIVE",
            provider="copernicus",
            assets={
                "product": SatelliteAsset(
                    key="product", href=f"{served['base']}/S2.SAFE.zip", roles=["data"]
                )
            },
        )
        res = stream_scene(d, _aoi_wgs84(), tmp_path, provider=_Signer())
        assert res.status == DownloadStatus.FAILED
        assert res.error_type == "streaming_unsupported"

    def test_one_failing_band_does_not_lose_the_others(self, served, tmp_path):
        bad = SatelliteAsset(
            key="B08",
            href=f"{served['base']}/nope.tif",
            roles=["data"],
            media_type="image/tiff",
        )
        res = stream_scene(
            _scene(served["base"], extra_assets={"B08": bad}),
            _aoi_wgs84(),
            tmp_path,
            bands=["B04", "B08"],
            provider=_Signer(),
        )
        assert res.status == DownloadStatus.COMPLETED
        assert [p.name for p in res.output_paths] == ["S2A_TEST_20240101_B04.tif"]
        assert "B08" in (res.error or "")

    def test_mixed_resolution_bands_keep_their_own_grids(self, served):
        got = read_scene_bands(
            _scene(served["base"]), _aoi_wgs84(), ["B04", "B11"], provider=_Signer()
        )
        a, b = got["B04"], got["B11"]
        assert a.data.shape[1] == pytest.approx(b.data.shape[1] * 2, abs=2)
        assert a.transform.a == pytest.approx(10.0) and b.transform.a == pytest.approx(
            20.0
        )
        # Same ground footprint (within a pixel of the coarser grid)
        for x, y in zip(a.bounds, b.bounds):
            assert abs(x - y) <= 20.0

    def test_stream_many_preserves_order_and_isolates_failures(self, served, tmp_path):
        ok = _scene(served["base"])
        bad = SatelliteData(id="BAD", provider="stub", assets={})
        out = stream_many(
            [ok, bad, ok],
            _aoi_wgs84(),
            tmp_path,
            ["B04"],
            parallel=3,
            provider=_Signer(),
        )
        assert [r.status for r in out] == [
            DownloadStatus.COMPLETED,
            DownloadStatus.FAILED,
            DownloadStatus.COMPLETED,
        ]


class TestProviderHooks:
    def test_base_provider_sign_href_is_identity(self):
        from pygeofetch.providers.base import AbstractBaseProvider

        assert (
            AbstractBaseProvider.sign_href(object(), "https://a/b.tif")
            == "https://a/b.tif"
        )

    def test_planetary_computer_signs_and_caches_token(self, monkeypatch):
        from pygeofetch.providers.planetary_computer import PlanetaryComputerProvider

        p = PlanetaryComputerProvider()
        calls = []
        monkeypatch.setattr(
            p, "_get_sas_token", lambda c: calls.append(c) or "se=2030&sig=abc"
        )
        d = SatelliteData(
            id="x", provider="planetary_computer", collection="sentinel-2-l2a"
        )
        u1 = p.sign_href("https://blob/x/B04.tif", d)
        u2 = p.sign_href("https://blob/x/B08.tif?a=1", d)
        assert u1 == "https://blob/x/B04.tif?se=2030&sig=abc"
        assert u2 == "https://blob/x/B08.tif?a=1&se=2030&sig=abc"
        assert calls == ["sentinel-2-l2a"], "one token fetch must serve many bands"

    def test_engine_stream_facade_end_to_end(self, served, tmp_path, monkeypatch):
        from pygeofetch import PyGeoFetch

        monkeypatch.setenv("HOME", str(tmp_path))
        client = PyGeoFetch(log_level="ERROR")
        res = client.stream(
            _scene(served["base"], provider="element84"),
            _aoi_wgs84(),
            tmp_path / "raw",
            bands=["B04"],
        )
        assert len(res) == 1 and res[0].status == DownloadStatus.COMPLETED, res[0].error
        assert res[0].output_paths[0].parent.name == "element84"


# --------------------------------------------------------------------------
class TestCoverageAndNaming:
    def _edge_bbox_native(self):
        right = ORIGIN_X + 4096 * 10.0
        mid_y = ORIGIN_Y - 20000
        return (
            right - 1250,
            mid_y,
            right + 1250,
            mid_y + 2500,
        )  # half outside the scene

    def test_straddling_aoi_reports_partial_coverage(self, served):
        r = read_window(
            f"{served['base']}/B04.tif", self._edge_bbox_native(), bbox_crs=CRS
        )
        assert r.aoi_coverage == pytest.approx(0.5, abs=0.02)
        assert r.data.shape[2] == pytest.approx(
            125, abs=2
        )  # only the inside half is read

    def test_fully_inside_aoi_reports_full_coverage(self, served):
        assert (
            read_window(f"{served['base']}/B04.tif", _aoi_wgs84()).aoi_coverage == 1.0
        )

    def test_stream_scene_surfaces_partial_coverage_and_on_resume(
        self, served, tmp_path
    ):
        d = _scene(served["base"])
        for _ in range(
            2
        ):  # second pass is a resume (no network) and must report the same
            res = stream_scene(
                d,
                self._edge_bbox_native(),
                tmp_path,
                ["B04"],
                provider=_Signer(),
                bbox_crs=CRS,
            )
            assert res.status == DownloadStatus.COMPLETED
            assert res.aoi_coverage == pytest.approx(0.5, abs=0.03)

    def test_full_coverage_for_normal_scene(self, served, tmp_path):
        res = stream_scene(
            _scene(served["base"]), _aoi_wgs84(), tmp_path, ["B04"], provider=_Signer()
        )
        assert res.aoi_coverage == pytest.approx(1.0, abs=0.01)

    def test_streamed_band_paths_resolve_aliases(self):
        from pygeofetch.streaming import streamed_band_paths

        def mk(k):
            return SatelliteAsset(
                key=k,
                href=f"https://h/{k}.tif",
                roles=["data"],
                media_type="image/tiff",
            )

        d = SatelliteData(
            id="E84 scene/1",
            provider="element84",
            assets={
                k: mk(k) for k in ["blue", "green", "red", "nir", "swir16", "swir22"]
            },
        )
        want = ["B02", "B03", "B04", "B08", "B11", "B12", "B99"]
        got = streamed_band_paths(d, want, "/out")
        assert sorted(got) == [
            "B02",
            "B03",
            "B04",
            "B08",
            "B11",
            "B12",
        ], "B99 must be omitted"
        assert (
            got["B04"].name == "E84_scene_1_red.tif"
            and got["B04"].parent.name == "element84"
        )
        assert got["B11"].name.endswith("_swir16.tif") and got["B12"].name.endswith(
            "_swir22.tif"
        )
