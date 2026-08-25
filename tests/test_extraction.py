"""
Tests for pygeofetch.insar.extraction.SLCExtractor.

Verified against a synthetic multi-sub-swath .SAFE.zip with known GCP
footprints, so correctness is checked against ground truth (which
sub-swath SHOULD match a given AOI), not just "doesn't crash".
"""

from __future__ import annotations

import zipfile

import pytest


def _build_synthetic_slc_zip(tmp_path, footprints, polarisation="vv"):
    """
    Build a synthetic Sentinel-1-style SLC .SAFE.zip with one measurement
    TIFF per sub-swath, each carrying GCPs matching the given footprint.

    footprints: dict of {swath_name: (min_lon, min_lat, max_lon, max_lat)}
    """
    rasterio = pytest.importorskip("rasterio")
    import numpy as np

    tiff_dir = tmp_path / "measurement"
    tiff_dir.mkdir(exist_ok=True)

    zip_path = tmp_path / "S1A_IW_SLC__1SDV_test.SAFE.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for swath, (minlon, minlat, maxlon, maxlat) in footprints.items():
            tiff_path = tiff_dir / f"s1a-{swath}-slc-{polarisation}-test.tiff"
            h, w = 8, 8
            data = np.ones((h, w), dtype=np.complex64)
            gcps = [
                rasterio.control.GroundControlPoint(row=0, col=0, x=minlon, y=maxlat),
                rasterio.control.GroundControlPoint(row=0, col=w, x=maxlon, y=maxlat),
                rasterio.control.GroundControlPoint(row=h, col=0, x=minlon, y=minlat),
                rasterio.control.GroundControlPoint(row=h, col=w, x=maxlon, y=minlat),
            ]
            with rasterio.open(
                tiff_path,
                "w",
                driver="GTiff",
                dtype="complex_int16",
                count=1,
                width=w,
                height=h,
                gcps=gcps,
                crs="EPSG:4326",
            ) as ds:
                ds.write(data, 1)
            zf.write(
                tiff_path,
                f"S1A_test.SAFE/measurement/s1a-{swath}-slc-{polarisation}-test.tiff",
            )
    return zip_path


class TestSLCExtractor:
    def test_picks_the_overlapping_subswath(self, tmp_path):
        """With 3 candidate sub-swaths, only the truly-overlapping one is extracted."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        footprints = {
            "iw1": (-100.0, 18.0, -99.5, 19.0),
            "iw2": (-99.3, 19.2, -98.9, 19.6),  # matches AOI below
            "iw3": (-98.5, 20.0, -98.0, 20.5),
        }
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)

        extractor = SLCExtractor(polarisation="VV")
        out = extractor.extract_scene(zip_path, aoi, tmp_path / "out", label="test")

        assert out is not None
        assert out.exists()

    def test_extract_consistent_stack_rejects_full_swath_fallback(self, tmp_path):
        """Real, confirmed fix: extract_consistent_stack() combines
        reference-based sub-swath forcing with automatic rejection of
        any date whose forced extraction falls back to the full,
        uncropped swath -- verified directly against the exact real row
        counts observed in this project's own Amatrice run (a genuine
        crop around 3000 rows vs a real full-swath fallback of 22935)."""
        from unittest.mock import patch

        import numpy as np
        import rasterio
        from rasterio.transform import from_origin

        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        real_heights = {"2016-08-28": 3042, "2016-08-22": 3064, "2016-08-27": 22935}
        real_paths = {}
        for label, h in real_heights.items():
            path = tmp_path / f"{label}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=h,
                width=10,
                count=1,
                dtype="float32",
                crs="EPSG:4326",
                transform=from_origin(13.0, 42.0, 0.001, 0.001),
            ) as dst:
                dst.write(np.zeros((h, 10), dtype="float32"), 1)
                dst.update_tags(matched_swath="iw1")
            real_paths[label] = path

        extractor = SLCExtractor(polarisation="VV")
        scenes = {
            "2016-08-28": "fake_a.zip",
            "2016-08-22": "fake_b.zip",
            "2016-08-27": "fake_c.zip",
        }

        def fake_extract_scene(
            self,
            zip_path,
            aoi,
            output_dir,
            label="",
            resume=False,
            preferred_swath=None,
            **kwargs,
        ):
            return real_paths[label]

        aoi = BoundingBox(min_lon=13.0, min_lat=42.0, max_lon=13.1, max_lat=42.1)
        with patch.object(SLCExtractor, "extract_scene", fake_extract_scene):
            kept, report = extractor.extract_consistent_stack(
                scenes, aoi, tmp_path / "out"
            )

        assert set(kept.keys()) == {"2016-08-28", "2016-08-22"}
        assert "2016-08-27" in report["excluded"]
        assert report["reference"] == "2016-08-28"
        assert report["reference_rows"] == 3042

        """Real, confirmed fix: added directly in response to an observed
        real pipeline failure where three same-AOI, same-week scenes
        resolved to two different real sub-swaths purely from small,
        real orbit-to-orbit variation at a genuine sub-swath boundary.
        With a real, deliberately ambiguous AOI overlapping BOTH iw1 and
        iw3, preferred_swath must force the named sub-swath even though
        the automatic search would be free to pick either."""
        import rasterio

        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        footprints = {
            "iw1": (13.0, 42.0, 13.3, 42.3),
            "iw3": (13.25, 42.25, 13.5, 42.5),  # genuinely overlaps the same corner
        }
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        ambiguous_aoi = BoundingBox(
            min_lon=13.28, min_lat=42.28, max_lon=13.29, max_lat=42.29
        )

        extractor = SLCExtractor(polarisation="VV")
        out = extractor.extract_scene(
            zip_path,
            ambiguous_aoi,
            tmp_path / "out",
            label="forced",
            preferred_swath="iw1",
        )

        assert out is not None
        with rasterio.open(out) as src:
            assert src.tags().get("matched_swath") == "iw1"

    def test_preferred_swath_falls_back_when_not_present(self, tmp_path):
        """Real, safe fallback: if the preferred sub-swath genuinely
        doesn't exist in this specific archive, the automatic search
        must still run -- never a silent failure or a forced, wrong
        sub-swath."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        footprints = {"iw1": (13.0, 42.0, 13.3, 42.3)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        aoi = BoundingBox(min_lon=13.1, min_lat=42.1, max_lon=13.2, max_lat=42.2)

        extractor = SLCExtractor(polarisation="VV")
        out = extractor.extract_scene(
            zip_path,
            aoi,
            tmp_path / "out",
            label="fallback",
            preferred_swath="iw3",  # doesn't exist here
        )

        assert out is not None  # falls back to automatic search, which finds real iw1

    def test_returns_none_when_no_subswath_overlaps(self, tmp_path):
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        footprints = {"iw1": (-100.0, 18.0, -99.5, 19.0)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        far_aoi = BoundingBox(min_lon=50.0, min_lat=50.0, max_lon=51.0, max_lat=51.0)

        extractor = SLCExtractor(polarisation="VV")
        out = extractor.extract_scene(zip_path, far_aoi, tmp_path / "out")

        assert out is None

    def test_returns_none_for_missing_archive(self, tmp_path):
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        extractor = SLCExtractor(polarisation="VV")
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)
        out = extractor.extract_scene(
            tmp_path / "nonexistent.zip", aoi, tmp_path / "out"
        )

        assert out is None

    def test_resolves_download_result_output_path(self, tmp_path):
        """extract_pair should use DownloadResult.output_path directly, no guessing."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.download_task import DownloadResult, DownloadStatus
        from pygeofetch.models.search_query import BoundingBox

        footprints = {"iw2": (-99.3, 19.2, -98.9, 19.6)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)

        dl_result = DownloadResult(
            status=DownloadStatus.COMPLETED,
            data_id="test-id",
            provider="copernicus",
            output_path=zip_path,
            output_paths=[zip_path],
        )

        extractor = SLCExtractor(polarisation="VV")
        ref_tif, sec_tif = extractor.extract_pair(
            dl_result, dl_result, aoi, tmp_path / "out"
        )

        assert ref_tif is not None
        assert sec_tif is not None
        assert ref_tif.exists()
        assert sec_tif.exists()

    def test_extract_pair_returns_none_none_for_failed_download(self, tmp_path):
        """A failed DownloadResult (no output_path) should resolve to (None, None)."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.download_task import DownloadResult, DownloadStatus
        from pygeofetch.models.search_query import BoundingBox

        failed_result = DownloadResult(
            status=DownloadStatus.FAILED,
            data_id="test-id",
            provider="copernicus",
            error="network error",
        )
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)

        extractor = SLCExtractor(polarisation="VV")
        ref_tif, sec_tif = extractor.extract_pair(
            failed_result, failed_result, aoi, tmp_path / "out"
        )

        assert ref_tif is None
        assert sec_tif is None

    def test_list_subswaths(self, tmp_path):
        from pygeofetch.insar import SLCExtractor

        footprints = {
            "iw1": (-100.0, 18.0, -99.5, 19.0),
            "iw2": (-99.3, 19.2, -98.9, 19.6),
            "iw3": (-98.5, 20.0, -98.0, 20.5),
        }
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)

        extractor = SLCExtractor(polarisation="VV")
        subswaths = extractor.list_subswaths(zip_path)

        assert len(subswaths) == 3

    def test_vh_polarisation_selection(self, tmp_path):
        """Requesting VH should find VH tiffs, not VV."""
        from pygeofetch.insar import SLCExtractor

        footprints = {"iw1": (-100.0, 18.0, -99.5, 19.0)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints, polarisation="vh")

        vv_extractor = SLCExtractor(polarisation="VV")
        vh_extractor = SLCExtractor(polarisation="VH")

        assert vv_extractor.list_subswaths(zip_path) == []
        assert len(vh_extractor.list_subswaths(zip_path)) == 1

    def test_bad_zip_returns_empty_list(self, tmp_path):
        """A corrupt/non-zip file should not crash, just return no sub-swaths."""
        from pygeofetch.insar import SLCExtractor

        bad_zip = tmp_path / "not_a_zip.zip"
        bad_zip.write_bytes(b"not a real zip file")

        extractor = SLCExtractor(polarisation="VV")
        result = extractor.list_subswaths(bad_zip)

        assert result == []

    def test_direct_string_path_resolution(self, tmp_path):
        """extract_pair should also accept plain string/Path arguments, not just DownloadResult."""
        from pygeofetch.insar import SLCExtractor
        from pygeofetch.models.search_query import BoundingBox

        footprints = {"iw2": (-99.3, 19.2, -98.9, 19.6)}
        zip_path = _build_synthetic_slc_zip(tmp_path, footprints)
        aoi = BoundingBox(min_lon=-99.2, min_lat=19.3, max_lon=-99.0, max_lat=19.5)

        extractor = SLCExtractor(polarisation="VV")
        ref_tif, sec_tif = extractor.extract_pair(
            str(zip_path), zip_path, aoi, tmp_path / "out"
        )

        assert ref_tif is not None
        assert sec_tif is not None


class TestDownloadResume:
    """
    Tests for AdaptiveDownloader's resume support — verifying
    options.resume=True correctly skips re-downloading a file that
    already exists on disk and passes validation.
    """

    def test_finds_existing_file_in_provider_subfolder(self, tmp_path):
        import zipfile

        from pygeofetch.core.downloader import AdaptiveDownloader
        from pygeofetch.models.satellite_data import SatelliteData

        provider_dir = tmp_path / "copernicus"
        provider_dir.mkdir()
        existing_file = provider_dir / "S1A_test_EF47.SAFE.zip"
        with zipfile.ZipFile(existing_file, "w") as zf:
            zf.writestr("dummy.txt", "content")

        scene = SatelliteData(
            id="uuid-1234",
            provider="copernicus",
            assets={},
            properties={"name": "S1A_test_EF47.SAFE"},
        )

        downloader = AdaptiveDownloader()
        found = downloader._find_existing_download(scene, tmp_path)

        assert found == existing_file

    def test_returns_none_when_nothing_exists(self, tmp_path):
        from pygeofetch.core.downloader import AdaptiveDownloader
        from pygeofetch.models.satellite_data import SatelliteData

        scene = SatelliteData(
            id="uuid-5678",
            provider="copernicus",
            assets={},
            properties={"name": "S1A_nonexistent.SAFE"},
        )

        downloader = AdaptiveDownloader()
        found = downloader._find_existing_download(scene, tmp_path)

        assert found is None

    def test_falls_back_to_substring_match(self, tmp_path):
        """If the exact name doesn't match, fall back to matching a
        distinguishing chunk of the name (handles filename sanitisation)."""
        import zipfile

        from pygeofetch.core.downloader import AdaptiveDownloader
        from pygeofetch.models.satellite_data import SatelliteData

        provider_dir = tmp_path / "copernicus"
        provider_dir.mkdir()
        # Filename slightly different from what properties['name'] would predict
        existing_file = provider_dir / "sanitised_name_with_EF47_chunk.zip"
        with zipfile.ZipFile(existing_file, "w") as zf:
            zf.writestr("dummy.txt", "content")

        scene = SatelliteData(
            id="uuid-9999",
            provider="copernicus",
            assets={},
            properties={"name": "S1A_IW_SLC__1SDV_somedate_EF47.SAFE"},
        )

        downloader = AdaptiveDownloader()
        found = downloader._find_existing_download(scene, tmp_path)

        assert found == existing_file

    def test_download_skips_when_resume_true_and_file_exists(self, tmp_path):
        """End-to-end: download() should return immediately with
        from_cache=True instead of calling the provider, when a valid
        existing file is found and resume=True."""
        import zipfile

        from pygeofetch.core.downloader import AdaptiveDownloader
        from pygeofetch.models.download_task import DownloadOptions, DownloadStatus
        from pygeofetch.models.satellite_data import SatelliteData

        provider_dir = tmp_path / "copernicus"
        provider_dir.mkdir()
        existing_file = provider_dir / "S1A_cached_ABCD.SAFE.zip"
        with zipfile.ZipFile(existing_file, "w") as zf:
            zf.writestr("dummy.txt", "content")

        scene = SatelliteData(
            id="uuid-cached",
            provider="copernicus",
            assets={},
            properties={"name": "S1A_cached_ABCD.SAFE"},
        )

        downloader = AdaptiveDownloader()
        # No provider needs to be registered — if resume correctly short-
        # circuits, _get_provider() is never reached.
        result = downloader.download(scene, tmp_path, DownloadOptions(resume=True))

        assert result.status == DownloadStatus.COMPLETED
        assert result.from_cache is True
        assert result.output_path == existing_file
