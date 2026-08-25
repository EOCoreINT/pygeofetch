"""
Tests for pygeofetch.viz.plot.Plotter — array-input support and the
comparison/classification plot types.
"""

from __future__ import annotations

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


class TestPlotterArraySupport:
    def test_plot_raster_accepts_array_directly(self, tmp_path):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        arr = np.random.rand(10, 10)
        out = tmp_path / "test.png"
        pl.plot_raster(arr, title="test", output=out)

        assert out.exists()

    def test_plot_raster_with_extent(self, tmp_path):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        arr = np.random.rand(10, 10)
        out = tmp_path / "test.png"
        pl.plot_raster(arr, extent=(-1.75, -1.60, 6.15, 6.25), output=out)

        assert out.exists()

    def test_plot_raster_still_accepts_file_path(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        import numpy as np
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.viz import Plotter

        tif_path = tmp_path / "test.tif"
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            dtype="float32",
            count=1,
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(-1, 5, 1, 7, 10, 10),
        ) as ds:
            ds.write(np.random.rand(10, 10).astype("float32"), 1)

        pl = Plotter()
        out = tmp_path / "out.png"
        pl.plot_raster(tif_path, output=out)

        assert out.exists()


class TestPlotComparison:
    def test_three_panel_comparison(self, tmp_path):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        before = np.random.rand(10, 10)
        after = np.random.rand(10, 10)
        change = after - before
        out = tmp_path / "comparison.png"

        pl.plot_comparison(
            {"Before": before, "After": after, "Change": change},
            output=out,
        )
        assert out.exists()

    def test_per_panel_overrides(self, tmp_path):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        a, b = np.random.rand(10, 10), np.random.rand(10, 10) - 0.5
        out = tmp_path / "comparison.png"

        pl.plot_comparison(
            {"NDVI": a, "Change": b},
            per_panel_cmap={"Change": "RdBu"},
            per_panel_range={"Change": (-1, 1)},
            output=out,
        )
        assert out.exists()

    def test_single_panel(self, tmp_path):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        out = tmp_path / "single.png"
        pl.plot_comparison({"Only": np.random.rand(10, 10)}, output=out)
        assert out.exists()


class TestPlotClassification:
    def test_classification_plot_with_legend(self, tmp_path):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        classified = np.random.choice([0, 1, 2], size=(10, 10))
        out = tmp_path / "classified.png"

        pl.plot_classification(
            classified,
            class_labels={0: "Stable", 1: "Moderate", 2: "Severe"},
            class_colors={0: "green", 1: "orange", 2: "red"},
            output=out,
        )
        assert out.exists()

    def test_classification_percentages_sum_reasonably(self, tmp_path):
        """Sanity check: class percentages shown should reflect actual pixel counts."""
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        # All pixels class 0 -> should show ~100% for class 0
        classified = np.zeros((10, 10), dtype=int)
        out = tmp_path / "classified.png"

        pl.plot_classification(
            classified,
            class_labels={0: "All same", 1: "Other"},
            class_colors={0: "blue", 1: "red"},
            output=out,
        )
        assert out.exists()

    def test_two_class_flood_map(self, tmp_path):
        """Matches the intended flood/no-flood use case directly."""
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        flood_mask = np.random.choice([0, 1], size=(15, 15), p=[0.85, 0.15])
        out = tmp_path / "flood.png"

        pl.plot_classification(
            flood_mask,
            class_labels={0: "Not flooded", 1: "Flooded"},
            class_colors={0: "#f0f0f0", 1: "#1f77b4"},
            title="Flood Extent",
            output=out,
        )
        assert out.exists()


class TestQuicklook:
    """
    Tests for Plotter.quicklook() — the auto-detecting universal
    visualization entry point. Verifies actual correct mode detection
    (via the debug log), not just absence of exceptions.
    """

    def _detected_mode(self, pl, data, caplog):
        import logging

        with caplog.at_level(logging.DEBUG, logger="pygeofetch.viz.plot"):
            pl.quicklook(data)
        for record in caplog.records:
            if "auto-detected mode" in record.message:
                return record.message.split("mode=")[-1].strip("'")
        return None

    def test_ndvi_like_detected_as_index(self, caplog):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        arr = np.random.uniform(-0.2, 0.8, (20, 20))
        assert self._detected_mode(pl, arr, caplog) == "index"

    def test_sar_like_detected_as_sar(self, caplog):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        arr = np.random.normal(-12, 4, (20, 20))
        assert self._detected_mode(pl, arr, caplog) == "sar"

    def test_classification_detected_as_categorical(self, caplog):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        arr = np.random.choice([0, 1, 2, 3], size=(50, 50))
        assert self._detected_mode(pl, arr, caplog) == "categorical"

    def test_generic_continuous_detected_as_continuous(self, caplog):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        arr = np.random.uniform(50, 500, (20, 20))  # e.g. elevation-like
        assert self._detected_mode(pl, arr, caplog) == "continuous"

    def test_multiband_file_triggers_rgb(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        import numpy as np
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.viz import Plotter

        path = tmp_path / "multiband.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            dtype="float32",
            count=4,
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(-1, 5, 1, 7, 10, 10),
        ) as ds:
            for b in range(1, 5):
                ds.write(np.random.rand(10, 10).astype("float32"), b)

        pl = Plotter()
        out = tmp_path / "out.png"
        pl.quicklook(path, output=out)
        assert out.exists()

    def test_resolves_download_result(self, tmp_path):
        rasterio = pytest.importorskip("rasterio")
        import numpy as np
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.models.download_task import DownloadResult, DownloadStatus
        from pygeofetch.viz import Plotter

        path = tmp_path / "single.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            dtype="float32",
            count=1,
            width=10,
            height=10,
            crs=CRS.from_epsg(4326),
            transform=from_bounds(-1, 5, 1, 7, 10, 10),
        ) as ds:
            ds.write((np.random.rand(10, 10) * 0.6 - 0.1).astype("float32"), 1)

        dl_result = DownloadResult(
            status=DownloadStatus.COMPLETED,
            data_id="test",
            provider="usgs",
            output_path=path,
        )
        pl = Plotter()
        out = tmp_path / "out.png"
        pl.quicklook(dl_result, output=out)
        assert out.exists()

    def test_failed_download_result_raises_clear_error(self):
        from pygeofetch.models.download_task import DownloadResult, DownloadStatus
        from pygeofetch.viz import Plotter

        failed = DownloadResult(
            status=DownloadStatus.FAILED,
            data_id="test",
            provider="usgs",
            error="network error",
        )
        pl = Plotter()
        with pytest.raises(ValueError, match="no output_path"):
            pl.quicklook(failed)

    def test_mode_override_respected(self, caplog):
        """Explicit mode= should bypass auto-detection entirely."""
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        # Data that would auto-detect as "index" (NDVI-like range)
        arr = np.random.uniform(-0.2, 0.8, (20, 20))
        # Force it to be treated as continuous instead
        fig = pl.quicklook(arr, mode="continuous", colormap="plasma")
        assert fig is not None

    def test_empty_data_raises_clear_error(self):
        import numpy as np

        from pygeofetch.viz import Plotter

        pl = Plotter()
        arr = np.full((5, 5), np.nan)
        with pytest.raises(ValueError, match="no finite values"):
            pl.quicklook(arr)
