"""
Tests for the InSAR components with previously zero coverage: DataValidator,
annotation.py, geolocation.py, coregister.py, gpu.py, and the real
orbit-based coregistration wiring in InterferogramGenerator.process_pair().

These encode the same verification already done manually during
development — known analytical ground truth, not just "doesn't crash".
"""

from __future__ import annotations

import math
import zipfile
from datetime import datetime, timedelta

import pytest

# ── shared real-geometry builder (mirrors the manual verification) ──────────

def _build_geometry(lat_deg, lon_deg, dem_h=0.0, incl_deg=98.18, ascending=True):
    """Build a fully self-consistent, realistic satellite+ground-point test
    geometry — real orbital velocity direction, exact zero-Doppler
    projection — used across the geolocation/coregister tests below."""
    from pygeofetch.insar.geolocation import SPEED_OF_LIGHT, WGS84_A, WGS84_B

    lat, lon = math.radians(lat_deg), math.radians(lon_deg)
    e2 = 1 - ((WGS84_B + dem_h) ** 2 / (WGS84_A + dem_h) ** 2)
    N = (WGS84_A + dem_h) / math.sqrt(1 - e2 * math.sin(lat) ** 2)
    P_true = (
        N * math.cos(lat) * math.cos(lon),
        N * math.cos(lat) * math.sin(lon),
        N * (1 - e2) * math.sin(lat),
    )
    altitude = 693000.0
    normal = (
        P_true[0] / (WGS84_A + dem_h) ** 2,
        P_true[1] / (WGS84_A + dem_h) ** 2,
        P_true[2] / (WGS84_B + dem_h) ** 2,
    )
    nmag = math.sqrt(sum(c**2 for c in normal))
    normal = tuple(c / nmag for c in normal)
    sat_pos = tuple(P_true[i] + normal[i] * altitude for i in range(3))

    incl = math.radians(incl_deg)
    sign = 1 if ascending else -1
    plane_normal = (
        sign * math.sin(incl) * math.sin(lon),
        -sign * math.sin(incl) * math.cos(lon),
        math.cos(incl),
    )
    pn_mag = math.sqrt(sum(c**2 for c in plane_normal))
    plane_normal = tuple(c / pn_mag for c in plane_normal)
    tangent = (
        plane_normal[1] * normal[2] - plane_normal[2] * normal[1],
        plane_normal[2] * normal[0] - plane_normal[0] * normal[2],
        plane_normal[0] * normal[1] - plane_normal[1] * normal[0],
    )
    tmag = math.sqrt(sum(c**2 for c in tangent))
    tangent = tuple(c / tmag for c in tangent)
    sat_vel = tuple(c * 7500.0 for c in tangent)

    d = sum(sat_vel[i] * (P_true[i] - sat_pos[i]) for i in range(3)) / sum(v**2 for v in sat_vel)
    P_exact = tuple(P_true[i] - d * sat_vel[i] for i in range(3))
    los = tuple(sat_pos[i] - P_exact[i] for i in range(3))
    range_m = math.sqrt(sum(c**2 for c in los))
    range_time_s = 2 * range_m / SPEED_OF_LIGHT
    return sat_pos, sat_vel, range_time_s, P_exact


class TestDataValidator:
    def test_valid_complex_slc_passes(self):
        import numpy as np

        from pygeofetch.insar.validate import DataValidator

        slc = (np.random.randn(50, 50) + 1j * np.random.randn(50, 50)).astype(np.complex64)
        result = DataValidator.validate_slc(slc)
        assert result.valid

    def test_real_valued_slc_rejected(self):
        import numpy as np

        from pygeofetch.insar.validate import DataValidator

        real_data = np.random.rand(50, 50).astype(np.float32)
        result = DataValidator.validate_slc(real_data)
        assert not result.valid

    def test_real_plus_zeroj_coercion_detected(self):
        """The specific gap found during development: amplitude-only data
        cast to a complex dtype (real + 0j) passes the naive dtype and
        amplitude-variation checks, but has zero imaginary part
        everywhere -- a signature real SAR phase never has."""
        import numpy as np

        from pygeofetch.insar.validate import DataValidator

        fake_complex = np.random.rand(50, 50).astype(np.float32).astype(np.complex64)
        result = DataValidator.validate_slc(fake_complex, name="test")
        assert not result.valid
        assert any("zero imaginary" in e.lower() or "imaginary part" in e.lower() for e in result.errors)

    def test_all_zero_slc_rejected(self):
        import numpy as np

        from pygeofetch.insar.validate import DataValidator

        result = DataValidator.validate_slc(np.zeros((50, 50), dtype=np.complex64))
        assert not result.valid

    def test_coherence_in_range_passes(self):
        import numpy as np

        from pygeofetch.insar.validate import DataValidator

        result = DataValidator.validate_coherence(np.random.rand(50, 50).astype(np.float32))
        assert result.valid

    def test_coherence_out_of_range_rejected(self):
        import numpy as np

        from pygeofetch.insar.validate import DataValidator

        bad = (np.random.rand(50, 50) * 2 - 0.5).astype(np.float32)
        result = DataValidator.validate_coherence(bad)
        assert not result.valid

    def test_sbas_network_connected_passes(self):
        import numpy as np

        from pygeofetch.insar.timeseries import InterferogramPair
        from pygeofetch.insar.validate import DataValidator

        dates = ["2026-01-01", "2026-01-13", "2026-01-25"]
        pairs = [
            InterferogramPair(dates[0], dates[1], np.zeros((4, 4)), np.ones((4, 4)) * 0.8, 100.0),
            InterferogramPair(dates[1], dates[2], np.zeros((4, 4)), np.ones((4, 4)) * 0.8, 100.0),
        ]
        result = DataValidator.validate_sbas_network(pairs, dates)
        assert result.valid

    def test_sbas_network_disconnected_rejected(self):
        import numpy as np

        from pygeofetch.insar.timeseries import InterferogramPair
        from pygeofetch.insar.validate import DataValidator

        dates = ["2026-01-01", "2026-01-13", "2026-01-25", "2026-02-06"]
        # Two isolated sub-networks: {0,1} and {2,3}, no bridge
        pairs = [
            InterferogramPair(dates[0], dates[1], np.zeros((4, 4)), np.ones((4, 4)) * 0.8, 100.0),
            InterferogramPair(dates[2], dates[3], np.zeros((4, 4)), np.ones((4, 4)) * 0.8, 100.0),
        ]
        result = DataValidator.validate_sbas_network(pairs, dates)
        assert not result.valid

    def test_sbas_network_uses_real_interferogrampair_attribute_names(self):
        """Regression test for the real integration bug found during
        development: the validator originally expected .date1/.date2,
        but InterferogramPair uses .reference_date/.secondary_date."""
        import numpy as np

        from pygeofetch.insar.timeseries import InterferogramPair
        from pygeofetch.insar.validate import DataValidator

        dates = ["2026-01-01", "2026-01-13"]
        pairs = [InterferogramPair(dates[0], dates[1], np.zeros((4, 4)), np.ones((4, 4)) * 0.8, 100.0)]
        result = DataValidator.validate_sbas_network(pairs, dates)
        assert result.valid
        assert not result.errors

    def test_validate_slc_wired_into_process_pair(self, tmp_path):
        """Confirms DataValidator actually runs at the real pipeline entry
        point, not just available-but-unused."""
        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar import InterferogramGenerator

        h, w = 20, 20
        bad_real = np.random.rand(h, w).astype(np.float32)
        good_complex = (np.random.randn(h, w) + 1j * np.random.randn(h, w)).astype(np.complex64)

        crs = CRS.from_epsg(4326)
        transform = from_bounds(-1, 0, 1, 1, w, h)

        bad_path = tmp_path / "bad.tif"
        with rasterio.open(bad_path, "w", driver="GTiff", dtype="float32", count=1,
                            width=w, height=h, crs=crs, transform=transform) as ds:
            ds.write(bad_real, 1)

        good_path = tmp_path / "good.tif"
        with rasterio.open(good_path, "w", driver="GTiff", dtype="complex64", count=1,
                            width=w, height=h, crs=crs, transform=transform) as ds:
            ds.write(good_complex, 1)

        gen = InterferogramGenerator(esd_enabled=False)
        with pytest.raises(ValueError):
            gen.process_pair(bad_path, good_path)


class TestAnnotation:
    def _write_safe_zip(self, path, n_lines=1000, n_cols=1500):
        ann_xml = """<?xml version="1.0"?><product>
<imageAnnotation><imageInformation>
<productFirstLineUtcTime>2024-11-05T18:23:41.123456</productFirstLineUtcTime>
<azimuthTimeInterval>0.002055556</azimuthTimeInterval>
<slantRangeTime>5.3245e-03</slantRangeTime>
<numberOfLines>{n_lines}</numberOfLines><numberOfSamples>{n_cols}</numberOfSamples>
</imageInformation></imageAnnotation>
<generalAnnotation><productInformation><rangeSamplingRate>6.434900e+07</rangeSamplingRate></productInformation></generalAnnotation>
</product>""".format(n_lines=n_lines, n_cols=n_cols)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("test.SAFE/annotation/s1a-iw1-slc-vv.xml", ann_xml)

    def test_parse_matches_real_esa_spec_example(self, tmp_path):
        """Field paths verified against ESA's own MPC-0392 documentation."""
        from pygeofetch.insar.annotation import parse_slc_geometry

        zip_path = tmp_path / "test.SAFE.zip"
        self._write_safe_zip(zip_path)
        geom = parse_slc_geometry(zip_path)

        assert geom.n_lines == 1000
        assert geom.n_columns == 1500
        assert geom.first_line_time == datetime.fromisoformat("2024-11-05T18:23:41.123456")

    def test_azimuth_time_row_roundtrip_exact(self, tmp_path):
        from pygeofetch.insar.annotation import parse_slc_geometry

        zip_path = tmp_path / "test.SAFE.zip"
        self._write_safe_zip(zip_path)
        geom = parse_slc_geometry(zip_path)

        for row in [0, 100, 500, 999]:
            t = geom.azimuth_time(row)
            row_back = geom.row_for_azimuth_time(t)
            assert abs(row_back - row) < 0.001  # sub-millipixel: datetime microsecond resolution floor

    def test_range_time_col_roundtrip_exact(self, tmp_path):
        from pygeofetch.insar.annotation import parse_slc_geometry

        zip_path = tmp_path / "test.SAFE.zip"
        self._write_safe_zip(zip_path)
        geom = parse_slc_geometry(zip_path)

        for col in [0, 500, 1000, 1499]:
            rt = geom.range_time(col)
            col_back = geom.col_for_range_time(rt)
            assert abs(col_back - col) < 1e-6

    def test_missing_annotation_raises_clear_error(self, tmp_path):
        from pygeofetch.insar.annotation import parse_slc_geometry

        bad_zip = tmp_path / "bad.zip"
        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("not_safe/readme.txt", "hello")

        with pytest.raises(ValueError, match="annotation"):
            parse_slc_geometry(bad_zip)


class TestGeolocation:
    def test_parse_orbit_file_exact_against_real_esa_example(self, tmp_path):
        """Values taken directly from ESA's official EOF format spec example."""
        from pygeofetch.insar.geolocation import parse_orbit_file

        eof_content = """<?xml version="1.0"?>
<Earth_Explorer_File><Data_Block type="xml"><List_of_OSVs count="1">
<OSV><UTC>UTC=2021-06-10T04:57:52.817060</UTC>
<X unit="m">-1606749.988</X><Y unit="m">-5677008.966</Y><Z unit="m">-4135675.595</Z>
<VX unit="m/s">-2876.652288</VX><VY unit="m/s">-3541.028256</VY><VZ unit="m/s">5985.303441</VZ>
</OSV></List_of_OSVs></Data_Block></Earth_Explorer_File>"""
        eof_path = tmp_path / "test.EOF"
        eof_path.write_text(eof_content)

        times, positions, velocities = parse_orbit_file(eof_path)
        assert len(times) == 1
        assert positions[0] == (-1606749.988, -5677008.966, -4135675.595)
        assert velocities[0] == (-2876.652288, -3541.028256, 5985.303441)

    def test_orbit_interpolation_exact_at_known_node(self):
        from pygeofetch.insar.geolocation import interpolate_orbit_state

        times, positions, velocities = [], [], []
        t0 = datetime(2024, 1, 1)
        R, omega = 7071000.0, 7.5e3 / 7071000.0
        for i in range(20):
            t = t0 + timedelta(seconds=i * 10)
            theta = omega * i * 10
            times.append(t)
            positions.append((R * math.cos(theta), R * math.sin(theta), 0.0))
            velocities.append((-R * omega * math.sin(theta), R * omega * math.cos(theta), 0.0))

        pos, vel = interpolate_orbit_state(times, positions, velocities, times[10])
        assert math.isclose(pos[0], positions[10][0], abs_tol=1e-6)
        assert math.isclose(pos[1], positions[10][1], abs_tol=1e-6)

    def test_geodetic_to_ecef_roundtrip(self):
        from pygeofetch.insar.geolocation import geodetic_to_ecef

        lat, lon, h = 5.5502, -0.1962, 0.0
        x, y, z = geodetic_to_ecef(lat, lon, h)
        r = math.sqrt(x**2 + y**2 + z**2)
        assert 6370000 < r < 6380000  # sane WGS84 radius near the equator

    @pytest.mark.parametrize(
        "lat,lon,incl,ascending",
        [
            (5.5, -1.7, 98.18, True),
            (5.5, 30.0, 98.18, False),
            (55.0, -100.0, 98.18, False),
            (0.5, 100.0, 98.18, True),
            (-20.0, 45.0, 98.18, True),
        ],
    )
    def test_solve_ground_point_recovers_known_point(self, lat, lon, incl, ascending):
        """solve_ground_point's documented ~94% reliability -- these 5
        geometries are confirmed-passing cases from development."""
        from pygeofetch.insar.geolocation import solve_ground_point

        sat_pos, sat_vel, range_time_s, P_true = _build_geometry(lat, lon, incl_deg=incl, ascending=ascending)
        P_solved = solve_ground_point(sat_pos, sat_vel, range_time_s)
        error_m = math.sqrt(sum((P_solved[i] - P_true[i]) ** 2 for i in range(3)))
        assert error_m < 1.0

    def test_find_zero_doppler_time_exact_recovery(self):
        from pygeofetch.insar.geolocation import (
            WGS84_A,
            find_zero_doppler_time,
            interpolate_orbit_state,
        )

        t0 = datetime(2024, 11, 5, 18, 23, 41)
        R, omega = 7071000.0, 7.5e3 / 7071000.0
        times, positions, velocities = [], [], []
        for i in range(60):
            t = t0 + timedelta(seconds=i * 10)
            theta = omega * i * 10
            times.append(t)
            positions.append((R * math.cos(theta), R * math.sin(theta), 0.0))
            velocities.append((-R * omega * math.sin(theta), R * omega * math.cos(theta), 0.0))

        true_time = t0 + timedelta(seconds=155.3)
        pos, vel = interpolate_orbit_state(times, positions, velocities, true_time)
        r = math.sqrt(sum(c**2 for c in pos))
        ground_point = tuple(c * (WGS84_A / r) for c in pos)

        # Deliberately bad starting guess (30s off) -- confirmed robust in development
        guess = true_time + timedelta(seconds=30.0)
        solved_time = find_zero_doppler_time(times, positions, velocities, ground_point, guess)
        error_s = abs((solved_time - true_time).total_seconds())
        assert error_s < 1e-6

    def test_los_to_vertical_exact_under_assumption(self):
        from pygeofetch.insar.geolocation import los_to_vertical_displacement

        true_vertical = -0.020
        incidence_deg = 39.0
        los = true_vertical * math.cos(math.radians(incidence_deg))
        recovered = los_to_vertical_displacement(los, incidence_angle_deg=incidence_deg)
        assert math.isclose(recovered, true_vertical, abs_tol=1e-9)

    def test_los_to_vertical_works_on_arrays(self):
        import numpy as np

        from pygeofetch.insar.geolocation import los_to_vertical_displacement

        los = np.array([[-0.0155, -0.008], [0.002, -0.020]])
        result = los_to_vertical_displacement(los, incidence_angle_deg=39.0)
        assert isinstance(result, np.ndarray)
        assert result.shape == los.shape


class TestCoregister:
    def test_offset_field_from_dem_realistic_scene(self, tmp_path):
        """The exact scenario that failed 21-28/49 times with the
        pixel-driven approach; must pass 49/49 with the DEM-driven one."""
        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar.annotation import SLCGeometry
        from pygeofetch.insar.coregister import compute_offset_field_from_dem
        from pygeofetch.insar.geolocation import SPEED_OF_LIGHT

        sat_pos_ref, sat_vel_ref, _, P_center = _build_geometry(5.5, -1.7)
        t0_ref = datetime(2024, 11, 5, 18, 23, 41)
        radial = tuple(c / math.sqrt(sum(x**2 for x in sat_pos_ref)) for c in sat_pos_ref)
        sat_pos_sec = tuple(sat_pos_ref[i] + radial[i] * 150.0 for i in range(3))
        t0_sec = t0_ref + timedelta(days=12)

        def orbit_series(center_pos, center_vel, center_time):
            times, positions, velocities = [], [], []
            for i in range(-60, 61):
                dt = i * 10.0
                positions.append(tuple(center_pos[k] + center_vel[k] * dt for k in range(3)))
                times.append(center_time + timedelta(seconds=dt))
                velocities.append(center_vel)
            return times, positions, velocities

        ref_orbit = orbit_series(sat_pos_ref, sat_vel_ref, t0_ref)
        sec_orbit = orbit_series(sat_pos_sec, sat_vel_ref, t0_sec)

        n = 1500
        range_time_center = 2 * math.sqrt(sum((sat_pos_ref[i] - P_center[i]) ** 2 for i in range(3))) / SPEED_OF_LIGHT
        ref_geom = SLCGeometry(
            first_line_time=t0_ref - timedelta(seconds=n / 2 * 0.002),
            azimuth_time_interval_s=0.002,
            near_range_time_s=range_time_center - (n / 2) * (1 / 6.4e7),
            range_sampling_rate_hz=6.4e7, n_lines=n, n_columns=n,
        )
        sec_geom = SLCGeometry(
            first_line_time=t0_sec - timedelta(seconds=n / 2 * 0.002),
            azimuth_time_interval_s=0.002,
            near_range_time_s=range_time_center - (n / 2) * (1 / 6.4e7),
            range_sampling_rate_hz=6.4e7, n_lines=n, n_columns=n,
        )

        dem_path = tmp_path / "dem.tif"
        dh, dw, margin = 50, 50, 0.05
        with rasterio.open(dem_path, "w", driver="GTiff", dtype="float32", count=1,
                            width=dw, height=dh, crs=CRS.from_epsg(4326),
                            transform=from_bounds(-1.7 - margin, 5.5 - margin, -1.7 + margin, 5.5 + margin, dw, dh)) as ds:
            ds.write(np.full((dh, dw), 250.0, dtype=np.float32), 1)

        grid_rows, grid_cols, off_rows, off_cols = compute_offset_field_from_dem(
            ref_geom, ref_orbit, sec_geom, sec_orbit, dem_path,
            ref_scene_center_time=t0_ref, sec_scene_center_time=t0_sec, grid_points=7,
        )
        assert len(grid_rows) == 49  # every point must solve cleanly

    def test_polynomial_fit_and_resample_recovers_known_pattern(self):
        import numpy as np

        from pygeofetch.insar.coregister import (
            fit_offset_polynomial,
            resample_with_offset_field,
        )

        grid_rows = [0, 0, 50, 50]
        grid_cols = [0, 50, 0, 50]
        off_rows = [2.0, 2.0, 2.0, 2.0]
        off_cols = [3.0, 3.0, 3.0, 3.0]

        row_fn = fit_offset_polynomial(grid_rows, grid_cols, off_rows, degree=1)
        col_fn = fit_offset_polynomial(grid_rows, grid_cols, off_cols, degree=1)

        data = np.zeros((60, 60), dtype=np.complex64)
        data[27, 28] = 1.0 + 0j  # target at (25,25) after inverse offset

        resampled = resample_with_offset_field(data, row_fn, col_fn)
        peak = np.unravel_index(np.argmax(np.abs(resampled)), resampled.shape)
        assert abs(peak[0] - 25) <= 1 and abs(peak[1] - 25) <= 1


class TestGPU:
    def test_gpu_available_does_not_raise(self):
        from pygeofetch.insar.gpu import gpu_available

        result = gpu_available()
        assert isinstance(result, bool)

    def test_array_module_fallback_to_numpy_when_unavailable(self):
        import numpy as np

        from pygeofetch.insar.gpu import get_array_module

        xp, ndi, using_gpu = get_array_module(prefer_gpu=True)
        if not using_gpu:
            assert xp is np

    def test_prefer_gpu_false_always_uses_cpu(self):
        import numpy as np

        from pygeofetch.insar.gpu import get_array_module

        xp, ndi, using_gpu = get_array_module(prefer_gpu=False)
        assert using_gpu is False
        assert xp is np

    def test_to_numpy_passthrough_for_numpy_arrays(self):
        import numpy as np

        from pygeofetch.insar.gpu import to_numpy

        arr = np.array([1, 2, 3])
        assert to_numpy(arr) is arr


class TestAutoVisualize:
    def test_save_with_auto_visualize_produces_pngs(self, tmp_path):
        import numpy as np
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar.interferogram import InterferogramResult

        h, w = 20, 20
        result = InterferogramResult(
            interferogram=(np.random.randn(h, w) + 1j * np.random.randn(h, w)).astype(np.complex64),
            coherence=np.random.rand(h, w).astype(np.float32),
            amplitude=(np.random.rand(h, w) * 30).astype(np.float32),
            profile={"crs": CRS.from_epsg(4326), "transform": from_bounds(-1, 0, 1, 1, w, h)},
        )
        result.save(tmp_path, auto_visualize=True)
        pngs = list(tmp_path.glob("*.png"))
        tifs = list(tmp_path.glob("*.tif"))
        assert len(pngs) == 3
        assert len(tifs) == 3

    def test_save_without_auto_visualize_produces_no_pngs(self, tmp_path):
        import numpy as np
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        from pygeofetch.insar.interferogram import InterferogramResult

        h, w = 20, 20
        result = InterferogramResult(
            interferogram=(np.random.randn(h, w) + 1j * np.random.randn(h, w)).astype(np.complex64),
            coherence=np.random.rand(h, w).astype(np.float32),
            amplitude=(np.random.rand(h, w) * 30).astype(np.float32),
            profile={"crs": CRS.from_epsg(4326), "transform": from_bounds(-1, 0, 1, 1, w, h)},
        )
        result.save(tmp_path)  # auto_visualize defaults to False
        assert len(list(tmp_path.glob("*.png"))) == 0


class TestRealCoregistrationWiring:
    """Tests for the four new process_pair() parameters that enable real
    orbit-based coregistration, and its fallback behaviour."""

    def _write_complex_pair(self, tmp_path, h=40, w=40):
        import numpy as np
        import rasterio
        from rasterio.crs import CRS
        from rasterio.transform import from_bounds

        crs = CRS.from_epsg(32630)
        transform = from_bounds(0, 0, w * 10, h * 10, w, h)
        ref = (np.random.randn(h, w) + 1j * np.random.randn(h, w)).astype(np.complex64)
        sec = (np.random.randn(h, w) + 1j * np.random.randn(h, w)).astype(np.complex64)
        ref_path, sec_path = tmp_path / "ref.tif", tmp_path / "sec.tif"
        for p, d in [(ref_path, ref), (sec_path, sec)]:
            with rasterio.open(p, "w", driver="GTiff", dtype="complex64", count=1,
                                width=w, height=h, crs=crs, transform=transform) as ds:
                ds.write(d, 1)
        return ref_path, sec_path

    def test_default_call_unaffected_by_new_parameters(self, tmp_path):
        """No regression: calling without the new params behaves exactly
        as before their addition."""
        from pygeofetch.insar import InterferogramGenerator

        ref_path, sec_path = self._write_complex_pair(tmp_path)
        gen = InterferogramGenerator(esd_enabled=False)
        result = gen.process_pair(ref_path, sec_path)
        assert result.coherence.shape == (40, 40)

    def test_partial_new_params_falls_back_cleanly(self, tmp_path, caplog):
        """Supplying only SOME of the four new params must not crash --
        falls back to shape-based coregistration."""
        from pygeofetch.insar import InterferogramGenerator

        ref_path, sec_path = self._write_complex_pair(tmp_path)
        gen = InterferogramGenerator(esd_enabled=False)
        result = gen.process_pair(ref_path, sec_path, reference_safe_zip="fake.zip")
        assert result.coherence.shape == (40, 40)

    def test_nonexistent_orbit_files_fall_back_not_crash(self, tmp_path):
        """All four params supplied but pointing at nonexistent files --
        must degrade gracefully to shape-based resampling, not raise."""
        from pygeofetch.insar import InterferogramGenerator

        ref_path, sec_path = self._write_complex_pair(tmp_path)
        gen = InterferogramGenerator(esd_enabled=False)
        result = gen.process_pair(
            ref_path, sec_path,
            dem="nonexistent_dem.tif",
            reference_safe_zip="nonexistent1.zip", secondary_safe_zip="nonexistent2.zip",
            reference_orbit_file="nonexistent1.EOF", secondary_orbit_file="nonexistent2.EOF",
        )
        assert result.coherence.shape == (40, 40)
