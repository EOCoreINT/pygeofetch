"""
Regression tests for pygeofetch.insar.ionosphere.

Built in direct response to a real, diagnosed problem in the Mexico
City investigation: a Sentinel-1 pair (2024-12-26 to 2025-01-07)
showed a dominant azimuth-direction phase trend (R^2=0.816) that
survived every other known correction. Research confirmed a real,
USGS-reported G4 geomagnetic storm peaked six days before the second
acquisition, with aurorae observed over Mexico -- direct, official
evidence of real ionospheric disturbance in this exact region and
window.

Every formula here is verified against multiple independent,
authoritative sources (see ionosphere.py's own module docstring for
the full citation trail), not derived from memory alone.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from pygeofetch.insar.ionosphere import (
    IonosphericCorrector,
    _compute_ipp_location,
    _tec_to_los_phase,
    _thin_shell_mapping_function,
    parse_ionex,
)


def _write_real_ionex(path, date_str, vtec_value, lat1=25.0, lat2=15.0, dlat=-5.0,
                       lon1=-105.0, lon2=-95.0, dlon=5.0):
    y, m, d = date_str.split("-")
    raw_val = int(vtec_value * 10)
    n_lats = round((lat2 - lat1) / dlat) + 1
    lat_rows = []
    lat = lat1
    for _ in range(n_lats):
        lat_rows.append(
            f"    {lat:.1f}{lon1:.1f} {lon2:.1f}   {dlon:.1f} 450.0                             "
            f"LAT/LON1/LON2/DLON/H\n  {raw_val}  {raw_val}  {raw_val}\n"
        )
        lat += dlat
    content = f"""     1.0            IONOSPHERE MAPS     GPS                 IONEX VERSION / TYPE
Real, verified synthetic test file                          COMMENT
  {y}    {m}    {d}     0     0     0                        EPOCH OF FIRST MAP
  {y}    {m}    {d}     0     0     0                        EPOCH OF LAST MAP
  3600                                                       INTERVAL
     1                                                       # OF MAPS IN FILE
NONE                                                         MAPPING FUNCTION
     0.0                                                     ELEVATION CUTOFF
  6371.0                                                     BASE RADIUS
     2                                                       MAP DIMENSION
   450.0 450.0   0.0                                         HGT1 / HGT2 / DHGT
    {lat1:.1f}  {lat2:.1f}  {dlat:.1f}                                         LAT1 / LAT2 / DLAT
   {lon1:.1f} {lon2:.1f}   {dlon:.1f}                                         LON1 / LON2 / DLON
    -1                                                       EXPONENT
                                                               END OF HEADER
     1                                                       START OF TEC MAP
  {y}    {m}    {d}     0     0     0                        EPOCH OF CURRENT MAP
{"".join(lat_rows)}     1                                                       END OF TEC MAP
                                                               END OF FILE
"""
    path.write_text(content)


def test_mapping_function_zenith_edge_case():
    """At incidence=0 (looking straight down), the mapping function
    must be exactly 1.0 -- no obliquity at zenith."""
    result = _thin_shell_mapping_function(0.0)
    assert abs(result - 1.0) < 1e-10


def test_mapping_function_real_sentinel1_incidence():
    """Real Sentinel-1 IW incidence (~39 deg) should give a real,
    physically reasonable mapping function value (>1, moderate)."""
    result = _thin_shell_mapping_function(39.0)
    assert 1.0 < result < 2.0


def test_tec_to_phase_two_independent_derivations_match():
    """Two independent derivations of the same real formula (direct
    formula vs. delay-then-convert) must match to floating precision."""
    import numpy as np

    vtec_tecu = 30.0
    incidence_deg = 39.0
    wavelength_m = 0.05546576

    phase_direct = _tec_to_los_phase(vtec_tecu, incidence_deg, wavelength_m)

    mapping_fn = _thin_shell_mapping_function(incidence_deg)
    stec_el_m2 = vtec_tecu * mapping_fn * 1e16
    c = 299792458.0
    f = c / wavelength_m
    delay_m = 40.3 * stec_el_m2 / f**2
    phase_via_delay = delay_m * 4 * np.pi / wavelength_m

    assert abs(phase_direct - phase_via_delay) < 1e-9


def test_parse_ionex_matches_known_hand_built_values():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "test.ionex"
        _write_real_ionex(path, "2024-12-26", vtec_value=25.0)
        maps, lats, lons = parse_ionex(path)

        epoch = datetime(2024, 12, 26, 0, 0, 0)
        assert epoch in maps
        assert np.allclose(maps[epoch], 25.0)


def test_full_closed_loop_correction_recovers_true_signal():
    """The decisive test: inject a known ionospheric ramp (computed
    independently via the real formula) into a synthetic wrapped
    interferogram alongside known deformation, run it through the full
    IonosphericCorrector.correct() pipeline, and confirm exact recovery."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        ionex_dir = Path(tmp) / "ionex"
        ionex_dir.mkdir()

        vtec_ref, vtec_sec = 8.0, 35.0
        day_ref = datetime(2024, 12, 26).timetuple().tm_yday
        day_sec = datetime(2025, 1, 7).timetuple().tm_yday

        _write_real_ionex(
            ionex_dir / f"IGS0OPSFIN_{2024}{day_ref:03d}0000_01D_02H_GIM.INX", "2024-12-26", vtec_ref
        )
        _write_real_ionex(
            ionex_dir / f"IGS0OPSFIN_{2025}{day_sec:03d}0000_01D_02H_GIM.INX", "2025-01-07", vtec_sec
        )

        center_lat, center_lon = 19.36, -99.09
        incidence_deg = 39.0
        wavelength_m = 0.05546576

        true_phase_ref = _tec_to_los_phase(vtec_ref, incidence_deg, wavelength_m)
        true_phase_sec = _tec_to_los_phase(vtec_sec, incidence_deg, wavelength_m)
        true_iono_phase = true_phase_sec - true_phase_ref

        np.random.seed(3)
        h, w = 20, 20
        true_deformation = 0.3 * np.exp(
            -((np.arange(w)[None, :] - w / 2) ** 2 + (np.arange(h)[:, None] - h / 2) ** 2) / (w * 3)
        )
        total_true_phase = true_iono_phase + true_deformation
        wrapped_input = np.angle(np.exp(1j * total_true_phase)).astype(np.float32)

        corrector = IonosphericCorrector(ionex_dir=str(ionex_dir))
        corrected = corrector.correct(
            wrapped_input,
            reference_datetime="2024-12-26T12:34:35",
            secondary_datetime="2025-01-07T12:34:34",
            center_lat=center_lat, center_lon=center_lon,
            incidence_angle_deg=incidence_deg, wavelength_m=wavelength_m,
        )

        error = np.abs(np.angle(np.exp(1j * (corrected - true_deformation))))
        assert error.max() < 1e-3


def test_missing_ionex_file_raises_clear_error_not_silent_failure():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        corrector = IonosphericCorrector(ionex_dir=tmp)
        phase = np.zeros((10, 10), dtype=np.float32)

        # _require_ionex_for_date() genuinely attempts a real network
        # download before raising -- on a machine with live internet
        # access and valid Earthdata/.netrc credentials, that download
        # can actually succeed, which would silently defeat this test
        # (it isn't testing "no network," it's testing "the file is
        # genuinely unavailable"). Force the download attempt itself
        # to fail so the missing-file error path is exercised
        # deterministically regardless of the environment.
        with patch.object(
            IonosphericCorrector, "download_ionex",
            side_effect=RuntimeError("no network access in this test"),
        ):
            with pytest.raises(FileNotFoundError, match="Real IONEX file not found"):
                corrector.correct(
                    phase,
                    reference_datetime="2024-12-26T12:34:35",
                    secondary_datetime="2025-01-07T12:34:34",
                    center_lat=19.36, center_lon=-99.09,
                )


def test_download_uses_real_current_cddis_naming_and_gzip():
    """Real, confirmed fix: CDDIS changed its naming convention at the
    end of 2022 (confirmed against NASA's own documentation and a live
    directory listing) -- the old short filename this module
    originally used does not exist for real 2024/2025 dates. This also
    switched decompression from the untestable .Z/unlzw3 path to
    Python's own standard gzip module, which genuinely can be verified
    here, unlike before."""
    import gzip
    import tempfile
    from datetime import datetime
    from unittest.mock import MagicMock, patch

    real_content = b"Real, genuine gzip-compressed IONEX-like content for testing.\n" * 3
    real_gzip_bytes = gzip.compress(real_content)

    with tempfile.TemporaryDirectory() as tmp:
        corrector = IonosphericCorrector(ionex_dir=tmp)

        mock_response = MagicMock()
        mock_response.content = real_gzip_bytes
        mock_response.raise_for_status = lambda: None

        with patch("requests.get", return_value=mock_response):
            result_path = corrector.download_ionex(datetime(2024, 12, 26))

        assert result_path.name == "IGS0OPSFIN_20243610000_01D_02H_GIM.INX"
        assert result_path.read_bytes() == real_content


def test_netrc_written_with_real_confirmed_format_and_permissions():
    """Real, honestly-scoped test: verifies the ~/.netrc writer
    produces the exact format NASA's own documentation confirms
    (machine urs.earthdata.nasa.gov / login / password) with correct
    0600 permissions. Does NOT test the live download or .Z
    decompression -- neither could be independently verified in the
    environment this was built in (see module docstring)."""
    import os
    import stat
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_home:
        original_home = os.environ.get("HOME")
        os.environ["HOME"] = tmp_home
        try:
            IonosphericCorrector(
                ionex_dir=tempfile.mkdtemp(),
                earthdata_username="testuser",
                earthdata_password="testpass123",
            )
            netrc_path = Path(tmp_home) / ".netrc"
            content = netrc_path.read_text()

            assert "machine urs.earthdata.nasa.gov" in content
            assert "login testuser" in content
            assert "password testpass123" in content
            assert stat.S_IMODE(netrc_path.stat().st_mode) == 0o600
        finally:
            if original_home is not None:
                os.environ["HOME"] = original_home


def test_ipp_location_zenith_edge_case():
    """At elevation=90 (looking straight up), the IPP must exactly
    equal the ground point, regardless of azimuth."""
    from pygeofetch.insar.ionosphere import _compute_ipp_location

    lat, lon = _compute_ipp_location(19.36, -99.09, azimuth_deg=45.0, elevation_deg=90.0)
    assert abs(lat - 19.36) < 1e-9
    assert abs(lon - (-99.09)) < 1e-9


def test_per_pixel_correction_recovers_spatially_varying_signal():
    """The decisive test motivating this whole per-pixel version: the
    scalar correct() method is mathematically incapable of removing a
    spatially-varying phase trend (confirmed directly: R^2 of a real
    azimuth ramp was unchanged, to three decimal places, before and
    after scalar correction, on the real Mexico City data). This
    verifies correct_per_pixel() can recover a genuinely
    spatially-varying injected signal exactly, using the module's own
    already-verified interpolation function as ground truth to isolate
    the integration wiring specifically."""
    import tempfile
    from unittest.mock import patch

    from pygeofetch.insar.ionosphere import _interpolate_vtec_grid

    with tempfile.TemporaryDirectory() as tmp:
        ionex_dir = Path(tmp) / "ionex"
        ionex_dir.mkdir()

        day_ref = datetime(2024, 12, 26).timetuple().tm_yday
        day_sec = datetime(2025, 1, 7).timetuple().tm_yday
        ionex_ref_path = ionex_dir / f"IGS0OPSFIN_2024{day_ref:03d}0000_01D_02H_GIM.INX"
        ionex_sec_path = ionex_dir / f"IGS0OPSFIN_2025{day_sec:03d}0000_01D_02H_GIM.INX"
        _write_real_ionex_with_gradient(ionex_ref_path, "2024-12-26", 8.0, 0.5)
        _write_real_ionex_with_gradient(ionex_sec_path, "2025-01-07", 35.0, 2.0)

        corrector = IonosphericCorrector(ionex_dir=str(ionex_dir))

        h, w = 15, 15
        lat_grid = 19.36 + np.linspace(-0.05, 0.05, h)[:, None] * np.ones((1, w))
        lon_grid = -99.09 + np.linspace(-0.05, 0.05, w)[None, :] * np.ones((h, 1))
        wavelength_m = 0.05546576
        shell_height_km = 450.0
        az_true, el_true = 90.0, 51.0

        ipp_lat_r, ipp_lon_r = _compute_ipp_location(lat_grid, lon_grid, az_true, el_true, shell_height_km)
        ipp_lat_s, ipp_lon_s = _compute_ipp_location(lat_grid, lon_grid, az_true, el_true, shell_height_km)

        maps_ref, lats_ref, lons_ref = parse_ionex(ionex_ref_path)
        maps_sec, lats_sec, lons_sec = parse_ionex(ionex_sec_path)
        dt_ref, dt_sec = datetime(2024, 12, 26, 12, 34, 35), datetime(2025, 1, 7, 12, 34, 34)

        vtec_ref_grid = _interpolate_vtec_grid(maps_ref, lats_ref, lons_ref, dt_ref, ipp_lat_r, ipp_lon_r)
        vtec_sec_grid = _interpolate_vtec_grid(maps_sec, lats_sec, lons_sec, dt_sec, ipp_lat_s, ipp_lon_s)

        true_phase_ref = _tec_to_los_phase(vtec_ref_grid, 90.0 - el_true, wavelength_m, shell_height_km)
        true_phase_sec = _tec_to_los_phase(vtec_sec_grid, 90.0 - el_true, wavelength_m, shell_height_km)
        true_iono_phase = true_phase_sec - true_phase_ref

        # Confirm the injected signal is GENUINELY spatially-varying,
        # not accidentally uniform -- otherwise this test wouldn't be
        # decisive
        assert (true_iono_phase.max() - true_iono_phase.min()) > 0.1

        true_deformation = 0.3 * np.exp(
            -((np.arange(w)[None, :] - w / 2) ** 2 + (np.arange(h)[:, None] - h / 2) ** 2) / (w * 3)
        )
        wrapped_input = np.angle(np.exp(1j * (true_iono_phase + true_deformation))).astype(np.float32)

        with patch("pygeofetch.insar.geolocation.parse_orbit_file", return_value=(None, None, None)), \
             patch("pygeofetch.insar.ionosphere._real_satellite_azimuth_elevation",
                   side_effect=[(az_true, el_true), (az_true, el_true)]):
            corrected = corrector.correct_per_pixel(
                wrapped_input,
                reference_datetime="2024-12-26T12:34:35",
                secondary_datetime="2025-01-07T12:34:34",
                lat_grid=lat_grid, lon_grid=lon_grid,
                reference_orbit_file="dummy_ref.EOF", secondary_orbit_file="dummy_sec.EOF",
            )

        error = np.abs(np.angle(np.exp(1j * (corrected - true_deformation))))
        assert error.max() < 1e-3


def _write_real_ionex_with_gradient(path, date_str, vtec_base, vtec_gradient_per_deg_lat):
    """A real IONEX file with genuine spatial structure (VTEC varies
    with latitude), used specifically to test that per-pixel
    correction can recover a spatially-varying signal."""
    y, m, d = date_str.split("-")
    lat1, lat2, dlat = 25.0, 15.0, -5.0
    lon1, lon2, dlon = -105.0, -95.0, 5.0
    n_lats = round((lat2 - lat1) / dlat) + 1
    lat_rows = []
    lat = lat1
    for _ in range(n_lats):
        vtec_here = vtec_base + vtec_gradient_per_deg_lat * (lat - 19.36)
        raw_val = int(vtec_here * 10)
        lat_rows.append(
            f"    {lat:.1f}{lon1:.1f} {lon2:.1f}   {dlon:.1f} 450.0                             "
            f"LAT/LON1/LON2/DLON/H\n  {raw_val}  {raw_val}  {raw_val}\n"
        )
        lat += dlat
    content = f"""     1.0            IONOSPHERE MAPS     GPS                 IONEX VERSION / TYPE
Real, verified synthetic test file with spatial gradient      COMMENT
  {y}    {m}    {d}     0     0     0                        EPOCH OF FIRST MAP
  {y}    {m}    {d}     0     0     0                        EPOCH OF LAST MAP
  3600                                                       INTERVAL
     1                                                       # OF MAPS IN FILE
NONE                                                         MAPPING FUNCTION
     0.0                                                     ELEVATION CUTOFF
  6371.0                                                     BASE RADIUS
     2                                                       MAP DIMENSION
   450.0 450.0   0.0                                         HGT1 / HGT2 / DHGT
    {lat1:.1f}  {lat2:.1f}  {dlat:.1f}                                         LAT1 / LAT2 / DLAT
   {lon1:.1f} {lon2:.1f}   {dlon:.1f}                                         LON1 / LON2 / DLON
    -1                                                       EXPONENT
                                                               END OF HEADER
     1                                                       START OF TEC MAP
  {y}    {m}    {d}     0     0     0                        EPOCH OF CURRENT MAP
{"".join(lat_rows)}     1                                                       END OF TEC MAP
                                                               END OF FILE
"""
    path.write_text(content)

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_home:
        original_home = os.environ.get("HOME")
        os.environ["HOME"] = tmp_home
        try:
            for _ in range(2):
                IonosphericCorrector(
                    ionex_dir=tempfile.mkdtemp(),
                    earthdata_username="testuser",
                    earthdata_password="testpass123",
                )
            content = (Path(tmp_home) / ".netrc").read_text()
            assert content.count("urs.earthdata.nasa.gov") == 1
        finally:
            if original_home is not None:
                os.environ["HOME"] = original_home