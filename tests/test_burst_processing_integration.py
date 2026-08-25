"""
Regression tests for the complete integration of real burst-aware
processing (annotation.parse_burst_info, deburst.deburst_array,
esd.estimate_esd_shift_per_burst_overlap) into
InterferogramGenerator.process_pair(), via the opt-in
use_real_burst_processing flag.

These are real, end-to-end tests through the actual process_pair()
method, not isolated unit tests of the individual pieces (those live in
test_burst_metadata.py, test_deburst.py, test_esd.py) -- this file
verifies the pieces are actually wired together correctly.
"""

import tempfile
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.crs import CRS
from rasterio.transform import from_bounds

from pygeofetch.insar import InterferogramGenerator
from pygeofetch.insar.esd import SENTINEL1_IW_DELTA_F_OVL_HZ


def _make_annotation_xml(
    first_line_time, n_bursts=3, lines_per_burst=100, samples_per_burst=150
):
    burst_interval_s = 90.0
    bursts_xml = []
    for i in range(n_bursts):
        base = datetime.fromisoformat(first_line_time)
        t = base + timedelta(seconds=i * burst_interval_s)
        first_valid = " ".join("0" for _ in range(lines_per_burst))
        last_valid = " ".join(
            str(samples_per_burst - 1) for _ in range(lines_per_burst)
        )
        bursts_xml.append(
            f"<burst><azimuthTime>{t.isoformat()}</azimuthTime>"
            f"<sensingTime>{t.isoformat()}</sensingTime><byteOffset>{i * 1000}</byteOffset>"
            f"<firstValidSample>{first_valid}</firstValidSample>"
            f"<lastValidSample>{last_valid}</lastValidSample></burst>"
        )
    total_lines = n_bursts * lines_per_burst
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<product><imageAnnotation><imageInformation>
<productFirstLineUtcTime>{first_line_time}</productFirstLineUtcTime>
<azimuthTimeInterval>1.0</azimuthTimeInterval>
<slantRangeTime>0.0053</slantRangeTime>
<numberOfLines>{total_lines}</numberOfLines>
<numberOfSamples>{samples_per_burst}</numberOfSamples>
</imageInformation></imageAnnotation>
<generalAnnotation><productInformation>
<rangeSamplingRate>64345238.12571428</rangeSamplingRate>
</productInformation></generalAnnotation>
<swathTiming><linesPerBurst>{lines_per_burst}</linesPerBurst>
<samplesPerBurst>{samples_per_burst}</samplesPerBurst>
<burstList count="{n_bursts}">{"".join(bursts_xml)}</burstList>
</swathTiming></product>"""


def _build_fixture(
    tmp_path,
    n_bursts=3,
    lines_per_burst=100,
    samples_per_burst=150,
    delta_f_ovl=SENTINEL1_IW_DELTA_F_OVL_HZ,
    true_shift_s=0.00001,
    seed=9,
):
    ref_zip = tmp_path / "ref.SAFE.zip"
    sec_zip = tmp_path / "sec.SAFE.zip"
    with zipfile.ZipFile(ref_zip, "w") as zf:
        zf.writestr(
            "ref.SAFE/annotation/s1a-iw2-slc-vv-ref.xml",
            _make_annotation_xml(
                "2024-11-08T18:18:20.000000",
                n_bursts,
                lines_per_burst,
                samples_per_burst,
            ),
        )
    with zipfile.ZipFile(sec_zip, "w") as zf:
        zf.writestr(
            "sec.SAFE/annotation/s1a-iw2-slc-vv-sec.xml",
            _make_annotation_xml(
                "2024-11-20T18:18:20.000000",
                n_bursts,
                lines_per_burst,
                samples_per_burst,
            ),
        )

    np.random.seed(seed)
    h, w = n_bursts * lines_per_burst, samples_per_burst
    scene = np.random.randn(h, w) + 1j * np.random.randn(h, w)
    scene /= np.abs(scene)
    ref_data = scene.copy()
    sec_data = scene.copy()

    f_bw, f_fw = -delta_f_ovl / 2, delta_f_ovl / 2
    overlap_row_len = 10
    for i in range(n_bursts - 1):
        bw_rows = slice(
            i * lines_per_burst + (lines_per_burst - overlap_row_len),
            i * lines_per_burst + lines_per_burst,
        )
        fw_rows = slice(
            (i + 1) * lines_per_burst, (i + 1) * lines_per_burst + overlap_row_len
        )
        sec_data[bw_rows] = scene[bw_rows] * np.exp(
            -1j * 2 * np.pi * f_bw * true_shift_s
        )
        sec_data[fw_rows] = scene[fw_rows] * np.exp(
            -1j * 2 * np.pi * f_fw * true_shift_s
        )
    sec_data *= np.exp(-1j * 2 * np.pi * f_bw * true_shift_s)

    crs = CRS.from_epsg(4326)
    transform = from_bounds(-99.15, 19.30, -99.05, 19.40, w, h)

    def write_complex(path, data):
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            dtype="float32",
            count=2,
            width=w,
            height=h,
            crs=crs,
            transform=transform,
        ) as ds:
            ds.write(data.real.astype(np.float32), 1)
            ds.write(data.imag.astype(np.float32), 2)

    ref_path, sec_path = tmp_path / "ref.tif", tmp_path / "sec.tif"
    write_complex(ref_path, ref_data)
    write_complex(sec_path, sec_data)

    return ref_path, sec_path, ref_zip, sec_zip, h, w, true_shift_s


def test_real_burst_processing_actually_runs_when_opted_in():
    with tempfile.TemporaryDirectory() as tmp:
        ref_path, sec_path, ref_zip, sec_zip, h, w, true_shift_s = _build_fixture(
            Path(tmp)
        )

        gen = InterferogramGenerator(
            esd_enabled=True, use_gpu=False, use_real_burst_processing=True
        )
        result = gen.process_pair(
            ref_path,
            sec_path,
            dem=None,
            reference_date="d1",
            secondary_date="d2",
            reference_safe_zip=ref_zip,
            secondary_safe_zip=sec_zip,
        )

        assert result.metadata["deburst_applied"] is True
        assert result.metadata["esd_method"] == "real_per_burst_esd_and_deburst"
        assert (
            result.interferogram.shape[0] < h
        ), "deburst must have removed real overlap rows"


def test_esd_shift_matches_known_deliberate_misregistration():
    with tempfile.TemporaryDirectory() as tmp:
        ref_path, sec_path, ref_zip, sec_zip, h, w, true_shift_s = _build_fixture(
            Path(tmp)
        )

        gen = InterferogramGenerator(
            esd_enabled=True, use_gpu=False, use_real_burst_processing=True
        )
        result = gen.process_pair(
            ref_path,
            sec_path,
            dem=None,
            reference_date="d1",
            secondary_date="d2",
            reference_safe_zip=ref_zip,
            secondary_safe_zip=sec_zip,
        )

        # true_shift_s was injected in TIME; the real, recovered value
        # here is in seconds too (esd_azimuth_shift_px is in pixels,
        # but with azimuth_time_interval_s=1.0 in this fixture, pixels
        # and seconds are numerically identical)
        assert abs(result.esd_azimuth_shift_px - true_shift_s) < 1e-6


def test_default_behaviour_unaffected_when_not_opted_in():
    """use_real_burst_processing defaults to False -- existing callers
    must see exactly the previous whole-image ESD behaviour, no deburst."""
    with tempfile.TemporaryDirectory() as tmp:
        ref_path, sec_path, ref_zip, sec_zip, h, w, true_shift_s = _build_fixture(
            Path(tmp)
        )

        gen = InterferogramGenerator(
            esd_enabled=True, use_gpu=False
        )  # real_burst_processing NOT set
        result = gen.process_pair(
            ref_path,
            sec_path,
            dem=None,
            reference_date="d1",
            secondary_date="d2",
            reference_safe_zip=ref_zip,
            secondary_safe_zip=sec_zip,
        )
        assert result.metadata["deburst_applied"] is False
        assert result.metadata["esd_method"] == "whole_image_esd"
        assert (
            result.interferogram.shape[0] == h
        ), "no rows should be removed without opting in"


def test_falls_back_gracefully_when_safe_zips_missing():
    """Opting in without supplying SAFE zips must not crash -- falls
    back to the existing whole-image ESD."""
    with tempfile.TemporaryDirectory() as tmp:
        ref_path, sec_path, ref_zip, sec_zip, h, w, true_shift_s = _build_fixture(
            Path(tmp)
        )

        gen = InterferogramGenerator(
            esd_enabled=True, use_gpu=False, use_real_burst_processing=True
        )
        result = gen.process_pair(
            ref_path,
            sec_path,
            dem=None,
            reference_date="d1",
            secondary_date="d2",
            # no reference_safe_zip / secondary_safe_zip supplied
        )
        assert result.metadata["deburst_applied"] is False
        assert result.interferogram.shape[0] == h
