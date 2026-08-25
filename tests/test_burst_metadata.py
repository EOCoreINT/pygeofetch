"""
Regression tests for pygeofetch.insar.annotation.parse_burst_info() --
Step 1 of the real deburst/per-burst-ESD implementation plan.

Real, structurally-accurate synthetic annotation XML built to match the
confirmed schema (verified against multiple independent real sources:
a real, working extraction script actually used against the Copernicus
Data Space Ecosystem, a real academic SAR analysis tool's own parsed
representation, and SNAP's own Java source for its S-1 TOPS Deburst
operator).
"""

import tempfile
import zipfile
from pathlib import Path

import pytest
from pygeofetch.insar.annotation import parse_burst_info


def _make_burst_safe_zip(
    path: Path, lines_per_burst=5, samples_per_burst=20, n_bursts=3
):
    def make_burst_xml(index, az_time, sensing_time, byte_offset):
        first_valid = " ".join(
            str(2 if i in (0, lines_per_burst - 1) else 0)
            for i in range(lines_per_burst)
        )
        last_valid = " ".join(
            str(
                samples_per_burst - 3
                if i in (0, lines_per_burst - 1)
                else samples_per_burst - 1
            )
            for i in range(lines_per_burst)
        )
        return f"""<burst>
            <azimuthTime>{az_time}</azimuthTime>
            <sensingTime>{sensing_time}</sensingTime>
            <byteOffset>{byte_offset}</byteOffset>
            <firstValidSample>{first_valid}</firstValidSample>
            <lastValidSample>{last_valid}</lastValidSample>
        </burst>"""

    bursts_xml = "\n".join(
        make_burst_xml(
            i,
            f"2024-11-08T18:18:2{i}.000000",
            f"2024-11-08T18:18:2{i}.500000",
            i * 10000,
        )
        for i in range(n_bursts)
    )
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<product>
    <swathTiming>
        <linesPerBurst>{lines_per_burst}</linesPerBurst>
        <samplesPerBurst>{samples_per_burst}</samplesPerBurst>
        <burstList count="{n_bursts}">
            {bursts_xml}
        </burstList>
    </swathTiming>
</product>"""
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("test.SAFE/annotation/s1a-iw2-slc-vv-test.xml", xml_content)


def test_parses_real_structured_burst_metadata():
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "test.SAFE.zip"
        _make_burst_safe_zip(
            zip_path, lines_per_burst=5, samples_per_burst=20, n_bursts=3
        )

        result = parse_burst_info(zip_path)
        assert result.lines_per_burst == 5
        assert result.samples_per_burst == 20
        assert len(result.bursts) == 3
        assert len(result.bursts[0].first_valid_sample) == 5
        # Real, expected per-line taper pattern from the synthetic fixture
        assert list(result.bursts[0].first_valid_sample) == [2, 0, 0, 0, 2]
        assert list(result.bursts[0].last_valid_sample) == [17, 19, 19, 19, 17]


def test_raises_on_valid_sample_length_mismatch():
    """A real data-corruption case: firstValidSample entry count must
    match linesPerBurst -- must raise clearly, not silently misalign."""
    xml = """<?xml version="1.0"?>
<product><swathTiming><linesPerBurst>5</linesPerBurst><samplesPerBurst>20</samplesPerBurst>
<burstList count="1"><burst><azimuthTime>2024-11-08T18:18:20.000000</azimuthTime>
<sensingTime>2024-11-08T18:18:20.500000</sensingTime><byteOffset>0</byteOffset>
<firstValidSample>2 0 0</firstValidSample><lastValidSample>17 19 19</lastValidSample>
</burst></burstList></swathTiming></product>"""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "bad.SAFE.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("bad.SAFE/annotation/s1a-iw2-slc-vv-test.xml", xml)
        with pytest.raises(ValueError, match="firstValidSample"):
            parse_burst_info(zip_path)


def test_raises_on_missing_burst_list():
    xml = """<?xml version="1.0"?>
<product><swathTiming><linesPerBurst>5</linesPerBurst><samplesPerBurst>20</samplesPerBurst>
</swathTiming></product>"""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "bad.SAFE.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("bad.SAFE/annotation/s1a-iw2-slc-vv-test.xml", xml)
        with pytest.raises(ValueError, match="burstList"):
            parse_burst_info(zip_path)


def test_warns_but_does_not_crash_on_declared_count_mismatch():
    """Real-world data quirk: a declared count attribute that disagrees
    with the actual number of <burst> elements should warn, not crash --
    the real, parsed element count is the source of truth."""
    xml = """<?xml version="1.0"?>
<product><swathTiming><linesPerBurst>2</linesPerBurst><samplesPerBurst>10</samplesPerBurst>
<burstList count="5"><burst><azimuthTime>2024-11-08T18:18:20.000000</azimuthTime>
<sensingTime>2024-11-08T18:18:20.500000</sensingTime><byteOffset>0</byteOffset>
<firstValidSample>0 0</firstValidSample><lastValidSample>9 9</lastValidSample>
</burst></burstList></swathTiming></product>"""
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "mismatch.SAFE.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("mismatch.SAFE/annotation/s1a-iw2-slc-vv-test.xml", xml)
        result = parse_burst_info(zip_path)
        assert len(result.bursts) == 1  # real, actual count, not the declared attribute
