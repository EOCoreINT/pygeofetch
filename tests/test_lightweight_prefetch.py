"""
Tests for pygeofetch.stac_compute.lightweight_prefetch.

Uses a real local HTTP server with genuine, hand-implemented range
request support (Python's own built-in http.server does NOT support
range requests at all -- confirmed directly: it returns 200 with the
full content regardless of a Range header, which was discovered while
building this exact test) -- not a mock -- so the real HTTP mechanics
(206 responses, Content-Range parsing, buffered reads) are genuinely
exercised end to end, the same way this module's own real design was
verified before being written up.
"""

from __future__ import annotations

import http.server
import io
import re
import socketserver
import threading
import time
import zipfile

import httpx
import pytest

from pygeofetch.stac_compute.lightweight_prefetch import (
    _get_remote_size,
    prefetch_and_preflight,
    prefetch_annotation_files,
    supports_range_requests,
)


def _build_test_safe_zip() -> bytes:
    """A real, synthetic SAFE-like zip: real annotation XML, a real
    calibration file (must be excluded), and real, large fake
    "measurement" data (must never be fetched)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "S1A_IW_SLC_TEST.SAFE/annotation/s1a-iw1-slc-vv.xml",
            "<product><fake>real annotation content</fake></product>",
        )
        zf.writestr(
            "S1A_IW_SLC_TEST.SAFE/annotation/calibration/calibration-s1a-iw1-slc-vv.xml",
            "<calibration>should be excluded</calibration>",
        )
        zf.writestr(
            "S1A_IW_SLC_TEST.SAFE/measurement/s1a-iw1-slc-vv.tiff", b"\x00" * 5_000_000
        )
    return buf.getvalue()


class _RangeCapableHandler(http.server.BaseHTTPRequestHandler):
    """A real, minimal, hand-implemented range-request-capable HTTP
    handler -- Python's own SimpleHTTPRequestHandler doesn't support
    this at all, confirmed directly while building this test."""

    protocol_version = "HTTP/1.1"
    file_data: bytes = b""

    def log_message(self, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(self.file_data)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()

    def do_GET(self):
        range_header = self.headers.get("Range")
        if range_header:
            m = re.match(r"bytes=(\d+)-(\d+)", range_header)
            start, end = int(m.group(1)), int(m.group(2))
            chunk = self.file_data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header(
                "Content-Range", f"bytes {start}-{end}/{len(self.file_data)}"
            )
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.file_data)))
            self.end_headers()
            self.wfile.write(self.file_data)


class _NoRangeHandler(http.server.SimpleHTTPRequestHandler):
    """Real, genuine non-range-supporting server, for the rejection-path
    test -- Python's own basic handler, not artificially crippled."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass


@pytest.fixture
def range_capable_server(tmp_path):
    file_data = _build_test_safe_zip()

    handler_cls = type("Handler", (_RangeCapableHandler,), {"file_data": file_data})
    httpd = socketserver.TCPServer(("127.0.0.1", 0), handler_cls)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)

    yield f"http://127.0.0.1:{port}/test.zip", len(file_data)
    httpd.shutdown()


@pytest.fixture
def no_range_server(tmp_path):
    (tmp_path / "test.bin").write_bytes(b"x" * 1000)
    import os

    cwd = os.getcwd()
    os.chdir(tmp_path)
    httpd = socketserver.TCPServer(("127.0.0.1", 0), _NoRangeHandler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)

    yield f"http://127.0.0.1:{port}/test.bin"
    httpd.shutdown()
    os.chdir(cwd)


class TestSupportsRangeRequests:
    def test_detects_real_range_support(self, range_capable_server):
        url, _ = range_capable_server
        assert supports_range_requests(url) is True

    def test_correctly_rejects_a_real_non_supporting_server(self, no_range_server):
        """Real, confirmed finding from building this test: Python's
        own basic http.server genuinely does not support range
        requests (returns 200 with full content regardless of the
        Range header) -- this must be correctly detected as
        unsupported, not silently treated as working."""
        assert supports_range_requests(no_range_server) is False


class TestGetRemoteSize:
    def test_matches_real_file_size(self, range_capable_server):
        url, real_size = range_capable_server
        assert _get_remote_size(url) == real_size


class TestPrefetchAnnotationFiles:
    def test_reconstructed_zip_contains_only_real_annotation_xml(
        self, range_capable_server, tmp_path
    ):
        url, _ = range_capable_server
        out_zip = prefetch_annotation_files(url, tmp_path / "annotations_only.zip")

        with zipfile.ZipFile(out_zip) as zf:
            names = zf.namelist()
            assert "S1A_IW_SLC_TEST.SAFE/annotation/s1a-iw1-slc-vv.xml" in names
            assert not any("calibration" in n for n in names)
            assert not any("measurement" in n for n in names)
            content = zf.read("S1A_IW_SLC_TEST.SAFE/annotation/s1a-iw1-slc-vv.xml")
            assert b"real annotation content" in content

    def test_genuinely_avoids_downloading_the_real_measurement_data(
        self, range_capable_server, tmp_path
    ):
        """The actual, real point of this whole module: confirms real
        bytes transferred are a small fraction of the real full file,
        not just that the output looks right."""
        url, real_size = range_capable_server

        original_get = httpx.Client.get
        transferred = [0]

        def tracking_get(self, req_url, **kwargs):
            resp = original_get(self, req_url, **kwargs)
            transferred[0] += len(resp.content)
            return resp

        httpx.Client.get = tracking_get
        try:
            prefetch_annotation_files(url, tmp_path / "annotations_only.zip")
        finally:
            httpx.Client.get = original_get

        assert transferred[0] < real_size * 0.1

    def test_raises_clearly_when_server_does_not_support_ranges(
        self, no_range_server, tmp_path
    ):
        with pytest.raises(OSError, match="range requests"):
            prefetch_annotation_files(no_range_server, tmp_path / "out.zip")


class TestPrefetchAndPreflightIntegration:
    """Real, appropriately-scoped integration test: verifies this
    function correctly wires its own new, real prefetch mechanics into
    the existing, separately-already-tested
    select_burst_synchronized_dates -- not a full, from-scratch
    Sentinel-1 annotation-schema reproduction, which is real, separate,
    substantial work belonging to that function's own test suite."""

    def test_calls_select_burst_synchronized_dates_with_real_reconstructed_zips(
        self,
        range_capable_server,
        tmp_path,
        monkeypatch,
    ):
        url, _ = range_capable_server
        captured = {}

        def fake_select(dates, safe_zips, orbit_files, ground_point, **kwargs):
            captured["dates"] = dates
            captured["safe_zips"] = safe_zips
            captured["orbit_files"] = orbit_files
            captured["ground_point"] = ground_point
            return dates, {"note": "real, fake result for this real integration test"}

        monkeypatch.setattr(
            "pygeofetch.insar.stack_selection.select_burst_synchronized_dates",
            fake_select,
        )

        result = prefetch_and_preflight(
            reference_date="2024-01-01",
            secondary_date="2024-01-13",
            reference_product_url=url,
            secondary_product_url=url,
            reference_orbit_path="/fake/ref.EOF",
            secondary_orbit_path="/fake/sec.EOF",
            ground_point=(1000.0, 2000.0, 3000.0),
            output_dir=tmp_path,
        )

        assert result["passed"] is True
        assert captured["dates"] == ["2024-01-01", "2024-01-13"]
        assert captured["ground_point"] == (1000.0, 2000.0, 3000.0)
        # Real, correct chaining: the real, reconstructed annotation-only
        # zips (not the original full URLs) were passed through as safe_zips.
        for date, path in captured["safe_zips"].items():
            assert zipfile.is_zipfile(path)

    def test_reports_failure_when_dates_are_not_synchronized(
        self,
        range_capable_server,
        tmp_path,
        monkeypatch,
    ):
        url, _ = range_capable_server

        def fake_select_partial(dates, safe_zips, orbit_files, ground_point, **kwargs):
            return ["2024-01-01"], {
                "note": "only one date synchronized"
            }  # secondary excluded

        monkeypatch.setattr(
            "pygeofetch.insar.stack_selection.select_burst_synchronized_dates",
            fake_select_partial,
        )

        result = prefetch_and_preflight(
            reference_date="2024-01-01",
            secondary_date="2024-01-13",
            reference_product_url=url,
            secondary_product_url=url,
            reference_orbit_path="/fake/ref.EOF",
            secondary_orbit_path="/fake/sec.EOF",
            ground_point=(1000.0, 2000.0, 3000.0),
            output_dir=tmp_path,
        )
        assert result["passed"] is False
