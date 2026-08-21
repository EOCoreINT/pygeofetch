"""
Validates pygeofetch.providers.copernicus_nodes -- URL construction,
response parsing, and (critically) that the mini-zip it produces is
directly readable by the REAL annotation.parse_burst_info(), end to
end, with no modifications to that function.

Everything here uses mocked HTTP responses (via unittest.mock) -- no
live network calls. See the module's own docstring for what that does
and doesn't confirm about the real, live CDSE API.
"""
import sys
sys.path.insert(0, "/home/claude/work/pygeofetch2")

from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile


import pygeofetch.providers.copernicus_nodes as nodes_mod


class FakeSecretStr:
    def __init__(self, value):
        self._value = value

    def get_secret_value(self):
        return self._value


class FakeAuthSession:
    def __init__(self, access_token):
        self.access_token = access_token


class FakeAuthManager:
    def __init__(self, sessions):
        self._sessions = sessions

    def get_session(self, provider):
        return self._sessions.get(provider)


class FakeClient:
    def __init__(self, sessions):
        self.auth = FakeAuthManager(sessions)


class FakeSatelliteData:
    def __init__(self, id_, name):
        self.id = id_
        self.properties = {"name": name}


def test_unwrap_token_plain_string():
    print("=== 1. _unwrap_token handles a plain string ===")
    assert nodes_mod._unwrap_token("abc123") == "abc123"
    print("  PASS\n")


def test_unwrap_token_secretstr():
    print("=== 2. _unwrap_token handles a SecretStr-like object ===")
    assert nodes_mod._unwrap_token(FakeSecretStr("abc123")) == "abc123"
    print("  PASS\n")


def test_get_bearer_token_no_session_raises():
    print("=== 3. get_bearer_token raises clearly when there's no valid session ===")
    client = FakeClient({})
    try:
        nodes_mod.get_bearer_token(client, "copernicus")
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        print(f"  correctly raised: {exc}")
    print("  PASS\n")


def test_nodes_url_construction_and_escaping():
    print("=== 4. _nodes_url builds correct URLs, including quote-escaping ===")
    url = nodes_mod._nodes_url("UUID123", ["My.SAFE", "annotation"], list_children=True)
    print(f"  listing url: {url}")
    assert url == "https://download.dataspace.copernicus.eu/odata/v1/Products(UUID123)/Nodes('My.SAFE')/Nodes('annotation')/Nodes"

    url2 = nodes_mod._nodes_url("UUID123", ["My.SAFE", "annotation", "file.xml"], list_children=False)
    print(f"  content url:  {url2}")
    assert url2.endswith("/Nodes('file.xml')/$value")

    # quote escaping
    url3 = nodes_mod._nodes_url("UUID123", ["weird'name"], list_children=True)
    assert "weird''name" in url3, f"single quote should be OData-doubled, got: {url3}"
    print("  PASS\n")


def test_list_nodes_result_key():
    print("=== 5. list_nodes parses the real documented 'result' key ===")
    client = FakeClient({"copernicus": FakeAuthSession("tok")})
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"result": [{"Name": "annotation", "ChildrenNumber": 3}]}
    fake_resp.raise_for_status.return_value = None

    with patch("httpx.get", return_value=fake_resp) as mock_get:
        result = nodes_mod.list_nodes(client, "UUID", ["My.SAFE"])
        assert result == [{"Name": "annotation", "ChildrenNumber": 3}]
        # confirm the real auth header was actually sent
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["Authorization"] == "Bearer tok"
    print("  PASS\n")


def test_list_nodes_value_key_fallback():
    print("=== 6. list_nodes falls back to 'value' key if 'result' is absent ===")
    client = FakeClient({"copernicus": FakeAuthSession("tok")})
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"value": [{"Name": "annotation"}]}
    fake_resp.raise_for_status.return_value = None
    with patch("httpx.get", return_value=fake_resp):
        result = nodes_mod.list_nodes(client, "UUID", ["My.SAFE"])
        assert result == [{"Name": "annotation"}]
    print("  PASS\n")


def test_list_nodes_unknown_shape_raises_not_silent():
    print("=== 7. list_nodes raises (doesn't silently return []) on an unrecognized response shape ===")
    client = FakeClient({"copernicus": FakeAuthSession("tok")})
    fake_resp = MagicMock()
    fake_resp.json.return_value = {"totally_unexpected_key": []}
    fake_resp.raise_for_status.return_value = None
    with patch("httpx.get", return_value=fake_resp):
        try:
            nodes_mod.list_nodes(client, "UUID", ["My.SAFE"])
            raise AssertionError("expected ValueError, not a silent empty result")
        except ValueError as exc:
            print(f"  correctly raised: {exc}")
    print("  PASS\n")


def test_find_annotation_members_filters_correctly():
    print("=== 8. find_annotation_members excludes calibration/RFI and filters by polarisation ===")
    client = FakeClient({"copernicus": FakeAuthSession("tok")})
    sat = FakeSatelliteData("UUID", "S1A_TEST.SAFE")

    responses = [
        {"result": [{"Name": "annotation", "ChildrenNumber": 6}]},  # top-level listing
        {"result": [  # annotation folder listing
            {"Name": "s1a-iw1-slc-vv-20190721.xml", "ChildrenNumber": 0},
            {"Name": "s1a-iw2-slc-vv-20190721.xml", "ChildrenNumber": 0},
            {"Name": "s1a-iw1-slc-vh-20190721.xml", "ChildrenNumber": 0},
            {"Name": "calibration-s1a-iw1-slc-vv-20190721.xml", "ChildrenNumber": 0},
            {"Name": "rfi-s1a-iw1-slc-vv-20190721.xml", "ChildrenNumber": 0},
            {"Name": "not-an-xml.txt", "ChildrenNumber": 0},
        ]},
    ]

    def fake_get(url, headers=None, timeout=None):
        fake_resp = MagicMock()
        fake_resp.json.return_value = responses.pop(0)
        fake_resp.raise_for_status.return_value = None
        return fake_resp

    with patch("httpx.get", side_effect=fake_get):
        members = nodes_mod.find_annotation_members(client, sat, polarisation="vv")

    names = sorted(f for _, f in members)
    print(f"  found members: {names}")
    assert names == ["s1a-iw1-slc-vv-20190721.xml", "s1a-iw2-slc-vv-20190721.xml"], \
        "should keep only real VV, non-calibration, non-RFI XML files"
    print("  PASS\n")


REAL_SHAPED_ANNOTATION_XML = """<?xml version="1.0" encoding="UTF-8"?>
<product>
  <swathTiming>
    <linesPerBurst>3</linesPerBurst>
    <samplesPerBurst>20000</samplesPerBurst>
    <burstList count="2">
      <burst>
        <azimuthTime>2019-07-21T18:18:01.251000</azimuthTime>
        <sensingTime>2019-07-21T18:18:01.251000</sensingTime>
        <byteOffset>0</byteOffset>
        <firstValidSample count="3">0 0 0</firstValidSample>
        <lastValidSample count="3">19999 19999 19999</lastValidSample>
      </burst>
      <burst>
        <azimuthTime>2019-07-21T18:18:03.759000</azimuthTime>
        <sensingTime>2019-07-21T18:18:03.759000</sensingTime>
        <byteOffset>60000000</byteOffset>
        <firstValidSample count="3">0 0 0</firstValidSample>
        <lastValidSample count="3">19999 19999 19999</lastValidSample>
      </burst>
    </burstList>
  </swathTiming>
</product>
"""


def test_end_to_end_zip_readable_by_real_parse_burst_info():
    print("=== 9. END-TO-END: the produced mini-zip is readable by the REAL parse_burst_info(), unmodified ===")
    client = FakeClient({"copernicus": FakeAuthSession("tok")})
    sat = FakeSatelliteData("UUID", "S1A_TEST_END2END.SAFE")

    list_responses = [
        {"result": [{"Name": "annotation", "ChildrenNumber": 1}]},
        {"result": [{"Name": "s1a-iw1-slc-vv-20190721.xml", "ChildrenNumber": 0}]},
    ]

    def fake_get(url, headers=None, timeout=None):
        fake_resp = MagicMock()
        fake_resp.raise_for_status.return_value = None
        if url.endswith("/Nodes"):
            fake_resp.json.return_value = list_responses.pop(0)
        else:
            assert url.endswith("/$value")
            fake_resp.content = REAL_SHAPED_ANNOTATION_XML.encode("utf-8")
        return fake_resp

    with tempfile.TemporaryDirectory() as tmpdir:
        with patch("httpx.get", side_effect=fake_get):
            zip_path = nodes_mod.fetch_annotation_zip(client, sat, tmpdir, polarisation="vv")

        print(f"  produced: {zip_path}")
        assert zip_path.exists()

        # Now hand it to the REAL, unmodified annotation.parse_burst_info()
        sys.path.insert(0, "/home/claude/work/pygeofetch2")
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "pygeofetch.insar.annotation", "/home/claude/work/pygeofetch2/pygeofetch/insar/annotation.py",
        )
        annotation_mod = importlib.util.module_from_spec(spec)
        sys.modules["pygeofetch.insar.annotation"] = annotation_mod
        spec.loader.exec_module(annotation_mod)

        swath_timing = annotation_mod.parse_burst_info(zip_path)
        print(f"  real parse_burst_info() succeeded: {len(swath_timing.bursts)} bursts, "
              f"linesPerBurst={swath_timing.lines_per_burst}")
        assert len(swath_timing.bursts) == 2
        assert swath_timing.lines_per_burst == 3
    print("  PASS -- the real, unmodified annotation parser reads this mini-zip correctly\n")


if __name__ == "__main__":
    test_unwrap_token_plain_string()
    test_unwrap_token_secretstr()
    test_get_bearer_token_no_session_raises()
    test_nodes_url_construction_and_escaping()
    test_list_nodes_result_key()
    test_list_nodes_value_key_fallback()
    test_list_nodes_unknown_shape_raises_not_silent()
    test_find_annotation_members_filters_correctly()
    test_end_to_end_zip_readable_by_real_parse_burst_info()
    print("ALL TESTS PASSED")
