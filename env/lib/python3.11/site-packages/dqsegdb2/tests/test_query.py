# DQSEGDB2
# Copyright (C) 2018  Duncan Macleod
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for :mod:`dqsegdb.api`
"""

from unittest import mock

import pytest

import dqsegdb2

TEST_SEGMENT_SERVER = "https://dqsegdb.example.com"

KNOWN = [[0, 10]]
ACTIVE = [[1, 3], [3, 4], [6, 10]]
QUERY_SEGMENT = [2, 8]
KNOWN_COALESCED = [[2, 8]]
ACTIVE_COALESCED = [[2, 4], [6, 8]]


IFOS_RESPONSE = {
    "query_information": {"uri": "/dq"},
    "Ifos": ["C1", "A1", "B1"],
}
NAMES_RESPONSE = {
    "results": ['name1', 'name2', 'name2'],
}
VERSIONS_RESPONSE = {
    "version": [1, 2, 3, 4],
}
SEGMENTS_RESPONSE = {
    "ifo": "X1",
    "name": "TEST",
    "version": 1,
    "known": KNOWN,
    "active": ACTIVE
}


@pytest.fixture(autouse=True)
def mock_default_segment_server():
    with mock.patch.dict(
        "os.environ",
        {"DEFAULT_SEGMENT_SERVER": TEST_SEGMENT_SERVER},
    ):
        yield


@pytest.mark.parametrize("raw", (False, True))
def test_query_ifos(requests_mock, raw):
    """Check that `dqsegdb2.query_ifos2` works as advertised.
    """
    requests_mock.get(
        "https://dqsegdb.example.com/dq",
        json=IFOS_RESPONSE,
    )
    # disable all auth just in case DEFAULT_SEGMENT_SERVER doesn't work
    resp = dqsegdb2.query_ifos(raw=raw, cert=False, token=False)
    if raw:
        assert resp == IFOS_RESPONSE
    else:
        assert resp == {"A1", "B1", "C1"}


@pytest.mark.parametrize("raw", (False, True))
def test_query_names(requests_mock, raw):
    requests_mock.get(
        "https://dqsegdb.example.com/dq/X1",
        json=NAMES_RESPONSE,
    )
    resp = dqsegdb2.query_names(
        "X1",
        host="https://dqsegdb.example.com",
        raw=raw,
        cert=False,
        token=False,
    )
    if raw:
        assert resp == NAMES_RESPONSE
    else:
        assert set(resp) == set(['X1:name1', 'X1:name2'])


@pytest.mark.parametrize("raw", (False, True))
def test_query_versions(requests_mock, raw):
    requests_mock.get(
        "https://dqsegdb.example.com/dq/X1/test",
        json=VERSIONS_RESPONSE,
    )
    resp = dqsegdb2.query_versions(
        "X1:test",
        host="https://dqsegdb.example.com",
        raw=raw,
    )
    if raw:
        assert resp == VERSIONS_RESPONSE
    else:
        assert resp == [1, 2, 3, 4]


@pytest.mark.parametrize("raw", (False, True))
@pytest.mark.parametrize('flag, coalesce, known, active', [
    ("X1:TEST:1", False, KNOWN, ACTIVE),
    ("X1:TEST:1", True, KNOWN_COALESCED, ACTIVE_COALESCED),
    ("X1:TEST:*", False, KNOWN + KNOWN, ACTIVE + ACTIVE),
    ("X1:TEST:*", True, KNOWN_COALESCED, ACTIVE_COALESCED),
])
def test_query_segments(flag, coalesce, known, active, raw, requests_mock):
    # mock the request
    versions = (1, 2)
    url = "https://dqsegdb.example.com/dq/X1/TEST"
    requests_mock.get(url, json={"version": versions})
    for ver in versions:
        vurl = url + f"/{ver}?s=2&e=8&include=active%2Cknown"
        if raw:
            vurl += "%2Cmetadata"
        requests_mock.get(vurl, json=SEGMENTS_RESPONSE)

    # check that we get the result we expect
    resp = dqsegdb2.query_segments(
        flag,
        2,
        8,
        coalesce=coalesce,
        host="https://dqsegdb.example.com",
        raw=raw,
    )

    # if raw multi-version query, result is a list of JSON responses
    if raw and flag.endswith("*"):
        assert resp == [
            SEGMENTS_RESPONSE,
            SEGMENTS_RESPONSE,
        ]
    # if raw single-version query, result is a single raw JSON response
    elif raw:
        assert resp == SEGMENTS_RESPONSE
    # if not raw query, result is a formatted JSON response
    else:
        assert resp.pop('version') == (None if flag.endswith('*') else 1)
        assert resp.pop('known') == list(map(tuple, known))
        assert resp.pop('active') == list(map(tuple, active))
        for key in set(SEGMENTS_RESPONSE) & set(resp):
            assert resp[key] == SEGMENTS_RESPONSE[key]
