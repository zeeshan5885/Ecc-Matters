# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2018-2022)
#
# This file is part of GWDataFind.
#
# GWDataFind is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWDataFind is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWDataFind.  If not, see <http://www.gnu.org/licenses/>.

import warnings
from functools import partial
from unittest import mock

import pytest

from ligo import segments

from .. import (api, ui)
from . import yield_fixture

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_SERVER = "test.datafind.org"
TEST_URL_BASE = "https://{}".format(TEST_SERVER)
TEST_DATA = {
    "A": {
        "A1_TEST": [(0, 10), (10, 20), (30, 50)],
        "A2_TEST": [],
        "A1_PROD": [],
    },
    "B": {
        "B1_TEST",
    },
    "C": {
        "C1_TEST",
    },
}


def _url(suffix):
    return "{}/{}".format(TEST_URL_BASE, suffix)


@yield_fixture(autouse=True)
def gwdatafind_server_env():
    with mock.patch.dict(
        "os.environ",
        {"GWDATAFIND_SERVER": TEST_SERVER},
    ):
        yield


@yield_fixture(autouse=True, scope="module")
def noauth():
    """Force the underlying _get() function to use no authentication.

    So that the tests don't fall over if the test runner has bad creds.
    """
    _get_noauth = partial(ui._get, cert=False, token=False)
    with mock.patch("gwdatafind.ui._get", _get_noauth):
        yield


@pytest.mark.parametrize(("in_", "url"), (
    # no scheme and no port, default to https
    ("datafind.example.com", "https://datafind.example.com"),
    # scheme specified, do nothing
    ("test://datafind.example.com", "test://datafind.example.com"),
    ("test://datafind.example.com:1234", "test://datafind.example.com:1234"),
    ("https://datafind.example.com:80", "https://datafind.example.com:80"),
    # no scheme and port 80, use http
    ("datafind.example.com:80", "http://datafind.example.com:80"),
    # no scheme and port != 80, use https
    ("datafind.example.com:443", "https://datafind.example.com:443"),
    # default host
    (None, TEST_URL_BASE),
))
def test_url_scheme_handling(in_, url):
    assert ui._url(in_, lambda: "test") == f"{url}/test"


def test_ping(requests_mock):
    requests_mock.get(_url(api.ping_path()), status_code=200)
    ui.ping()


@pytest.mark.parametrize(("match", "result"), (
    (None, ("A", "B", "C")),
    ("[AB]", ("A", "B")),
))
def test_find_observatories(match, result, requests_mock):
    requests_mock.get(
        _url(api.find_observatories_path()),
        json=list(TEST_DATA),
    )
    assert ui.find_observatories(match=match) == list(set(result))


@pytest.mark.parametrize(("site", "match", "result"), (
    (None, None, [ft for site in TEST_DATA for ft in TEST_DATA[site]]),
    ("A", None, list(TEST_DATA["A"])),
    ("A", "PROD", ["A1_PROD"]),
))
def test_find_types(site, match, result, requests_mock):
    if site:
        respdata = list(TEST_DATA[site])
    else:
        respdata = [ft for site in TEST_DATA for ft in TEST_DATA[site]]
    requests_mock.get(
        _url(api.find_types_path(site=site)),
        json=respdata,
    )
    assert ui.find_types(
        site=site,
        match=match,
    ) == list(set(result))


def test_find_times(requests_mock):
    site = "A"
    frametype = "A1_TEST"
    requests_mock.get(
        _url(api.find_times_path(site, frametype, 1, 100)),
        json=TEST_DATA[site][frametype],
    )
    assert ui.find_times(site, frametype, 1, 100) == segments.segmentlist([
        segments.segment(0, 10),
        segments.segment(10, 20),
        segments.segment(30, 50),
    ])


def test_find_url(requests_mock):
    urls = [
        "file:///data/A/A1_TEST/A-A1_TEST-0-1.gwf",
        "gsiftp://localhost:2811/data/A/A1_TEST/A-A1_TEST-0-1.gwf",
    ]
    requests_mock.get(
        _url(api.find_url_path("A-A1_TEST-0-1.gwf")),
        json=urls,
    )
    assert ui.find_url("/my/data/A-A1_TEST-0-1.gwf") == urls[:1]
    assert ui.find_url("/my/data/A-A1_TEST-0-1.gwf", urltype=None) == urls
    assert ui.find_url(
        "/my/data/A-A1_TEST-0-1.gwf",
        urltype="gsiftp",
    ) == urls[1:]


def test_find_url_on_missing(requests_mock):
    requests_mock.get(
        _url(api.find_url_path("A-A1_TEST-0-1.gwf")),
        json=[],
    )

    # on_missing="ignore"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ui.find_url("A-A1_TEST-0-1.gwf", on_missing="ignore") == []

    # on_missing="warn"
    with pytest.warns(UserWarning):
        assert ui.find_url("A-A1_TEST-0-1.gwf", on_missing="warn") == []

    # on_missing="error"
    with pytest.raises(RuntimeError):
        ui.find_url("A-A1_TEST-0-1.gwf", on_missing="error")


def test_find_latest(requests_mock):
    # NOTE: the target function is essentially identical to
    #       find_url, so we just do a minimal smoke test here
    urls = [
        "file:///data/A/A1_TEST/A-A1_TEST-0-1.gwf",
        "gsiftp://localhost:2811/data/A/A1_TEST/A-A1_TEST-0-1.gwf",
    ]
    requests_mock.get(
        _url(api.find_latest_path("A", "A1_TEST", "file")),
        json=urls[:1],
    )
    assert ui.find_latest("A", "A1_TEST") == urls[:1]


def _file_url(seg):
    return "file:///data/A/A1_TEST/A-A1_TEST-{}-{}.gwf".format(
        seg[0],
        seg[1]-seg[0],
    )


def test_find_urls(requests_mock):
    urls = list(map(_file_url, TEST_DATA["A"]["A1_TEST"][:2]))
    requests_mock.get(
        _url(api.find_urls_path("A", "A1_TEST", 0, 20, "file")),
        json=urls,
    )
    assert ui.find_urls("A", "A1_TEST", 0, 20, on_gaps="error") == urls


def test_find_urls_on_gaps(requests_mock):
    urls = list(map(_file_url, TEST_DATA["A"]["A1_TEST"]))
    requests_mock.get(
        _url(api.find_urls_path("A", "A1_TEST", 0, 100, "file")),
        json=urls,
    )

    # on_gaps="ignore"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ui.find_urls("A", "A1_TEST", 0, 100, on_gaps="ignore") == urls

    # on_missing="warn"
    with pytest.warns(UserWarning):
        assert ui.find_urls("A", "A1_TEST", 0, 100, on_gaps="warn") == urls

    # on_missing="error"
    with pytest.raises(RuntimeError):
        ui.find_urls("A", "A1_TEST", 0, 100, on_gaps="error")
