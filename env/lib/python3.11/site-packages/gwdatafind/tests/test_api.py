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

"""Test suite for `gwdatafind.api`.

This just asserts that the API implementation here matches the expectation
from the v1 API for gwdatfind_server.
"""

import pytest

from .. import api

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_ping_path():
    assert api.ping_path() == "LDR/services/data/v1/gwf/H/R/1,2"


def test_find_observatories_path():
    assert api.find_observatories_path() == "LDR/services/data/v1/gwf.json"


@pytest.mark.parametrize(("site", "result"), (
    (None, "LDR/services/data/v1/gwf/all.json"),
    ("X1", "LDR/services/data/v1/gwf/X.json"),
    ("X", "LDR/services/data/v1/gwf/X.json"),
))
def test_find_types_path(site, result):
    assert api.find_types_path(site) == result


def test_find_times_path():
    assert api.find_times_path("X", "TEST", 0, 1) == (
        "LDR/services/data/v1/gwf/X/TEST/segments/0,1.json"
    )


def test_find_url_path():
    assert api.find_url_path("/data/X-TEST-0-1.gwf") == (
        "LDR/services/data/v1/gwf/X/TEST/X-TEST-0-1.gwf.json"
    )


@pytest.mark.parametrize(("urltype", "result"), (
    (None, "LDR/services/data/v1/gwf/X/TEST/latest.json"),
    ("file", "LDR/services/data/v1/gwf/X/TEST/latest/file.json"),
))
def test_find_latest_path(urltype, result):
    assert api.find_latest_path("X", "TEST", urltype) == result


@pytest.mark.parametrize(("urltype", "match", "result"), (
    (None, None, "LDR/services/data/v1/gwf/X/TEST/0,1.json"),
    ("gsiftp", None, "LDR/services/data/v1/gwf/X/TEST/0,1/gsiftp.json"),
    (None, "test", "LDR/services/data/v1/gwf/X/TEST/0,1.json?match=test"),
    ("file", "test",
     "LDR/services/data/v1/gwf/X/TEST/0,1/file.json?match=test"),
))
def find_urls_path(urltype, match, result):
    assert api.find_urls_path(
        "X",
        "TEST",
        0,
        1,
        urltype=urltype,
        match=match,
    ) == result
