# -*- coding: utf-8 -*-
# Copyright (C) Scott Koranda (2012-2015)
#               Louisiana State University (2015-2017)
#               Cardiff University (2017-2022)
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Tests for :mod:`gwdatafind.utils`.
"""

from unittest import mock

import pytest

from .. import utils


@mock.patch.dict(
    "os.environ",
    {
        "GWDATAFIND_SERVER": "gwtest",
        "LIGO_DATAFIND_SERVER": "ligotest",
    },
)
def test_get_default_host():
    assert utils.get_default_host() == "gwtest"


@mock.patch.dict(
    "os.environ",
    {"LIGO_DATAFIND_SERVER": "ligotest"},
    clear=True,
)
def test_get_default_host_ligo():
    assert utils.get_default_host() == "ligotest"


@mock.patch.dict("os.environ", clear=True)
def test_get_default_host_error():
    with pytest.raises(ValueError):
        utils.get_default_host()


@mock.patch(
    "igwn_auth_utils.x509.validate_certificate",
    side_effect=(None, RuntimeError),
)
def test_validate_proxy(_):
    # check that no error ends up as 'True'
    with pytest.warns(DeprecationWarning):
        assert utils.validate_proxy("something") is True
    # but that an error is forwarded
    with pytest.warns(DeprecationWarning), pytest.raises(RuntimeError):
        assert utils.validate_proxy("something else")


@mock.patch(
    "igwn_auth_utils.x509.find_credentials",
    side_effect=("cert", ("cert", "key")),
)
def test_find_credential(_):
    # check that if upstream returns a single cert, we still get a tuple
    with pytest.warns(DeprecationWarning):
        assert utils.find_credential() == ("cert", "cert")
    # but if it returns a tuple, we get a tuple
    with pytest.warns(DeprecationWarning):
        assert utils.find_credential() == ("cert", "key")
