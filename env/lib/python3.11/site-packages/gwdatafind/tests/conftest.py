# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2018-2022)
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

"""Test utilities.
"""

import os
import tempfile
from unittest import mock

from . import yield_fixture


@yield_fixture
def response():
    """Patch an HTTPConnection to do nothing in particular.

    Yields the patch for `http.client.HTTPConnection.getresponse`
    """
    with mock.patch('http.client.HTTPConnection.request'), \
         mock.patch('http.client.HTTPConnection.getresponse') as resp:
        yield resp


@yield_fixture
def tmpname():
    """Return a temporary file name, cleaning up after the method returns.
    """
    name = tempfile.mktemp()
    open(name, 'w').close()
    try:
        yield name
    finally:
        if os.path.isfile(name):
            os.remove(name)
