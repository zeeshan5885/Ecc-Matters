# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2017-2022)
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

"""The client library for the GWDataFind service.

The GWDataFind service allows users to query for the location of
Gravitational-Wave data files containing data associated with
gravitational-wave detectors.

This package provides a number of functions to make requests to
a GWDataFind server with authorization credential handling.

-----------
Quick-start
-----------

The following convenience functions are provided to perform single queries
without a persistent connection:

.. currentmodule:: gwdatafind

.. autosummary::
    :nosignatures:

    ping
    find_observatories
    find_types
    find_times
    find_url
    find_urls
    find_latest

For example:

>>> from gwdatafind import find_urls
>>> urls = find_urls("L", "L1_GWOSC_O2_4KHZ_R1", 1187008880, 1187008884,
...                  host="datafind.gw-openscience.org")
>>> print(urls)
['file://localhost/cvmfs/gwosc.osgstorage.org/gwdata/O2/strain.4k/frame.v1/L1/1186988032/L-L1_GWOSC_O2_4KHZ_R1-1187008512-4096.gwf']

Additionally, one can create a `requests.Session` to reuse a
connection to a server.

For example:

>>> from gwdatafind import (find_observatories, find_urls, Session)
>>> with Session() as sess:
...     obs = find_observatories(
...               host="datafind.gw-openscience.org",
...               session=sess,
...     )
...     print(obs)
...     urls = {}
...     for ifo in obs:
...         urls[ifo] = find_urls(
...             ifo,
...             "{}1_GWOSC_O2_4KHZ_R1".format(ifo),
...             1187008880,
...             1187008884,
...             host="datafind.gw-openscience.org",
...             session=sess,
...         )
...     print(urls)
['H', 'V', 'L']
{'H': ['file://localhost/cvmfs/gwosc.osgstorage.org/gwdata/O2/strain.4k/frame.v1/H1/1186988032/H-H1_GWOSC_O2_4KHZ_R1-1187008512-4096.gwf'],
 'V': ['file://localhost/cvmfs/gwosc.osgstorage.org/gwdata/O2/strain.4k/frame.v1/V1/1186988032/V-V1_GWOSC_O2_4KHZ_R1-1187008512-4096.gwf'],
 'L': ['file://localhost/cvmfs/gwosc.osgstorage.org/gwdata/O2/strain.4k/frame.v1/L1/1186988032/L-L1_GWOSC_O2_4KHZ_R1-1187008512-4096.gwf']}
"""  # noqa: E501

from igwn_auth_utils.requests import Session

from .http import *
from .ui import *

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Scott Koranda <scott.koranda@ligo.org>'
__version__ = '1.1.3'
