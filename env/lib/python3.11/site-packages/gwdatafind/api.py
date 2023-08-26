# -*- coding: utf-8 -*-
# Copyright (C) 2012-2015  Scott Koranda, 2015+ Duncan Macleod
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

"""API URL implementation for GWDataFind.

The functions in this module return URL paths to request on a given host
to execute various GWDataFind queries.
"""

from functools import wraps
from os.path import basename

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

DEFAULT_SERVICE_PREFIX = "LDR/services/data/v1"


def _prefix(func):
    """Wrap ``func`` to prepend the path prefix automatically.

    This just simplifies the functional constructions below.
    """
    @wraps(func)
    def wrapped(*args, **kwargs):
        prefix = kwargs.pop("prefix", DEFAULT_SERVICE_PREFIX)
        suffix = func(*args, **kwargs)
        return "{}/{}".format(prefix, suffix)

    return wrapped


@_prefix
def ping_path():
    """Return the API path to ping the server.
    """
    return "gwf/H/R/1,2"


@_prefix
def find_observatories_path():
    """Return the API path to query for all observatories.
    """
    return "gwf.json"


@_prefix
def find_types_path(site=None):
    """Return the API path to query for datasets for one or all sites.
    """
    if site:
        return "gwf/{site[0]}.json".format(site=site)
    return "gwf/all.json"


@_prefix
def find_times_path(site, frametype, start, end):
    """Return the API path to query for data availability segments.
    """
    if start is None and end is None:
        return "gwf/{site}/{type}/segments.json".format(
            site=site,
            type=frametype,
        )
    return "gwf/{site}/{type}/segments/{start},{end}.json".format(
        site=site,
        type=frametype,
        start=start,
        end=end,
    )


@_prefix
def find_url_path(framefile):
    """Return the API path to query for the URL of a specific filename.
    """
    filename = basename(framefile)
    site, frametype, _ = filename.split("-", 2)
    return "gwf/{site}/{type}/{filename}.json".format(
        site=site,
        type=frametype,
        filename=filename,
    )


@_prefix
def find_latest_path(site, frametype, urltype):
    """Return the API path to query for the latest file in a dataset.
    """
    stub = "gwf/{site}/{type}/latest".format(
        site=site,
        type=frametype,
    )
    if urltype:
        return "{stub}/{urltype}.json".format(stub=stub, urltype=urltype)
    return stub + ".json"


@_prefix
def find_urls_path(site, frametype, start, end, urltype=None, match=None):
    """Return the API path to query for all URLs for a dataset in an interval.
    """
    stub = "gwf/{site}/{type}/{start},{end}".format(
        site=site,
        type=frametype,
        start=start,
        end=end,
    )
    if urltype:
        path = "{stub}/{urltype}.json".format(stub=stub, urltype=urltype)
    else:
        path = stub + ".json"
    if match:
        path += "?match={0}".format(match)
    return path
