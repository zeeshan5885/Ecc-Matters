# -*- coding: utf-8 -*-
# DQSEGDB2
# Copyright (C) 2018,2020,2022  Duncan Macleod
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

"""API implementation for dqsegdb2.

This attemps to define the endpoints for various actions as defined in
§5.1 of https://dcc.ligo.org/LIGO-T1300625-v1/public.
"""

from functools import wraps
from urllib.parse import urlencode

API_BASE_PATH = "/dq"


def _query_string(**params):
    """Encode a query string, ignoring params with value `None`.
    """
    return urlencode(
        {k: v for k, v in params.items() if v is not None},
    )


def _query_path(path, **params):
    """Format a path and query parameters for a DQSegDB query.
    """
    querystr = _query_string(**params)
    if querystr:
        return f"{path}?{querystr}"
    return path


def _query_params(func):
    """Decorate an API path function to attach a query string.
    """
    @wraps(func)
    def wrapper(*args, **query_params):
        return _query_path(func(*args), **query_params)
    return wrapper


# -- /dq API paths

def ifos_path():
    """Return the API path to query for available IFOs.

    Examples
    --------
    >>> from dqsegdb2.api import ifos_path
    >>> ifos_path()
    '/dq'
    """
    return API_BASE_PATH


def flags_path(ifo):
    """Return the API path to query for flags for a given IFO.

    Parameters
    ----------
    ifo : `str`
        The prefix (two-character) for an interferometer.

    Examples
    --------
    >>> from dqsegdb2.api import flags_path
    >>> flags_path('G1')
    '/dq/G1'
    """
    return f"{ifos_path()}/{ifo}"


def versions_path(ifo, flag):
    """Return the API path to query for a specific flag.

    Parameters
    ----------
    ifo : `str`
        The prefix (two-character) for an interferometer.

    flag : `str`
        The name (excluding the IFO prefix) of a flag.

    Examples
    --------
    >>> from dqsegdb2.api import versions_path
    >>> versions_path('G1', 'GEO-SCIENCE')
    '/dq/G1/GEO-SCIENCE'
    """
    return f"{flags_path(ifo)}/{flag}"


@_query_params
def resources_path(ifo, flag, version, s=None, e=None, include=None):
    """Return the API path to query for a specific flag version.

    Parameters
    ----------
    ifo : `str`
        The prefix (two-character) for an interferometer.

    flag : `str`
        The name (excluding the IFO prefix) of a flag.

    version : `int`
        The version number of the flag.

    s : `float`
        The GPS start time of the interval of interest.

    e : `float`
        The GPS end time of the interval of interest.

    include : `list` of `str`
        The resources to include in the returned data.
        Typically one or more of ``'active'``, ``'known'``, or
        ``'metadata'``.

    Examples
    --------
    >>> from dqsegdb2.api import resources_path
    >>> resources_path('G1', 'GEO-SCIENCE', 1)
    '/dq/G1/GEO-SCIENCE/1'
    >>> resources_path('G1', 'GEO-SCIENCE', 1, s=1, e=2)
    '/dq/G1/GEO-SCIENCE/1?s=1&e=2'
    """
    return f"{versions_path(ifo, flag)}/{version}"


@_query_params
def metadata_path(ifo, flag, version, s=None, e=None):
    """Return the API path to query for metadata for a specific flag.

    Parameters
    ----------
    ifo : `str`
        The prefix (two-character) for an interferometer.

    flag : `str`
        The name (excluding the IFO prefix) of a flag.

    version : `int`
        The version number of the flag.

    s : `float`
        The GPS start time of the interval of interest.

    e : `float`
        The GPS end time of the interval of interest.

    Examples
    --------
    >>> from dqsegdb2.api import resources_path
    >>> metadata_path('G1', 'GEO-SCIENCE', 1)
    '/dq/G1/GEO-SCIENCE/1/metadata'
    >>> metadata_path('G1', 'GEO-SCIENCE', 1, s=1, e=2)
    '/dq/G1/GEO-SCIENCE/1/metadata?s=1&e=2'
    """
    return f"{resources_path(ifo, flag, version)}/metadata"


@_query_params
def active_path(ifo, flag, version, s=None, e=None):
    """Return the API path to query for active segments for a specific flag.

    Parameters
    ----------
    ifo : `str`
        The prefix (two-character) for an interferometer.

    flag : `str`
        The name (excluding the IFO prefix) of a flag.

    version : `int`
        The version number of the flag.

    s : `float`
        The GPS start time of the interval of interest.

    e : `float`
        The GPS end time of the interval of interest.

    Examples
    --------
    >>> from dqsegdb2.api import resources_path
    >>> active_path('G1', 'GEO-SCIENCE', 1)
    '/dq/G1/GEO-SCIENCE/1/active'
    >>> active_path('G1', 'GEO-SCIENCE', 1, s=1, e=2)
    '/dq/G1/GEO-SCIENCE/1/active?s=1&e=2'
    """
    return f"{resources_path(ifo, flag, version)}/active"


@_query_params
def known_path(ifo, flag, version, s=None, e=None):
    """Return the API path to query for known segments for a specific flag.

    Parameters
    ----------
    ifo : `str`
        The prefix (two-character) for an interferometer.

    flag : `str`
        The name (excluding the IFO prefix) of a flag.

    version : `int`
        The version number of the flag.

    s : `float`
        The GPS start time of the interval of interest.

    e : `float`
        The GPS end time of the interval of interest.

    Examples
    --------
    >>> from dqsegdb2.api import resources_path
    >>> known_path('G1', 'GEO-SCIENCE', 1)
    '/dq/G1/GEO-SCIENCE/1/known'
    >>> known_path('G1', 'GEO-SCIENCE', 1, s=1, e=2)
    '/dq/G1/GEO-SCIENCE/1/known?s=1&e=2'
    """
    return f"{resources_path(ifo, flag, version)}/known"


@_query_params
def insert_history_path(ifo, flag, version, s=None, e=None):
    """Return the API path to query for the insert history for a specific flag.

    Parameters
    ----------
    ifo : `str`
        The prefix (two-character) for an interferometer.

    flag : `str`
        The name (excluding the IFO prefix) of a flag.

    version : `int`
        The version number of the flag.

    s : `float`
        The GPS start time of the interval of interest.

    e : `float`
        The GPS end time of the interval of interest.

    Examples
    --------
    >>> from dqsegdb2.api import resources_path
    >>> insert_history_path('G1', 'GEO-SCIENCE', 1)
    '/dq/G1/GEO-SCIENCE/1/insert_history'
    >>> insert_history_path('G1', 'GEO-SCIENCE', 1, s=1, e=2)
    '/dq/G1/GEO-SCIENCE/1/insert_history?s=1&e=2'
    """
    return f"{resources_path(ifo, flag, version)}/insert_history"


# -- /report API paths

def report_path():
    """Return the API path to quey for a data-quality report.

    Examples
    --------
    >>> from dqsegdb2.api import report_path
    >>> report_path()
    '/report'
    """
    return "/report"


def report_flags_path():
    """Return the API path to query for a report of all flags.

    Examples
    --------
    >>> from dqsegdb2.api import report_flags_path
    >>> report_flags_path()
    '/report/flags'
    """
    return f"{report_path()}/flags"


def report_coverage_path():
    """Return the API path to query for a coverage report of all flags.

    Examples
    --------
    >>> from dqsegdb2.api import report_coverage_path
    >>> report_coverage_path()
    '/report/coverage'
    """
    return f"{report_path()}/coverage"


def report_db_path():
    """Return the API path to query for a database statistics report.

    Examples
    --------
    >>> from dqsegdb2.api import report_db_path
    >>> report_db_path()
    '/report/db'
    """
    return f"{report_path()}/db"


def report_process_path():
    """Return the API path to query for a report of processing information.

    Examples
    --------
    >>> from dqsegdb2.api import report_process_path
    >>> report_process_path()
    '/report/process'
    """
    return f"{report_path()}/process"


@_query_params
def report_known_path(s=None, e=None):
    """Return the API path to query for a report of all flags
    in the known state.

    Examples
    --------
    >>> from dqsegdb2.api import report_known_path
    >>> report_known_path()
    '/report/known'
    >>> report_known_path(s=1, e=2)
    '/report/known?s=1&e=2'
    """
    return f"{report_path()}/known"


@_query_params
def report_active_path(s=None, e=None):
    """Return the API path to query for a report of all
    flags in the active state.

    Examples
    --------
    >>> from dqsegdb2.api import report_active_path
    >>> report_active_path()
    '/report/active'
    >>> report_active_path(s=1, e=2)
    '/report/active?s=1&e=2'
    """
    return f"{report_path()}/active"
