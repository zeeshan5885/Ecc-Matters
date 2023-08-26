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

"""Query methods for DQSEGDB2.
"""

from functools import partial

from ligo import segments

from igwn_auth_utils.scitokens import target_audience as scitoken_audience

from . import api
from .requests import (
    Session,
    get_json,
)
from .utils import get_default_host


def _url(host, path_func, *args, **kwargs):
    """Construct the full URL for a query to ``host`` using an API path func.
    """
    path = path_func(*args, **kwargs)
    if host is None:
        host = get_default_host()
    return f"{host.rstrip('/')}/{path.lstrip('/')}"


def query_ifos(
    host=None,
    raw=False,
    **request_kwargs,
):
    """Query for all defined interferometers (IFOs).

    Parameters
    ----------
    host : `str`, optional
        The URL of the DQSegDB server; if `None`
        :func:`~dqsegdb2.utils.get_default_host` will be used to discover
        the default host.

    raw : `bool`, optional
        Return the full JSON response from the request.

    request_kwargs
        Other keyword arguments are passed to :func:`igwn_auth_utils.get`.

    Returns
    -------
    `set`
        If ``raw=False``: the set of all known IFO prefices.

    `dict`
        If ``raw=True``: the full JSON response is returned.

    Examples
    --------
    >>> from dqsegdb2.query import query_ifos
    >>> query_ifos()
    {'H1', 'V1', 'G1', 'K1', 'L1'}
    >>> query_ifos(raw=True)
    {"query_information": {
         "start": 0,
         "server_timestamp": 1234567890,
         "end": 0,
         "server_elapsed_query_time": "0.12345",
         "include": [],
         "uri": "/dq",
         "api_version": "1.2.34",
         "server": "segments"},
     "Ifos": ["G1", "H1", "K1", "L1", "V1"]}
    """
    url = _url(host, api.ifos_path)
    out = get_json(url, **request_kwargs)
    if raw:
        return out
    return set(out["Ifos"])


def query_names(
    ifo,
    host=None,
    raw=False,
    **request_kwargs,
):
    """Query for all defined flags for the given ``ifo``

    Parameters
    ----------
    ifo : `str`
        The interferometer prefix for which to query.

    host : `str`, optional
        The URL of the DQSegDB server; if `None`
        :func:`~dqsegdb2.utils.get_default_host` will be used to discover
        the default host.

    raw : `bool`, optional
        Return the full JSON response from the request.

    request_kwargs
        Other keyword arguments are passed to :func:`igwn_auth_utils.get`.

    Returns
    -------
    `set`
        If ``raw=False`` the set of all define flag names in the format
        ``{ifo}:{name}``.

    `dict`
        If ``raw=True``: the full JSON response is returned.

    Examples
    --------
    >>> from dqsegdb2.query import query_names
    >>> query_names('X1')
    {'G1:GEO-FLAG_1', 'G1:GEO-FLAG_2'}
    >>> query_names('X1', raw=True)
    {'results': ['GEO-FLAG_1', 'GEO-FLAG_2'], 'query_information': ...}
    """
    url = _url(host, api.flags_path, ifo)
    out = get_json(url, **request_kwargs)
    if raw:
        return out
    return {f'{ifo}:{name}' for name in out['results']}


def query_versions(flag, host=None, raw=False, **request_kwargs):
    """Query for defined versions for the given flag.

    Parameters
    ----------
    flag : `str`
        The name for which to query.

    host : `str`, optional
        The URL of the DQSegDB server; if `None`
        :func:`~dqsegdb2.utils.get_default_host` will be used to discover
        the default host.

    raw : `bool`, optional
        Return the full JSON response from the request.

    request_kwargs
        Other keyword arguments are passed to :func:`igwn_auth_utils.get`.

    Returns
    -------
    `list` of `int`
        If ``raw=False`` (default), the list of defined versions
        for the given flag.

    `dict`
        If ``raw=True``: the full JSON response is returned.

    Examples
    --------
    >>> from dqsegdb2.query import query_versions
    >>> query_versions('G1:GEO-SCIENCE')
    [1, 2, 3]
    >>> query_versions('G1:GEO-SCIENCE', raw=True)
    {'resource_type': 'version', 'version': [1, 2, 3],
     'query_information': ...}
    """
    ifo, name = flag.split(':', 1)
    url = _url(host, api.versions_path, ifo, name)
    out = get_json(url, **request_kwargs)
    if raw:
        return out
    return sorted(out['version'])


def query_segments(
    flag,
    start,
    end,
    host=None,
    coalesce=True,
    raw=False,
    **request_kwargs,
):
    """Query for segments for the given flag in a ``[start, stop)`` interval.

    Parameters
    ----------
    flag : `str`
        The name for which to query, see _Notes_ for information on how
        versionless-flags are queried.

    start : `float`
        The GPS start time.

    end : `float`
        The GPS end time.

    host : `str`, optional
        The URL of the DQSegDB server; if `None`
        :func:`~dqsegdb2.utils.get_default_host` will be used to discover
        the default host.

    coalesce : `bool`, optional
        If `True`, coalesce the segmentlists returned by the server,
        and restrict them to lie fully within the ``[start, end)``
        request segment, otherwise return the 'raw' result,
        default: `True`.
        This option is ignored if ``raw=True`` is given.

    raw : `bool`, optional
        Return the full JSON response from the request.
        If an explicit version is not given, the result will be a
        `list` of JSON responses, one for each discovered version.

    request_kwargs
        Other keyword arguments are passed to :func:`igwn_auth_utils.get`.

    Returns
    -------
    `dict`
        If ``raw=False`` (default): a `dict` with the following keys

        - ``'ifo'`` - the interferometer prefix (`str`)
        - ``'name'`` - the flag name (`str`)
        - ``'version'`` - the flag version (`int`)
        - ``'known'`` - the known segments (`~ligo.segments.segmentlist`)
        - ``'active'`` - the active segments (`~ligo.segments.segmentlist`)
        - ``'metadata'`` - a `dict` of flag information (`dict`)
        - ``'query_information'`` - a `dict` of query information (`dict`)

    `object`
        If ``raw=True`` is given, **and** the flag name includes an explicit
        version, the result will be the raw JSON response from the single
        request, otherwise a `list` of JSON responses will be returned.

    Notes
    -----
    If ``flag`` is given without a version (e.g. ``'X1:FLAG-NAME'``) or the
    version is given as ``'*'`` (e.g. ``'X1:FLAG-NAME:*'``) the result of
    the query will be the intersection of queries over all versions found
    in the database.
    In that case the ``'metadata'`` and ``'query_information'`` in the output
    will be preserved for the highest version number only (if ``raw=False``).

    Examples
    --------
    >>> from dqsegdb2.query import query_segments
    >>> query_segments('G1:GEO-SCIENCE:1', 1000000000, 1000001000)
    """
    request = segments.segmentlist([
        segments.segment(float(start), float(end)),
    ])

    try:
        ifo, name, version = flag.split(':', 2)
        versions = [int(version)]
    except ValueError:
        single_version = False
        if flag.endswith(':*'):  # allow use of wildcard version
            flag = flag.rsplit(':', 1)[0]
        ifo, name = flag.split(':', 1)
    else:
        single_version = True

    # what to ask for:
    include = {"known", "active"}
    if raw:
        include.add("metadata")

    # construct partial func to use for each versioned URL
    _format_url = partial(
        _url,
        host,
        api.resources_path,
        ifo,
        name,
        s=start,
        e=end,
        include=",".join(sorted(include)),
    )

    # set default audience
    request_kwargs.setdefault(
        "token_audience",
        scitoken_audience(host or get_default_host()),
    )

    # use Session to query for and then loop over versions (if needed)
    with Session(**request_kwargs) as sess:

        if not single_version:  # query for all versions
            versions = sorted(query_versions(flag, host=host, session=sess))

        if raw and not single_version:
            out = []
        else:
            # custom subset of information, with at least the following
            # keys (and types)
            out = dict(
                known=segments.segmentlist(),
                active=segments.segmentlist(),
                ifo=ifo,
                name=name,
                version=versions[0],
            )

        for version in versions:
            url = _format_url(version)
            result = get_json(url, session=sess)

            if raw and single_version:
                return result

            if raw:
                out.append(result)
                continue

            # if not raw, convert to segment objects
            # and coalesce if asked
            for key in ('active', 'known'):
                out[key].extend(segments.segmentlist(map(
                    segments.segment,
                    result.pop(key),
                )))
                if coalesce:
                    out[key] = out[key].coalesce() & request
            out.update(result)

        if len(versions) > 1 and not raw:  # unset the version if multiple
            out["version"] = None

    return out
