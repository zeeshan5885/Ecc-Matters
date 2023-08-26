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

"""User interface functions for GWDataFind.

These are all imported to the top-level namespace, so should be
referred to in usage as ``gwdatafind.<function>`` and not
``gwdatafind.ui.<function>``
"""

from functools import wraps
from re import compile as compile_regex
from urllib.parse import urlparse
from warnings import warn

import requests

from igwn_auth_utils.requests import get as _get

from ligo import segments

from . import api
from .utils import (
    file_segment,
    get_default_host,
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

__all__ = [
    "connect",
    "ping",
    "find_observatories",
    "find_types",
    "find_times",
    "find_latest",
    "find_url",
    "find_urls",
]


# -- deprecated methods -------------------------

def connect(host=None, port=None):  # pragma: no cover
    """Open a new connection to a Datafind server.

    This method will auto-select between HTTP and HTTPS based on port,
    and (for HTTPS) will automatically load the necessary X509 credentials
    using :func:`gwdatafind.utils.find_credential`.

    .. warning::

       This method is deprecated and will be removed in a future release

    Parameters
    ----------
    host : `str`, optional
        the name of the datafind server to connect to; if not given will be
        taken from the ``GWDATAFIND_SERVER`` environment variable.

    port : `int`, optional
        the port on the server to use, if not given it will be stripped from
        the ``host`` name.

    Returns
    -------
    connection : `gwdatafind.HTTPConnection` or `gwdatafind.HTTPSConnection`
        a newly opened connection
    """
    import ssl
    from .http import (HTTPConnection, HTTPSConnection)
    from .utils import find_credential

    warn(
        "this method is deprecated and will be removed in a future release",
        DeprecationWarning,
    )
    if host is None:
        host = get_default_host()
    if port is None:
        try:
            host, port = host.rsplit(':', 1)
        except ValueError:
            pass
        else:
            port = int(port)
    if port not in (None, 80):
        cert, key = find_credential()
        context = ssl.create_default_context()
        context.load_cert_chain(cert, key)
        return HTTPSConnection(host=host, port=port, context=context)
    return HTTPConnection(host=host, port=port)


# -- new methods --------------------------------

@wraps(_get)
def get(url, *args, **kwargs):
    if url.startswith("http://") and requests.__version__ < "2.15.0":
        # workaround https://github.com/psf/requests/issues/4025
        kwargs.setdefault("cert", False)
    scheme, netloc = urlparse(url)[:2]
    host = netloc.split(":", 1)[0]  # remove the port
    kwargs.setdefault("token_audience", list({
        f"{scheme}://{netloc}",
        f"{scheme}://{host}",
        "ANY",
    }))
    kwargs.setdefault("token_scope", "gwdatafind.read")
    return _get(url, *args, **kwargs)


def get_json(*args, **kwargs):
    """Perform a GET request and return JSON.

    Parameters
    ----------
    *args, **kwargs
        all keyword arguments are passed directly to
        :meth:`igwn_auth_utils.requests.get`

    Returns
    -------
    data : `object`
        the URL reponse parsed with :func:`json.loads`

    See also
    --------
    igwn_auth_utils.requests.get
        for information on how the request is performed
    """
    response = get(*args, **kwargs)
    response.raise_for_status()
    return response.json()


def _url(host, api_func, *args, **kwargs):
    """Construct the full URL for a query to ``host`` using an API function.

    Parameters
    ----------
    host : `str`, `None`
        the host to query, if `None` `~gwdatafind.utils.get_default_host()`
        will be used to discover the default host.

    api_func : `callable`
        the function from the :mod:`gwdatafind.api` module to use in
        constructing the URL path.

    *args, **kwargs
        any positional or keyword arguments are passed directly as
        ``api_func(*args, **kwargs)``

    Returns
    -------
    url : `str`
        a full URL including scheme, host, and path
    """
    path = api_func(*args, **kwargs)
    if host is None:
        host = get_default_host()

    # handle host declarations with no scheme (no https:// prefix)
    if "://" not in host:
        host = host.lstrip("/")
        if host.split("/", 1)[0].endswith(":80"):
            host = f"http://{host}"
        else:
            host = f"https://{host}"

    return "{host}/{path}".format(host=host.rstrip("/"), path=path)


def ping(host=None, session=None, **request_kw):
    """Ping the GWDataFind host to test for life.

    Parameters
    ----------
    host : `str`, optional
        the URL/name of the GWDataFind host to query; if not given
        :func:`~gwdatafind.utils.get_default_host` will be used to
        discover the default host.

    session : `requests.Session`, optional
        the connection session to use; if not given, a
        :class:`igwn_auth_utils.requests.Session` will be
        created on-the-fly

    token, token_audience, token_scope
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`scitokens` for more details.

    cert
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`x509` for more details.

    request_kw
        other keywords are passed to :func:`igwn_auth_utils.get`

    Raises
    ------
    requests.RequestException
        if the request fails for any reason
    """
    qurl = _url(host, api.ping_path)
    response = get(qurl, session=session, **request_kw)
    response.raise_for_status()


def find_observatories(match=None, host=None, session=None, **request_kw):
    """Query a GWDataFind host for observatories with available data.

    Parameters
    ----------
    match : `str`, `re.Pattern`, optional
        restrict returned observatories to those matching a
        regular expression.

    host : `str`, optional
        the URL/name of the GWDataFind host to query; if not given
        :func:`~gwdatafind.utils.get_default_host` will be used to
        discover the default host.

    session : `requests.Session`, optional
        the connection session to use; if not given, a
        :class:`igwn_auth_utils.requests.Session` will be
        created on-the-fly

    token, token_audience, token_scope
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`scitokens` for more details.

    cert
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`x509` for more details.

    request_kw
        other keywords are passed to :func:`igwn_auth_utils.get`

    Returns
    -------
    obs : `list` of `str`
        the list of known observatories prefices (and combinations)

    Raises
    ------
    requests.RequestsException
        if the request fails for any reason

    Examples
    --------
    >>> find_observatories(host="datafind.gw-openscience.org")
    ['L', 'V', 'H']
    >>> find_observatories(match="H", host="datafind.gw-openscience.org")
    ['H']
    """
    qurl = _url(host, api.find_observatories_path)
    sites = set(get_json(qurl, session=session, **request_kw))
    if match:
        match = compile_regex(match).search
        return list(filter(match, sites))
    return list(sites)


def find_types(site=None, match=None, host=None, session=None, **request_kw):
    """Query a GWDataFind host for dataset types.

    Parameters
    ----------
    site : `str`, optional
        single-character name of site to match; if not given
        types for all sites will be returned.

    match : `str`, `re.Pattern`, optional
        regular expression to match against known types

    host : `str`, optional
        the URL/name of the GWDataFind host to query; if not given
        :func:`~gwdatafind.utils.get_default_host` will be used to
        discover the default host.

    session : `requests.Session`, optional
        the connection session to use; if not given, a
        :class:`~igwn_auth_utils.Session` will be
        created on-the-fly

    token, token_audience, token_scope
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`scitokens` for more details.

    cert
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`x509` for more details.

    request_kw
        other keywords are passed to :func:`igwn_auth_utils.get`

    Returns
    -------
    types : `list` of `str`
        list of dataset types

    Raises
    ------
    requests.RequestsException
        if the request fails for any reason

    Examples
    --------
    >>> find_types(host="datafind.gw-openscience.org")
    ['H2_LOSC_4_V1', 'V1_GWOSC_O3a_16KHZ_R1', 'H1_LOSC_16_V1', 'L1_LOSC_4_V1', 'V1_GWOSC_O2_16KHZ_R1', 'V1_GWOSC_O3a_4KHZ_R1', 'L1_GWOSC_O3a_4KHZ_R1', 'L1_GWOSC_O2_16KHZ_R1', 'L1_GWOSC_O2_4KHZ_R1', 'V1_GWOSC_O2_4KHZ_R1', 'H1_LOSC_4_V1', 'H1_GWOSC_O3a_16KHZ_R1', 'H1_GWOSC_O2_16KHZ_R1', 'H1_GWOSC_O3a_4KHZ_R1', 'L1_GWOSC_O3a_16KHZ_R1', 'H1_GWOSC_O2_4KHZ_R1', 'L1_LOSC_16_V1']
    >>> find_types(site='V', host="datafind.gw-openscience.org")
    ['V1_GWOSC_O3a_4KHZ_R1', 'V1_GWOSC_O3a_16KHZ_R1', 'V1_GWOSC_O2_4KHZ_R1', 'V1_GWOSC_O2_16KHZ_R1']

    (accurate as of Nov 18 2021)
    """  # noqa: E501
    qurl = _url(host, api.find_types_path, site=site)
    types = set(get_json(qurl, session=session, **request_kw))
    if match:
        match = compile_regex(match).search
        return list(filter(match, types))
    return list(types)


def find_times(
    site,
    frametype,
    gpsstart=None,
    gpsend=None,
    host=None,
    session=None,
    **request_kw,
):
    """Query a GWDataFind host for times in which data are available.

    Parameters
    ----------
    site : `str`
        single-character name of site to match

    frametype : `str`
        name of dataset to match

    gpsstart : `int`, optional
        GPS start time of query

    gpsend : `int`, optional
        GPS end time of query

    match : `str`, `re.Pattern`, optional
        regular expression to match against known types

    host : `str`, optional
        the URL/name of the GWDataFind host to query; if not given
        :func:`~gwdatafind.utils.get_default_host` will be used to
        discover the default host.

    session : `requests.Session`, optional
        the connection session to use; if not given, a
        :class:`igwn_auth_utils.requests.Session` will be
        created on-the-fly

    token, token_audience, token_scope
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`scitokens` for more details.

    cert
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`x509` for more details.

    request_kw
        other keywords are passed to :func:`igwn_auth_utils.get`

    Returns
    -------
    segments : `segments.segmentlist`
        the list of ``[start, end)`` intervals during which data are
        available for the relevant dataset.

    Examples
    --------
    >>> find_times(
    ...     "V",
    ...     "V1_GWOSC_O3a_4KHZ_R1",
    ...     gpsstart=1238249472,
    ...     gpsend=1239429120,
    ...     host="datafind.gw-openscience.org",
    ... )
    [segment(1238249472, 1238417408), segment(1238421504, 1238605824), segment(1238609920, 1238827008), segment(1238839296, 1239429120)]

    Raises
    ------
    requests.RequestsException
        if the request fails for any reason
    """  # noqa: E501
    qurl = _url(host, api.find_times_path, site, frametype, gpsstart, gpsend)
    times = get_json(qurl, session=session, **request_kw)
    return segments.segmentlist(map(segments.segment, times))


def _get_urls(qurl, scheme=None, on_missing="ignore", **kwargs):
    urls = get_json(qurl, **kwargs)

    if scheme:
        urls = list(filter(lambda e: urlparse(e).scheme == scheme, urls))

    if urls or on_missing == "ignore":
        return urls

    # warn or error on empty result
    err = "no files found"
    if on_missing == "warn":
        warn(err)
        return urls
    raise RuntimeError(err)


def find_url(
    framefile,
    urltype="file",
    on_missing="error",
    host=None,
    session=None,
    **request_kw,
):
    """Query a GWDataFind host for the URL of a single filename.

    Parameters
    ----------
    framefile : `str`
        the name of the file to match; note that only the basename of
        the file is relevant.

    urltype : `str`, optional
        URL scheme to search for

    on_missing : `str`, optional
        what to do when the requested file isn't found, one of:

        - ``'error'``: raise a `RuntimeError`
        - ``'warn'``: print a warning but return an empty `list`
        - ``'ignore'``: return an empty `list` with no warnings

    host : `str`, optional
        the URL/name of the GWDataFind host to query; if not given
        :func:`~gwdatafind.utils.get_default_host` will be used to
        discover the default host.

    session : `requests.Session`, optional
        the connection session to use; if not given, a
        :class:`igwn_auth_utils.requests.Session` will be
        created on-the-fly

    token, token_audience, token_scope
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`scitokens` for more details.

    cert
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`x509` for more details.

    request_kw
        other keywords are passed to :func:`igwn_auth_utils.get`

    Returns
    -------
    urls : `list` of `str`
        a list of URLs for all instances of ``filename``

    Raises
    ------
    requests.RequestsException
        if the request fails for any reason

    RuntimeError
        if no matching URLs are found and ``on_missing="error"`` was given
    """
    qurl = _url(host, api.find_url_path, framefile)
    return _get_urls(
        qurl,
        scheme=urltype,
        on_missing=on_missing,
        session=session,
        **request_kw,
    )


def find_latest(
    site,
    frametype,
    urltype="file",
    on_missing="error",
    host=None,
    session=None,
    **request_kw,
):
    """Query a GWDataFind host for the latest file in a given dataset.

    Parameters
    ----------
    site : `str`
        single-character name of site to match

    frametype : `str`
        name of dataset to match

    urltype : `str`, optional
        URL scheme to search for

    on_missing : `str`, optional
        what to do when the requested file isn't found, one of:

        - ``'error'``: raise a `RuntimeError`
        - ``'warn'``: print a warning but return an empty `list`
        - ``'ignore'``: return an empty `list` with no warnings

    host : `str`, optional
        the URL/name of the GWDataFind host to query; if not given
        :func:`~gwdatafind.utils.get_default_host` will be used to
        discover the default host.

    session : `requests.Session`, optional
        the connection session to use; if not given, a
        :class:`igwn_auth_utils.requests.Session` will be
        created on-the-fly

    token, token_audience, token_scope
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`scitokens` for more details.

    cert
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`x509` for more details.

    request_kw
        other keywords are passed to :func:`igwn_auth_utils.get`

    Returns
    -------
    urls : `list` of `str`
        a list of URLs for the latest file found

    Raises
    ------
    requests.RequestsException
        if the request fails for any reason

    RuntimeError
        if no latest file is found and ``on_missing="error"`` was given

    Examples
    --------
    >>> find_latest('H', 'H1_GWOSC_O2_4KHZ_R1', urltype='file', host='datafind.gw-openscience.org'))
    ['file://localhost/cvmfs/gwosc.osgstorage.org/gwdata/O2/strain.4k/frame.v1/H1/1186988032/H-H1_GWOSC_O2_4KHZ_R1-1187733504-4096.gwf']
    """  # noqa: E501
    qurl = _url(host, api.find_latest_path, site, frametype, urltype=urltype)
    return _get_urls(qurl, on_missing=on_missing, **request_kw)


def find_urls(
    site,
    frametype,
    gpsstart,
    gpsend,
    match=None,
    urltype="file",
    on_gaps="warn",
    host=None,
    session=None,
    **request_kw,
):
    """Query a GWDataFind host for all URLs for a dataset in an interval.

    Parameters
    ----------
    site : `str`
        single-character name of site to match

    frametype : `str`
        name of dataset to match

    gpsstart : `int`, optional
        GPS start time of interval

    gpsend : `int`, optional
        GPS end time of interval

    match : `str`, `re.Pattern`, optional
        regular expression pattern to match URLs against

    urltype : `str`, optional
        URL scheme to search for

    on_gaps : `str`, optional
        what to do when the requested all or some of the GPS interval
        is not covereed by the dataset, one of:

        - ``'error'``: raise a `RuntimeError`
        - ``'warn'``: print a warning but return all available URLs
        - ``'ignore'``: return the list of URLs with no warnings

    host : `str`, optional
        the URL/name of the GWDataFind host to query; if not given
        :func:`~gwdatafind.utils.get_default_host` will be used to
        discover the default host.

    session : `requests.Session`, optional
        the connection session to use; if not given, a
        :class:`igwn_auth_utils.requests.Session` will be
        created on-the-fly

    token, token_audience, token_scope
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`scitokens` for more details.

    cert
        passed directly to :class:`~igwn_auth_utils.Session`, see
        :ref:`x509` for more details.

    request_kw
        other keywords are passed to :func:`igwn_auth_utils.get`

    Raises
    ------
    requests.RequestsException
        if the request fails for any reason

    RuntimeError
        if gaps in the dataset are found and ``on_gaps="error"`` was given
    """
    qurl = _url(
        host,
        api.find_urls_path,
        site,
        frametype,
        gpsstart,
        gpsend,
        urltype=urltype,
        match=match,
    )
    urls = _get_urls(qurl, session=session, **request_kw)

    # ignore missing data
    if on_gaps == "ignore":
        return urls

    # handle missing data
    span = segments.segment(gpsstart, gpsend)
    seglist = segments.segmentlist(map(file_segment, urls)).coalesce()
    missing = (segments.segmentlist([span]) - seglist).coalesce()
    if not missing:  # no gaps
        return urls

    # warn or error on missing
    msg = "Missing segments: \n%s" % "\n".join(map(str, missing))
    if on_gaps == "warn":
        warn(msg)
        return urls
    raise RuntimeError(msg)
