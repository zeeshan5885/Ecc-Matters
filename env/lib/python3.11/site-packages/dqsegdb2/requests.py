# -*- coding: utf-8 -*-
# DQSEGDB2
# Copyright (C) 2022 Cardiff University
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

"""Request interface for dqsegdb2.
"""

from requests import __version__ as requests_version

from igwn_auth_utils.requests import (
    Session as _Session,
    get as _get,
)

DEFAULT_TOKEN_SCOPE = "dqsegdb.read"  # noqa: S105


class Session(_Session):
    def __init__(self, **kwargs):
        kwargs.setdefault("token_scope", DEFAULT_TOKEN_SCOPE)
        super().__init__(**kwargs)


def get(url, *args, **kwargs):
    """Send an HTTP GET request to a DQSegDB URL with IGWN Auth attached.

    This thin wrapper just sets the correct default `token_scope`
    argument.

    See also
    --------
    igwn_auth_utils.get
        For documentation of all arguments and keywords
    """
    # if given a session, use it without setting any parameters
    if kwargs.get("session"):
        return _get(url, *args, **kwargs)

    # otherwise attempt to correctly initialise auth
    if url.startswith("http://") and requests_version < "2.15.0":
        # workaround https://github.com/psf/requests/issues/4025
        kwargs.setdefault("cert", False)
    kwargs.setdefault("token_scope", DEFAULT_TOKEN_SCOPE)
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
