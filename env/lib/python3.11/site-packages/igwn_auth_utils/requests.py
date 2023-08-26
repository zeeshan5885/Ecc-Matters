# -*- coding: utf-8 -*-
# Copyright 2021-2023 Cardiff University
# Distributed under the terms of the BSD-3-Clause license

"""Python Requests interface with IGWN authentication.

This is heavily inspired by Leo Singer's excellent
:mod:`requests_gracedb` package.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Leo Singer <leo.singer@ligo.org>"

import sys
from functools import wraps
from unittest import mock

import requests
from requests.auth import AuthBase as _AuthBase
from requests import utils as requests_utils

from scitokens import SciToken

from .error import IgwnAuthError
from .scitokens import (
    find_token as find_scitoken,
    target_audience as scitoken_audience,
    token_authorization_header as scitoken_authorization_header,
)
from .x509 import (
    find_credentials as find_x509_credentials,
)

# handle https://github.com/psf/requests/issues/4025
# NOTE: 2.14.1 is the base version on EPEL7
REQUESTS_CERT_HTTP_BUG = '2.14.1' <= requests.__version__ < '2.15.0'


# -- Auth utilities -------------------

def _find_cred(func, *args, error=True, **kwargs):
    """Find a credential and maybe ignore an `~igwn_auth_utils.IgwnAuthError`

    This is an internal utility for the `SessionAuthMixin._init_auth`
    method which shouldn't necessary fail if it doesn't
    find a credential of any one type, but should just move on to the
    next option.
    """
    try:
        return func(*args, **kwargs)
    except IgwnAuthError:
        if error:
            raise
        return


@wraps(requests_utils.get_netrc_auth)
def get_netrc_auth(url, raise_errors=False):
    """Wrapper around `requests.utils.get_netrc_auth` to use `safe_netrc`.
    """
    import safe_netrc
    with mock.patch.dict(sys.modules, {"netrc": safe_netrc}):
        return requests_utils.get_netrc_auth(url, raise_errors=raise_errors)


class HTTPSciTokenAuth(_AuthBase):
    """Auth handler for SciTokens.
    """
    def __init__(
        self,
        token=None,
        audience=None,
        scope=None,
        issuer=None,
    ):
        self.token = token
        self.audience = audience
        self.scope = scope
        self.issuer = issuer

    def __eq__(self, other):
        return all([
            self.token == getattr(other, "token", None),
            self.audience == getattr(other, "audience", None),
            self.scope == getattr(other, "scope", None),
            self.issuer == getattr(other, "issuer", None),
        ])

    def __ne__(self, other):
        return not self == other

    @staticmethod
    def _auth_header_str(token):
        """Serialise a `scitokens.SciToken` and format an Authorization header.

        Parameters
        ----------
        token : `scitokens.SciToken`, `str`, `bytes`
            the token to serialize, or an already serialized representation
        """
        if isinstance(token, (str, bytes)):  # already serialised
            return f"Bearer {token}"
        return scitoken_authorization_header(token)

    def find_token(
        self,
        url=None,
        error=True,
    ):
        """Find a bearer token for authorization.

        Parameters
        ----------
        url : `str`
            The URL that will be queried.

        error : `bool`
            If `True`, `raise` exceptions, otherwise return `None`.
        """
        audience = self.audience
        if audience is None and url is not None:
            audience = scitoken_audience(url, include_any=False)
        return _find_cred(
            find_scitoken,
            audience,
            self.scope,
            issuer=self.issuer,
            error=error,
        )

    def __call__(self, r):
        """Augment the `Request` ``r`` with an ``Authorization`` header.
        """
        token = self.token
        if token in (None, True):
            token = self.find_token(
                url=getattr(r, "url", None),  # allow r as Session
                error=bool(token),
            )

        # if we ended up with a header, store it in the request.
        if token:
            r.headers["Authorization"] = self._auth_header_str(token)

        return r


def _prepare_auth(
    url=None,
    auth=None,
    cert=None,
    token=None,
    token_audience=None,
    token_scope=None,
    token_issuer=None,
    force_noauth=False,
    fail_if_noauth=False,
    session=None,
):
    """Prepare authorisation for a session or request.
    """
    # workaround https://github.com/psf/requests/issues/4025
    if (
        cert is None
        and str(url).startswith("http://")
        and REQUESTS_CERT_HTTP_BUG
    ):
        cert = False

    # merge settings from the session
    if session:
        if cert is None:
            cert = session.cert
        if token is None and session.auth is None:
            token = False
        if isinstance(session.auth, HTTPSciTokenAuth):
            if token is None:
                token = session.auth.token
            if token_audience is None:
                token_audience = session.auth.audience
            if token_scope is None:
                token_scope = session.auth.scope
            if token_issuer is None:
                token_issuer = session.auth.issuer

    # handle options
    if force_noauth and fail_if_noauth:
        raise ValueError(
            "cannot select both force_noauth and fail_if_noauth",
        )
    if force_noauth:
        return None, False

    # cert auth (always attach if we can)
    if cert in (None, True):  # not disabled and not given explicitly
        cert = _find_cred(find_x509_credentials, error=cert is True)

    # use existing auth object
    if auth is not None:
        pass

    # -- bearer token (scitoken)

    elif token is not False:
        # get the default audience from the URL
        if token_audience is None and url is not None:
            token_audience = scitoken_audience(url, include_any=False)
        auth = HTTPSciTokenAuth(
            token=token,
            audience=token_audience,
            scope=token_scope,
            issuer=token_issuer,
        )

    # -- basic auth (netrc)

    elif (
        # we know where the requests are heading
        url is not None
        # and basic auth not disabled
        and auth in (None, True)
    ):
        auth = get_netrc_auth(url, raise_errors=False)

    # -- handle fail_if_noauth

    # if no auth was found, and we need it, fail here
    if fail_if_noauth and cert in (None, False) and (
        auth is None
        or (
            isinstance(auth, HTTPSciTokenAuth)
            and not auth.find_token(url=url, error=False)
        )
    ):
        raise IgwnAuthError("no valid authorisation credentials found")

    return auth, cert


# -- Session handling -----------------

_auth_session_parameters = """
    Discovery/configuration of authorisation/authentication methods
    is attempted in the following order:

    1.  if ``force_noauth=True`` is given, no auth is configured;

    2.  for SciTokens:

        1.  if a bearer token is provided via the ``token`` keyword argument,
            then use that, or

        2.  look for a bearer token by passing the ``token_audience``
            and ``token_scope`` keyword parameters to
            :func:`igwn_auth_utils.find_scitokens`;

    3.  for X.509 credentials:

        1.  if an X.509 credential path is provided via the ``cert`` keyword
            argument, then use that, or

        2.  look for an X.509 credential using
            :func:`igwn_auth_utils.find_x509_credential`

    4.  for basic auth (username/password):

        1.  if ``auth`` keyword is provided, then use that, or

        2.  read the netrc file located at :file:`~/.netrc`, or at the path
            stored in the :envvar:`$NETRC` environment variable, and look
            for a username and password matching the hostname given in the
            ``url`` keyword argument;

    5.  if none of the above yield a credential, and ``fail_if_noauth=True``
        was provided, raise a `ValueError`.

    Steps 2 and 3 are all tried independently, with all valid credentials
    (one per type) configured for the session.
    Only when SciTokens are disabled (``token=False``), will step 4 will be
    tried to configure basic username/password auth.
    It is up to the request receiver to handle the multiple credential
    types and prioritise between them.

    Parameters
    ----------
    token : `scitokens.SciToken`, `str`, `bool`, optional
        Bearer token (scitoken) input, one of

        - a bearer token (`scitokens.SciToken`),
        - a serialised token (`str`, `bytes`),
        - `False`: disable using tokens completely
        - `True`: discover a valid token via
          :func:`igwn_auth_utils.find_scitoken` and
          error if something goes wrong
        - `None`: try and discover a valid token, but
          try something else if that fails

    token_audience : `str`, list` of `str`
        The value(s) of the audience (``aud``) claim to pass to
        :func:`igwn_auth_utils.find_scitoken` when discovering
        available tokens.

    token_scope : `str`
        The value(s) of the ``scope`` to pass to
        :func:`igwn_auth_utils.find_scitoken` when discovering
        available tokens.

    token_issuer : `str`
        The value of the issuer (``iss``) claim to pass to
        :func:`igwn_auth_utils.find_scitoken` when discovering
        available tokens.

    cert : `str`, `tuple`, `bool`, optional
        X.509 credential input, one of

        - path to a PEM-format certificate file,
        - a ``(cert, key)`` `tuple`,
        - `False`: disable using X.509 completely
        - `True`: discover a valid cert via
          :func:`igwn_auth_utils.find_x509_credentials` and
          error if something goes wrong
        - `None`: try and discover a valid cert, but
          try something else if that fails

    auth :  `tuple`, `object`, optional
        ``(username, password)`` `tuple` or other authentication/authorization
        object to attach to a `~requests.Request`.
        By default a new :class:`HTTPSciTokenAuth` handler will be attached
        to configure ``Authorization`` headers for each request.

    url : `str`, optional
        the URL/host that will be queried within this session; this is used
        to set the default ``token_audience`` and to access credentials
        via :mod:`safe_netrc`.

    force_noauth : `bool`, optional
        Disable the use of any authorisation credentials (mainly for testing).

    fail_if_noauth : `bool`, optional
        Raise a `~igwn_auth_utils.IgwnAuthError` if no authorisation
        credentials are presented or discovered.

    raise_for_status : `bool`, optional
        If `True` (default), automatically call
        :meth:`~requests.Response.raise_for_status` after receiving
        any response.

    Raises
    ------
    ~igwn_auth_utils.IgwnAuthError
        If ``cert=True`` or ``token=True`` is given and the relevant
        credential was not actually discovered, or
        if ``fail_if_noauth=True`` is given and no authorisation
        token/credentials of any valid type are presented or discovered.

    See also
    --------
    requests.Session
        for details of the standard options

    igwn_auth_utils.find_scitoken
        for details of the SciToken discovery

    igwn_auth_utils.find_x509_credentials
        for details of the X.509 credential discovery
    """.strip()


def _hook_raise_for_status(response, *args, **kwargs):
    """Response hook to raise exception for any HTTP error (status >= 400)

    Reproduced (with permission) from :mod:`requests_gracedb.errors`,
    authored by Leo Singer.
    """
    return response.raise_for_status()


class SessionErrorMixin:
    """A mixin for :class:`requests.Session` to raise exceptions for HTTP
    errors.

    Reproduced (with permission) from :mod:`requests_gracedb.errors`,
    authored by Leo Singer.
    """
    def __init__(self, *args, **kwargs):
        raise_for_status = kwargs.pop("raise_for_status", True)
        super().__init__(*args, **kwargs)
        if raise_for_status:
            self.hooks.setdefault("response", []).append(
                _hook_raise_for_status,
            )


class SessionAuthMixin:
    """Mixin for :class:`requests.Session` to add support for IGWN auth.

    By default this mixin will automatically attempt to discover/configure
    a bearer token (scitoken) or an X.509 credential, with options to
    require/disable either of those, or all authentication entirely.

    {parameters}
    """
    def __init__(
        self,
        token=None,
        token_audience=None,
        token_scope=None,
        token_issuer=None,
        cert=None,
        auth=None,
        url=None,
        force_noauth=False,
        fail_if_noauth=False,
        **kwargs,
    ):
        # initialise session
        super().__init__(**kwargs)

        # initialise auth handler and cert
        self._init_auth(
            url=url,
            auth=auth,
            cert=cert,
            token=token,
            token_audience=token_audience,
            token_scope=token_scope,
            token_issuer=token_issuer,
            force_noauth=force_noauth,
            fail_if_noauth=fail_if_noauth,
        )

    def _init_auth(self, url=None, token=None, **kwargs):
        """Initialise the auth handler for this `Session`.
        """
        # find creds if we can
        self.auth, self.cert = _prepare_auth(url=url, token=token, **kwargs)

        # if we were given a token, use it now
        if isinstance(token, (SciToken, str, bytes)):
            self.auth(self)

    @property
    def token(self):
        """The token object that will be used in authorised requests.

        If the :attr:`Session.auth` property isn't an instance of
        `HTTPSciTokenAuth` with a token attached, this returns `None`.
        """
        token = getattr(self.auth, "token", None)
        if isinstance(token, SciToken):
            return token


class Session(
    SessionAuthMixin,
    SessionErrorMixin,
    requests.Session,
):
    """`requests.Session` class with default IGWN authorization handling.

    {parameters}

    Examples
    --------
    To use the default authorisation discovery:

    >>> from igwn_auth_utils.requests import Session
    >>> with Session() as sess:
    ...     sess.get("https://science.example.com/api/important/data")

    To explicitly pass a specific :class:`~scitokens.SciToken` as the token:

    >>> with Session(token=mytoken) as sess:
    ...     sess.get("https://science.example.com/api/important/data")

    To explicitly *require* that a token is discovered, and *disable*
    any X.509 discovery:

    >>> with Session(token=True, x509=False) as sess:
    ...     sess.get("https://science.example.com/api/important/data")

    To use default authorisation discovery, but fail if no credentials
    are discovered:

    >>> with Session(fail_if_noauth=True) as sess:
    ...     sess.get("https://science.example.com/api/important/data")

    To disable all authorisation discovery:

    >>> with Session(force_noauth=True) as sess:
    ...     sess.get("https://science.example.com/api/important/data")
    """
    __attrs__ = requests.Session.__attrs__ = [
        "token",
    ]

    @wraps(requests.Session.request)
    def request(
        self,
        method,
        url,
        *args,
        token=None,
        token_audience=None,
        token_scope=None,
        token_issuer=None,
        cert=None,
        auth=None,
        force_noauth=False,
        fail_if_noauth=False,
        **kwargs,
    ):
        # handle request-specific auth
        auth, cert = _prepare_auth(
            url=url,
            auth=auth,
            cert=cert,
            token=token,
            token_audience=token_audience,
            token_scope=token_scope,
            token_issuer=token_issuer,
            force_noauth=force_noauth,
            fail_if_noauth=fail_if_noauth,
            session=self,
        )

        # continue with request
        return super().request(
            method,
            url,
            *args,
            auth=auth,
            cert=cert,
            **kwargs,
        )


# update the docstrings to include the same parameter info
for _obj in (Session, SessionAuthMixin):
    _obj.__doc__ = _obj.__doc__.format(parameters=_auth_session_parameters)


# -- standalone request handling ------

def request(method, url, *args, session=None, **kwargs):
    """Send a request of the specific method to the specified URL.

    Parameters
    ----------
    method : `str`
        The method to use.

    url : `str`,
        The URL to request.

    session : `requests.Session`, optional
        The connection session to use, if not given one will be
        created on-the-fly.

    args, kwargs
        All other keyword arguments are passed directly to
        `requests.Session.request`

    Returns
    -------
    resp : `requests.Response`
        the response object

    See also
    --------
    igwn_auth_utils.requests.Session.request
        for information on how the request is performed
    """
    # user's session
    if session:
        return session.request(method, url, *args, **kwargs)

    # new session
    with Session(force_noauth=kwargs.get("force_noauth", False)) as session:
        return session.request(method, url, *args, **kwargs)


_request_wrapper_doc = """
    Send an HTTP {METHOD} request to the specified URL with IGWN Auth attached.

    Parameters
    ----------
    url : `str`
        The URL to request.

    session : `requests.Session`, optional
        The connection session to use, if not given one will be
        created on-the-fly.

    args, kwargs
        All other keyword arguments are passed directly to
        :meth:`requests.Session.{method}`

    Returns
    -------
    resp : `requests.Response`
        the response object

    See also
    --------
    requests.Session.{method}
        for information on how the request is performed
""".strip()


def _request_wrapper_factory(method):
    """Factor function to wrap a :mod:`requests` HTTP method to use
    our request function.
    """
    def _request_wrapper(url, *args, session=None, **kwargs):
        return request(method, url, *args, session=session, **kwargs)

    _request_wrapper.__doc__ = _request_wrapper_doc.format(
        method=method,
        METHOD=method.upper(),
    )
    return _request_wrapper


# request methods
delete = _request_wrapper_factory("delete")
get = _request_wrapper_factory("get")
head = _request_wrapper_factory("head")
patch = _request_wrapper_factory("patch")
post = _request_wrapper_factory("post")
put = _request_wrapper_factory("put")
