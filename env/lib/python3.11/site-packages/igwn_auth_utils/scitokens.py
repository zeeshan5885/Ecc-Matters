# -*- coding: utf-8 -*-
# Copyright 2021 Cardiff University
# Distributed under the terms of the BSD-3-Clause license

"""Utility functions for discovering valid scitokens
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

import os
from pathlib import Path
from urllib.parse import urlparse

from jwt import (
    InvalidAudienceError,
    InvalidTokenError,
)
from scitokens import (
    Enforcer,
    SciToken,
)
from scitokens.utils.errors import SciTokensException

from .error import IgwnAuthError

TOKEN_ERROR = (
    InvalidAudienceError,
    InvalidTokenError,
    SciTokensException,
)


WINDOWS = os.name == "nt"

# -- utilities --------------


def is_valid_token(
    token,
    audience,
    scope,
    issuer=None,
    timeleft=600,
):
    """Test whether a ``token`` is valid according to the given claims.

    Parameters
    ----------
    token : `scitokens.SciToken`, `str`
        The token object, or serialisation, to test

    audience : `str`, `list` or `str`
        The audience(s) to accept.

    scope : `str`
        A single scope to enforce.

    timeleft : `float`
        The amount of time remaining (in seconds, from the `exp` claim)
        to require.

    issuer : `str`
        The value of the `iss` claim to enforce.

    Returns
    -------
    valid : `bool`
        `True` if the input ``token`` matches the required claims,
        otherwise `False`.
    """
    # if given a serialised token, deserialise it now
    if isinstance(token, (str, bytes)):
        try:
            token = SciToken.deserialize(token)
        except (InvalidTokenError, SciTokensException):
            return False

    # construct the enforcer
    if issuer is None:  # borrow the issuer from the token itself
        issuer = token["iss"]
    enforcer = Enforcer(issuer, audience=audience)

    # add validator for timeleft
    def _validate_timeleft(value):
        exp = float(value)
        return exp >= enforcer._now + timeleft

    enforcer.add_validator("exp", _validate_timeleft)

    # if scope wasn't given, borrow one from the token to pass validation
    if scope is None:
        scope = token["scope"].split(" ", 1)[0]
    # parse scope as scheme:path
    try:
        authz, path = scope.split(":", 1)
    except ValueError:
        authz = scope
        path = None

    # test
    return enforcer.test(token, authz, path=path)


def target_audience(url, include_any=True):
    """Return the expected ``aud`` claim to authorize a request to ``url``.

    Parameters
    ----------
    url : `str`
        The URL that will be requested.

    include_any : `bool`, optional
        If `True`, include ``"ANY"`` in the return list of
        audiences, otherwise, don't.

    Returns
    -------
    audiences : `list` of `str`
        A `list` of audience values (`str`), either of length 1
        if ``include_any=False`, otherwise of length 2.

    Examples
    --------
    >>> default_audience(
    ...     "https://datafind.ligo.org:443/LDR/services/data/v1/gwf.json",
    ...     include_any=True,
    ... )
    ["https://datafind.ligo.org", "ANY"]
    >>> default_audience(
    ...     "segments.ligo.org",
    ...     include_any=False,
    ... )
    ["https://segments.ligo.org"]

    Hostnames given without a URL scheme are presumed to be HTTPS:

    >>> default_audience("datafind.ligo.org")
    ["https://datafind.ligo.org"]
    """
    if "//" not in url:  # always match a hostname, not a path
        url = f"//{url}"
    parsed = urlparse(url, scheme="https")
    aud = [f"{parsed.scheme}://{parsed.hostname}"]
    if include_any:
        aud.append("ANY")
    return aud


# -- I/O --------------------

def deserialize_token(raw, **kwargs):
    """Deserialize a token.

    Parameters
    ----------
    raw : `str`
        the raw serialised token content to deserialise

    kwargs
        all keyword arguments are passed on to
        :meth:`scitokens.SciToken.deserialize`

    Returns
    -------
    token : `scitokens.SciToken`
        the deserialised token

    See also
    --------
    scitokens.SciToken.deserialize
        for details of the deserialisation, and any valid keyword arguments
    """
    return SciToken.deserialize(raw.strip(), **kwargs)


def load_token_file(path, **kwargs):
    """Load a SciToken from a file path.

    Parameters
    ----------
    path : `str`
        the path to the scitokens file

    kwargs
        all keyword arguments are passed on to :func:`deserialize_token`

    Returns
    -------
    token : `scitokens.SciToken`
        the deserialised token

    Examples
    --------
    To load a token and validate a specific audience:

    >>> load_token('mytoken', audience="my.service.org")

    See also
    --------
    scitokens.SciToken.deserialize
        for details of the deserialisation, and any valid keyword arguments
    """
    with open(path, "r") as fobj:
        return deserialize_token(fobj.read(), **kwargs)


# -- discovery --------------

def find_token(
    audience,
    scope,
    issuer=None,
    timeleft=600,
    skip_errors=True,
    **kwargs,
):
    """Find and load a `SciToken` for the given ``audience`` and ``scope``.

    Parameters
    ----------
    audience : `str`
        the required audience (``aud``).

    scope : `str`
        the required scope (``scope``).

    issuer : `str`
        the value of the `iss` claim to enforce.

    timeleft : `int`
        minimum required time left until expiry (in seconds)
        for a token to be considered 'valid'

    skip_errors : `bool`, optional
        skip over errors encoutered when attempting to deserialise
        discovered tokens; this may be useful to skip over invalid
        or expired tokens that exist, for example, which is why it
        is the default behaviour.

    kwargs
        all keyword arguments are passed on to
        :meth:`scitokens.SciToken.deserialize`

    Returns
    -------
    token : `scitokens.SciToken`
        the first token that matches the requirements

    Raises
    ------
    ~igwn_auth_utils.IgwnAuthError
        if no valid token can be found

    See also
    --------
    scitokens.SciToken.deserialize
        for details of the deserialisation, and any valid keyword arguments
    """
    # preserve error from parsing tokens
    error = None

    # iterate over all of the tokens we can find for this audience
    for token in _find_tokens(audience=audience, **kwargs):
        # parsing a token yielded an exception, handle it here:
        if isinstance(token, Exception):
            error = error or token  # record (first) error for later
            if skip_errors:
                continue  # move on
            raise IgwnAuthError(str(error)) from error  # stop here and raise

        # if this token is valid, stop here and return it
        if is_valid_token(
            token,
            audience,
            scope,
            issuer=issuer,
            timeleft=timeleft,
        ):
            return token

    # if we didn't find any valid tokens:
    raise IgwnAuthError(
        "could not find a valid SciToken, "
        "please verify the audience and scope, "
        "or generate a new token and try again",
    ) from error


def _find_tokens(**deserialize_kwargs):
    """Yield all tokens that we can find

    This function will `yield` exceptions that are raised when
    attempting to parse a token that was actually found, so that
    they can be handled by the caller.
    """
    def _token_or_exception(func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except TOKEN_ERROR as exc:
            return exc

    # read token directly from 'SCITOKEN{_FILE}' variable
    for envvar, loader in (
        ('SCITOKEN', deserialize_token),
        ('SCITOKEN_FILE', load_token_file),
    ):
        if envvar in os.environ:
            yield _token_or_exception(
                loader,
                os.environ[envvar],
                **deserialize_kwargs,
            )

    # try and find a token from HTCondor
    for tokenfile in _find_condor_creds_token_paths():
        yield _token_or_exception(
            load_token_file,
            tokenfile,
            **deserialize_kwargs,
        )

    try:
        yield _token_or_exception(SciToken.discover, **deserialize_kwargs)
    except OSError:  # no token
        pass  # try something else
    except AttributeError as exc:
        # windows doesn't have geteuid, that's ok, otherwise panic
        if not WINDOWS or "geteuid" not in str(exc):
            raise


def _find_condor_creds_token_paths():
    """Find all token files in the condor creds directory
    """
    try:
        _condor_creds_dir = Path(os.environ["_CONDOR_CREDS"])
    except KeyError:
        return
    try:
        for f in _condor_creds_dir.iterdir():
            if f.suffix == ".use":
                yield f
    except FileNotFoundError:   # creds dir doesn't exist
        return


# -- HTTP request helper ----

def token_authorization_header(token, scheme="Bearer"):
    """Format an in-memory token for use in an HTTP Authorization Header.

    Parameters
    ----------
    token : `scitokens.SciToken`
        the token to format

    scheme : `str` optional
        the Authorization scheme to use

    Returns
    -------
    header_str : `str`
        formatted content for an `Authorization` header

    Notes
    -----
    See `RFC-6750 <https://datatracker.ietf.org/doc/html/rfc6750>`__
    for details on the ``Bearer`` Authorization token standard.
    """
    return "{} {}".format(
        scheme,
        token._serialized_token or token.serialize().decode("utf-8"),
    )
