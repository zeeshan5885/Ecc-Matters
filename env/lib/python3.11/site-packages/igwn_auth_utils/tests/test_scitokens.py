# -*- coding: utf-8 -*-
# Copyright 2021 Cardiff University
# Distributed under the terms of the BSD-3-Clause license

"""Tests for :mod:`igwn_auth_utils.scitokens`.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

import os
import time
from functools import partial
from unittest import mock

from scitokens import (
    __version__ as scitokens_version,
    SciToken,
)
from scitokens.scitokens import InvalidPathError

import pytest

from .. import scitokens as igwn_scitokens
from ..error import IgwnAuthError

ISSUER = "local"
_SCOPE_PATH = "/igwn_auth_utils"
READ_AUDIENCE = "igwn_auth_utils"
READ_SCOPE = "read:{}".format(_SCOPE_PATH)
WRITE_AUDIENCE = "igwn_auth_utils2"
WRITE_SCOPE = "write:{}".format(_SCOPE_PATH)


def _os_error(*args, **kwargs):
    raise OSError


def _create_token(
    key=None,
    iss=ISSUER,
    aud=READ_AUDIENCE,
    scope=READ_SCOPE,
    **kwargs
):
    """Create a token
    """
    if key:
        from scitokens.utils.keycache import KeyCache
        keycache = KeyCache.getinstance()
        keycache.addkeyinfo(iss, "test_key", key.public_key())
    now = int(time.time())
    token = SciToken(key=key, key_id="test_key")
    token.update_claims({
        "iat": now,
        "nbf": now,
        "exp": now + 86400,
        "iss": iss,
        "aud": aud,
        "scope": scope,
    })
    token.update_claims(kwargs)
    return token


def _write_token(token, path):
    with open(path, "wb") as file:
        file.write(token.serialize(lifetime=86400))


@pytest.fixture
def rtoken(private_key):
    return _create_token(
        key=private_key,
        scope=READ_SCOPE,
    )


@pytest.fixture
def wtoken(private_key):
    return _create_token(
        key=private_key,
        aud=WRITE_AUDIENCE,
        scope=WRITE_SCOPE,
    )


@pytest.fixture
def rtoken_path(rtoken, tmp_path):
    token_path = tmp_path / "token.use"
    _write_token(rtoken, token_path)
    return token_path


@pytest.fixture
def condor_creds_path(rtoken, wtoken, tmp_path):
    for token, name in (
        (rtoken, "read.use"),
        (wtoken, "write.use"),
    ):
        _write_token(token, tmp_path / name)
    return tmp_path


# -- test utilities -----------------------------

def assert_tokens_equal(a, b):
    _assert_claims_equal(a, b)
    _assert_claims_equal(b, a)


_SKIP_ASSERT_CLAIMS = {
    "exp",
    "iat",
    "nbf",
}


def _assert_claims_equal(a, b):
    for claim, value in a.claims():
        if claim in _SKIP_ASSERT_CLAIMS:
            continue
        assert b.get(claim) == value


# -- tests --------------------------------------

@pytest.mark.parametrize(("scope", "validity"), (
    (READ_SCOPE, True),  # read scope matches token
    (WRITE_SCOPE, False),  # write scope doesn't
    (None, True),  # accept any scope
))
def test_is_valid_token(rtoken, scope, validity):
    assert igwn_scitokens.is_valid_token(
        rtoken,
        READ_AUDIENCE,
        scope,
    ) is validity


def test_is_valid_token_invalid_path(rtoken):
    with pytest.raises(InvalidPathError):
        igwn_scitokens.is_valid_token(rtoken, READ_AUDIENCE, "read")


def test_is_valid_token_serialized(rtoken, public_pem):
    _deserialize_local = partial(
        SciToken.deserialize,
        audience=READ_AUDIENCE,
        insecure=True,
        public_key=public_pem,
    )
    with mock.patch("scitokens.SciToken.deserialize", _deserialize_local):
        assert igwn_scitokens.is_valid_token(
            rtoken.serialize(lifetime=86400),
            READ_AUDIENCE,
            READ_SCOPE,
        )


def test_is_valid_token_serialized_false():
    assert igwn_scitokens.is_valid_token(
        "bad",
        READ_AUDIENCE,
        READ_SCOPE,
    ) is False


@pytest.mark.parametrize(("issuer", "result"), [
    (None, True),
    ("local", True),
    ("other", False),
])
def test_is_valid_token_issuer(rtoken, issuer, result):
    """Test that `igwn_auth_utils.scitokens.is_valid_token` enforces
    ``issuer`` properly.
    """
    assert igwn_scitokens.is_valid_token(
        rtoken,
        READ_AUDIENCE,
        None,
        issuer=issuer,
    ) is result


@pytest.mark.parametrize("include_any", (False, True))
@pytest.mark.parametrize(("url", "aud"), (
    # basic
    ("https://example.com/data", ["https://example.com"]),
    # no scheme
    ("example.com", ["https://example.com"]),
    # port
    ("https://example.com:443/data/test", ["https://example.com"]),
    # HTTP
    ("http://example.com:443/data/test", ["http://example.com"]),
))
def test_target_audience(url, aud, include_any):
    if include_any:
        aud += ["ANY"]
    assert igwn_scitokens.target_audience(url, include_any=include_any) == aud


def test_load_token_file(rtoken_path, rtoken, public_pem):
    assert_tokens_equal(
        igwn_scitokens.load_token_file(
            rtoken_path,
            audience=READ_AUDIENCE,
            insecure=True,
            public_key=public_pem,
        ),
        rtoken,
    )


@mock.patch.dict("os.environ")
@pytest.mark.parametrize("envname", (
    "SCITOKEN",
    "BEARER_TOKEN",
))
def test_find_token_env_scitoken(rtoken, public_pem, envname):
    os.environ[envname] = rtoken.serialize(lifetime=86400).decode("utf-8")
    assert_tokens_equal(
        igwn_scitokens.find_token(
            audience=READ_AUDIENCE,
            scope=READ_SCOPE,
            insecure=True,
            public_key=public_pem,
        ),
        rtoken,
    )


@mock.patch.dict("os.environ")
@pytest.mark.parametrize("envname", (
    "SCITOKEN_FILE",
    "BEARER_TOKEN_FILE",
))
def test_find_token_env_scitoken_file(
    rtoken,
    wtoken,
    rtoken_path,
    public_pem,
    envname,
):
    # set the wrong token as SCITOKEN
    os.environ["SCITOKEN"] = wtoken.serialize(lifetime=86400).decode("utf-8")
    # and the correct token as SCITOKEN_FILE
    os.environ[envname] = str(rtoken_path)
    # and make sure we get the correct token back
    assert_tokens_equal(
        igwn_scitokens.find_token(
            audience=READ_AUDIENCE,
            scope=READ_SCOPE,
            insecure=True,
            public_key=public_pem,
            skip_errors=True,
        ),
        rtoken,
    )


@mock.patch.dict("os.environ")
def test_find_token_condor_creds(
    rtoken,
    wtoken,
    public_pem,
    condor_creds_path,
):
    os.environ["_CONDOR_CREDS"] = str(condor_creds_path)
    for token, aud, scope in (
        (rtoken, READ_AUDIENCE, READ_SCOPE),
        (wtoken, WRITE_AUDIENCE, WRITE_SCOPE),
    ):
        assert_tokens_equal(
            igwn_scitokens.find_token(
                audience=aud,
                scope=scope,
                insecure=True,
                public_key=public_pem,
                skip_errors=True,
            ),
            token,
        )


@pytest.mark.parametrize(("audience", "msg"), [
    (READ_AUDIENCE, "could not find a valid SciToken"),
    (WRITE_AUDIENCE, (
        "could not find a valid SciToken" if scitokens_version >= "1.7.3"
        else "(Invalid audience|Audience doesn't match)"
    )),
])
@mock.patch.dict("os.environ")
# make sure a real token doesn't get in the way
@mock.patch("igwn_auth_utils.scitokens.SciToken.discover", _os_error)
def test_find_token_error(rtoken, public_pem, audience, msg):
    # token with the wrong claims
    os.environ["SCITOKEN"] = rtoken.serialize().decode("utf-8")
    # check that we get an error
    with pytest.raises(
        IgwnAuthError,
        match=msg,
    ):
        igwn_scitokens.find_token(
            audience,
            WRITE_SCOPE,
            insecure=True,
            public_key=public_pem,
            skip_errors=False,
        )


@mock.patch.dict("os.environ")
@pytest.mark.parametrize(("skip_errors", "message"), (
    (False, "Issuer is not over HTTPS"),
    (True, "could not find a valid SciToken"),
))
def test_find_token_skip_errors(rtoken, skip_errors, message):
    """Check that the ``skip_errors`` keyword for `find_token()` works
    """
    # configure a valid token (wrong claims) **HOWEVER**
    # don't add the necessary keyword arguments that would
    # enable `deserialize` to actually work
    # (see `test_find_token_error` above for that)
    os.environ["SCITOKEN"] = rtoken.serialize().decode("utf-8")

    # check that we get the normal error
    with pytest.raises(
        IgwnAuthError,
        match=message,
    ):
        igwn_scitokens.find_token(
            READ_AUDIENCE,
            READ_SCOPE,
            skip_errors=skip_errors,
        )


@mock.patch.dict("os.environ")
def test_find_condor_creds_no_env(tmp_path):
    """Check that `_find_condor_creds_token_paths()` handles missing
    ``_CONDOR_CREDS`` environment variable
    """
    assert not list(igwn_scitokens._find_condor_creds_token_paths())


@mock.patch.dict("os.environ")
def test_find_condor_creds_dir_missing(tmp_path):
    """Check that `_find_condor_creds_token_paths()` handles
    ``_CONDOR_CREDS`` pointing at a directory that doesn't exist
    """
    os.environ["_CONDOR_CREDS"] = str(tmp_path / "_condor_creds")
    assert not list(igwn_scitokens._find_condor_creds_token_paths())


@mock.patch.dict("os.environ")
def test_find_condor_creds_dir_empty(tmp_path):
    """Check that `_find_condor_creds_token_paths()` handles
    ``_CONDOR_CREDS`` pointing at an empty directory
    """
    os.environ["_CONDOR_CREDS"] = str(tmp_path)
    assert not list(igwn_scitokens._find_condor_creds_token_paths())


def test_token_authorization_header(rtoken):
    """Check that `token_authorization_header` works
    """
    expected = "Bearer {}".format(rtoken.serialize().decode("utf-8"))

    # reset _serialized_token attribute
    rtoken._serialized_token = None

    # do it once to check that the call to token.serialize() works
    a = igwn_scitokens.token_authorization_header(rtoken)
    assert a == expected

    # do it again to check that the use of _serialized_token attr works
    assert igwn_scitokens.token_authorization_header(rtoken) == a
