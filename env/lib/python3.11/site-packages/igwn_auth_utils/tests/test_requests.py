# -*- coding: utf-8 -*-
# Copyright 2021-2022 Cardiff University
# Distributed under the terms of the BSD-3-Clause license

"""Tests for :mod:`igwn_auth_utils.requests`.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Leo Singer <leo.singer@ligo.org>"

import os
import stat
from netrc import NetrcParseError
from unittest import mock
from urllib.parse import urlencode

import pytest

from requests import (
    __version__ as requests_version,
    RequestException,
)

from .. import requests as igwn_requests
from ..error import IgwnAuthError
from .test_scitokens import rtoken  # noqa: F401

SKIP_REQUESTS_NETRC = pytest.mark.skipif(
    requests_version < "2.25.0",
    reason=f"requests {requests_version} doesn't respect NETRC env",
)


# -- utilities ------------------------

def _empty(*args, **kwargs):
    return []


def _igwnerror(*args, **kwargs):
    raise IgwnAuthError("error")


def mock_no_scitoken():
    return mock.patch(
        "igwn_auth_utils.scitokens._find_tokens",
        _empty,
    )


def mock_no_x509():
    return mock.patch(
        "igwn_auth_utils.requests.find_x509_credentials",
        _igwnerror,
    )


@pytest.fixture
def netrc(tmp_path):
    netrc = tmp_path / "netrc"
    netrc.write_text(
        "machine example.org login albert.einstein password super-secret",
    )
    netrc.chmod(stat.S_IRWXU)
    return netrc


def has_auth(session):
    return bool(
        session.auth
        or session.cert
        or "Authorization" in session.headers
    )


class MockRequest(mock.MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.headers = {}


# -- get_netrc_auth -------------------

@mock.patch.dict(os.environ)
@SKIP_REQUESTS_NETRC
def test_get_netrc_auth(netrc):
    os.environ["NETRC"] = str(netrc)
    assert igwn_requests.get_netrc_auth("https://example.org/path") == (
        "albert.einstein",
        "super-secret",
    )


@mock.patch.dict(os.environ)
@SKIP_REQUESTS_NETRC
def test_get_netrc_auth_nomatch(netrc):
    os.environ["NETRC"] = str(netrc)
    assert igwn_requests.get_netrc_auth("https://bad.org/path") is None


@mock.patch.dict(os.environ)
@SKIP_REQUESTS_NETRC
def test_get_netrc_auth_notfound(tmp_path):
    netrc_file = tmp_path / "netrc"
    os.environ["NETRC"] = str(netrc_file)
    # file not found
    assert igwn_requests.get_netrc_auth(None) is None


@mock.patch.dict(os.environ)
@pytest.mark.skipif(
    os.name == "nt",
    reason="safe_netrc doesn't do anything on Windows",
)
@SKIP_REQUESTS_NETRC
def test_get_netrc_auth_permissions(netrc):
    os.environ["NETRC"] = str(netrc)
    netrc.chmod(0o444)
    with pytest.raises(NetrcParseError):
        igwn_requests.get_netrc_auth(None, raise_errors=True)
    assert igwn_requests.get_netrc_auth(None, raise_errors=False) is None


# -- HTTPSciTokenAuth -----------------

class TestHTTPSciTokenAuth:
    Auth = igwn_requests.HTTPSciTokenAuth

    def test_init(self):
        auth = self.Auth()
        assert auth.token is None
        assert auth.audience is None
        assert auth.scope is None
        assert auth.issuer is None

    def test_eq(self):
        a = self.Auth(token=None, audience="ANY")
        b = self.Auth(token=None, audience="ANY")
        assert a == b

    def test_neq(self):
        a = self.Auth(token=None, audience="ANY")
        b = self.Auth(token=None, audience="https://example.com")
        assert a != b

    @mock.patch("igwn_auth_utils.requests.find_scitoken")
    def test_token_header_empty(self, find_token):  # noqa: F811
        """Test that the auth class handles no tokens properly.
        """
        find_token.return_value = None
        req = MockRequest()
        auth = self.Auth()
        assert auth(req).headers.get("Authorization") is None

    @mock.patch("igwn_auth_utils.requests.find_scitoken")
    def test_token_header(self, find_token, rtoken):  # noqa: F811
        """Test that the auth class finds the token and serialises
        it into a header properly.
        """
        find_token.return_value = rtoken

        auth = self.Auth()
        req = MockRequest()
        assert auth(req).headers["Authorization"] == (
            igwn_requests.scitoken_authorization_header(rtoken)
        )


# -- Session --------------------------

class TestSession:
    Session = igwn_requests.Session

    # -- SessionErrorMixin

    def test_raise_for_status_hook(self, requests_mock):
        # define a request that returns 404 (not found)
        requests_mock.get(
            "https://test.org",
            status_code=404,
            reason="not found",
        )

        # with the kwarg a RequestException is raised
        with pytest.raises(
            RequestException,
            match=r"404 Client Error: not found for url: https://test.org/",
        ):
            igwn_requests.get("https://test.org")

    # -- session auth

    def test_noauth_args(self):
        """Test that `Session(force_noauth=True, fail_if_noauth=True)`
        is invalid.
        """
        with pytest.raises(ValueError):
            self.Session(force_noauth=True, fail_if_noauth=True)

    def test_fail_if_noauth(self):
        """Test that `Session(fail_if_noauth=True)` raises an error
        """
        with pytest.raises(IgwnAuthError):
            self.Session(
                token=False,
                cert=False,
                url=None,
                fail_if_noauth=True,
            )

    def test_force_noauth(self):
        """Test that `Session(force_noauth=True)` overrides auth kwargs
        """
        sess = self.Session(cert="cert.pem", force_noauth=True)
        assert sess.cert is False
        assert sess.auth is None

    @mock_no_scitoken()
    @mock_no_x509()
    def test_defaults(self):
        """Test that the `Session()` defaults work in a noauth environment
        """
        sess = self.Session()
        assert sess.cert is None
        assert isinstance(sess.auth, igwn_requests.HTTPSciTokenAuth)
        assert sess.auth.token is None

    # -- tokens

    def test_token_explicit(self, rtoken):  # noqa: F811
        """Test that tokens are handled properly
        """
        sess = self.Session(token=rtoken)
        assert sess.auth.token is rtoken
        # mock the request to get the header that would be used
        req = MockRequest()
        sess.auth(req)
        assert req.headers["Authorization"] == (
            igwn_requests.scitoken_authorization_header(rtoken)
        )

    def test_token_serialized(self, rtoken):  # noqa: F811
        """Test that serialized tokens are handled properly
        """
        serialized = rtoken.serialize()
        sess = self.Session(token=serialized)
        req = MockRequest()
        sess.auth(req)
        assert req.headers["Authorization"] == f"Bearer {serialized}"
        # will not deserialise a token for storage
        assert sess.token is None

    @mock.patch("igwn_auth_utils.requests.find_scitoken")
    def test_token_discovery(self, find_token, rtoken):  # noqa: F811
        find_token.return_value = rtoken
        sess = self.Session()
        sess.auth(sess)
        assert sess.headers["Authorization"] == (
            igwn_requests.scitoken_authorization_header(rtoken)
        )

    @mock.patch(
        "igwn_auth_utils.requests.find_scitoken",
        side_effect=IgwnAuthError,
    )
    def test_token_required_failure(self, _):
        with pytest.raises(IgwnAuthError), self.Session(token=True) as sess:
            sess.get("https://example.com")

    @pytest.mark.parametrize(("url", "aud"), (
        ("https://secret.example.com:8008", ["https://secret.example.com"]),
        (None, None)
    ))
    @mock.patch("igwn_auth_utils.requests.find_scitoken")
    def test_token_audience_default(self, find_scitoken, url, aud):
        """Check that the default `token_audience` is set correctly.
        """
        sess = self.Session(
            url=url,
            token=True,
            cert=False,
            auth=None,
        )
        assert sess.auth.audience == aud

    # -- X.509

    def test_cert_explicit(self):
        """Test that cert credentials are stored properly
        """
        sess = self.Session(token=False, cert="cert.pem")
        assert sess.cert == "cert.pem"
        assert sess.auth is None

    @mock.patch(
        "igwn_auth_utils.requests.find_x509_credentials",
        return_value="test.pem",
    )
    def test_cert_discovery(self, _):
        """Test that automatic certificate discovery works
        """
        assert self.Session(token=False).cert == "test.pem"

    @pytest.mark.skipif(
        not igwn_requests.REQUESTS_CERT_HTTP_BUG,
        reason=f"bug not present on requests {requests_version}",
    )
    def test_cert_requests_214(self):
        """Test that cert is disabled on requests 2.14 by default.
        """
        s = self.Session(url="http://example.com")
        assert s.cert is False

    @mock.patch(
        "igwn_auth_utils.requests.find_x509_credentials",
        side_effect=IgwnAuthError,
    )
    def test_cert_required_failure(self, _):
        with pytest.raises(IgwnAuthError):
            self.Session(token=False, cert=True)

    # -- basic auth

    @mock.patch.dict(os.environ)
    @pytest.mark.parametrize(("url", "auth"), (
        ("https://example.org", ("albert.einstein", "super-secret")),
        ("https://bad.org", None),
    ))
    @SKIP_REQUESTS_NETRC
    def test_basic_auth(self, netrc, url, auth):
        os.environ["NETRC"] = str(netrc)
        sess = self.Session(cert=False, token=False, url=url)
        assert sess.auth == auth

    # -- all

    @mock.patch("igwn_auth_utils.requests.find_scitoken")
    @mock.patch(
        "igwn_auth_utils.requests.find_x509_credentials",
        return_value=None,
    )
    @pytest.mark.parametrize(("cert", "token", "auth"), (
        (None, None, None),  # none
        ("A", False, ("C", "D")),  # no token
        ("A", True, None),  # no basic auth
        (None, True, None),  # no cert or basic auth
        (False, False, ("C", "D")),  # no cert or token
    ))
    def test_multi_auth(
        self,
        find_x509,
        find_token,
        rtoken,  # noqa: F811
        cert,
        token,
        auth,
    ):
        """Check that Session._init_auth records all auth options

        In case a remote host accepts X.509 but not tokens, but the user
        has a valid ANY token (for example).
        """
        find_token.return_value = rtoken
        sess = self.Session(cert=cert, token=token, auth=auth)
        assert sess.cert == cert
        if token:
            sess.auth(sess)
            sess.headers["Authorization"].startswith("Bearer")
        else:
            assert sess.auth == auth

    # -- request auth
    # test that Session auth and Request auth play nicely together

    @mock.patch(
        "igwn_auth_utils.x509.is_valid_certificate",
        return_value=False,
    )
    def test_request_x509(self, is_valid, requests_mock, tmp_path):
        """Test that a request does its own search for an X.509 cert.
        """
        x509 = tmp_path / "x509"
        requests_mock.get("https://example.com")
        with mock.patch.dict(os.environ):
            os.environ.pop("X509_USER_PROXY", None)
            with self.Session(cert=None) as sess:
                # check that the Session doesn't have a cert
                assert sess.cert is None

                # but that the request uses one because it calls
                # find_x509_credentials again
                x509.touch()
                os.environ["X509_USER_PROXY"] = str(x509)
                resp = sess.get("https://example.com", cert=None)
                assert resp.request.cert == str(x509)

    @mock.patch("igwn_auth_utils.requests.find_scitoken", return_value=None)
    @pytest.mark.parametrize(
        ("session_aud", "session_scope", "request_aud", "request_scope"),
        [
            (None, None, "aud", "scope"),
            ("aud", "scope", None, "newscope"),
            ("aud", "scope", "newaud", None),
            (None, None, None, None),
        ],
    )
    def test_request_token_auth(
        self,
        find_scitoken,
        requests_mock,
        session_aud,
        session_scope,
        request_aud,
        request_scope,
    ):
        """Test that a request correctly merges token claim settings.
        """
        requests_mock.get("https://example.com/api")
        with self.Session(
            cert=False,
            token_audience=session_aud,
            token_scope=session_scope,
        ) as sess:  # use a token
            # check that the session auth handler recorded what we gave it
            assert sess.auth.audience == session_aud
            assert sess.auth.scope == session_scope

            # but that the request auth uses any new settings we give it
            sess.get(
                "https://example.com/api",
                token_audience=request_aud,
                token_scope=request_scope,
            )
            assert find_scitoken.called_once_with(
                audience=request_aud or session_aud,
                scope=request_scope or session_scope,
            )

    @mock.patch("igwn_auth_utils.requests.find_scitoken", return_value=None)
    def test_request_fail_if_noauth(self, find_scitoken):
        """Test that `Session.get(fail_if_noauth=True)` raises an error.
        """
        with self.Session(fail_if_noauth=False) as sess, \
             pytest.raises(IgwnAuthError, match="no valid authorisation"):
            sess.get(
                "https://example.com",
                cert=False,
                fail_if_noauth=True,
            )


# -- standalone requests --------------

def test_get(requests_mock):
    """Test that `igwn_auth_utils.requests.get` can perform a simple request
    """
    requests_mock.get(
        "https://test.org",
        text="TEST",
    )
    assert igwn_requests.get("https://test.org").text == "TEST"


@mock.patch("igwn_auth_utils.requests.Session")
def test_get_session(mock_session):
    """Test that ``session`` for `igwn_auth_utils.requests.get` works
    """
    session = mock.MagicMock()
    assert igwn_requests.get("https://test.org", session=session)
    session.request.assert_called_once_with("get", "https://test.org")
    mock_session.assert_not_called()


@mock.patch("igwn_auth_utils.requests.find_scitoken")
@mock.patch("igwn_auth_utils.requests.find_x509_credentials")
def test_get_force_noauth(find_x509, find_scitoken, requests_mock):
    """Test that `igwn_auth_utils.requests.get` passes `force_noauth` properly.

    Regression: <https://git.ligo.org/computing/igwn-auth-utils/-/issues/12>
    """
    requests_mock.get("https://test.org")
    igwn_requests.get("https://test.org", force_noauth=True)
    find_x509.assert_not_called()
    find_scitoken.assert_not_called()


def test_post(requests_mock):
    """Test that `igwn_auth_utils.requests.post` can perform a simple request.
    """
    data = {"a": 1, "b": 2}
    requests_mock.post(
        "https://example.com",
        text="THANKS",
    )
    # check that the correct response got passed through
    assert igwn_requests.post(
        "https://example.com",
        data=data,
    ).text == "THANKS"
    # check that the data was encoded into the request properly
    req = requests_mock.request_history[0]
    assert req.body == urlencode(data)
