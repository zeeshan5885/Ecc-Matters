# -*- coding: utf-8 -*-
# Copyright 2021 Cardiff University
# Distributed under the terms of the BSD-3-Clause license

"""Tests for :mod:`igwn_auth_utils.x509`.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

import os
from datetime import (
    datetime,
    timedelta,
)
from pathlib import Path
from unittest import mock

from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import (
    hashes,
    serialization,
)

import pytest

from .. import x509 as igwn_x509
from ..error import IgwnAuthError


# -- fixtures ---------------

@pytest.fixture
def x509cert(private_key, public_key):
    name = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "test"),
    ])
    now = datetime.utcnow()
    return x509.CertificateBuilder(
        issuer_name=name,
        subject_name=name,
        public_key=public_key,
        serial_number=1000,
        not_valid_before=now,
        not_valid_after=now + timedelta(seconds=86400),
    ).sign(private_key, hashes.SHA256(), backend=default_backend())


def _write_x509(cert, path):
    with open(path, "wb") as file:
        file.write(cert.public_bytes(
            serialization.Encoding.PEM,
        ))


@pytest.fixture
def x509cert_path(x509cert, tmp_path):
    cert_path = tmp_path / "x509.pem"
    _write_x509(x509cert, cert_path)
    return cert_path


# -- tests ------------------

@mock.patch.dict("os.environ")
@mock.patch(
    "os.getlogin" if os.name == "nt" else "os.getuid",
    mock.MagicMock(return_value=123),
)
def test_default_cert_path():
    if os.name == "nt":
        os.environ["SYSTEMROOT"] = r"C:\WINDOWS"
        expected = r"C:\WINDOWS\Temp\x509up_123"
    else:
        expected = r"/tmp/x509up_u123"  # noqa: S108
    assert igwn_x509._default_cert_path() == Path(expected)


def test_validate_certificate(x509cert):
    igwn_x509.validate_certificate(x509cert)


def test_validate_certificate_path(x509cert_path):
    igwn_x509.validate_certificate(x509cert_path)


def test_validate_certificate_expiry_error(x509cert):
    with pytest.raises(
        ValueError,
        match="X.509 certificate has less than 10000000000 seconds remaining",
    ):
        igwn_x509.validate_certificate(x509cert, timeleft=int(1e10))


def test_is_valid_certificate(x509cert_path):
    assert igwn_x509.is_valid_certificate(x509cert_path)


def test_is_valid_certificate_false(tmp_path):
    assert not igwn_x509.is_valid_certificate(tmp_path / "does-not-exist")


@mock.patch.dict("os.environ")
def test_find_credentials_x509usercertkey(x509cert_path, public_pem_path):
    """Test that `find_credentials()` returns the X509_USER_{CERT,KEY} pair
    """
    os.environ.pop("X509_USER_PROXY", "test")  # make sure this doesn't win
    # configure the environment to return (cert, key)
    x509cert_filename = str(x509cert_path)
    x509key_filename = str(public_pem_path)
    os.environ["X509_USER_CERT"] = x509cert_filename
    os.environ["X509_USER_KEY"] = x509key_filename

    # check that find_credentials() returns the (cert, key) pair
    assert igwn_x509.find_credentials() == (
        x509cert_filename,
        x509key_filename,
    )


@mock.patch.dict("os.environ")
def test_find_credentials_x509userproxy(x509cert_path):
    """Test that `find_credentials()` returns the X509_USER_PROXY if set

    ... if X509_USER_{CERT,KEY} are not set
    """
    # remove CERT,KEY so that PROXY can win
    os.environ.pop("X509_USER_CERT", None)
    os.environ.pop("X509_USER_KEY", None)
    # set the PROXY variable
    x509cert_filename = str(x509cert_path)
    os.environ["X509_USER_PROXY"] = x509cert_filename
    # make sure it gets found
    assert igwn_x509.find_credentials() == x509cert_filename


@mock.patch.dict("os.environ", clear=True)
@mock.patch("igwn_auth_utils.x509._default_cert_path")
def test_find_credentials_default(_default_cert_path, x509cert_path):
    """Test that `find_credentials()` returns the default path

    ... if none of the X509_USER variable are set
    """
    _default_cert_path.return_value = x509cert_path
    assert igwn_x509.find_credentials() == str(x509cert_path)


@mock.patch.dict(
    "os.environ",
)
@mock.patch(
    "igwn_auth_utils.x509.is_valid_certificate",
    side_effect=(
        False,  # fail for _default_cert_path
        True,  # so that ~/.globus/usercert.pem passes
    ),
)
@mock.patch("os.access", return_value=True)
def test_find_credentials_globus(_, x509cert_path):
    """Test that .globus files are returned if all else fails
    """
    # clear X509 variables out of the environment
    for suffix in ("PROXY", "CERT", "KEY"):
        os.environ.pop(f"X509_USER_{suffix}", None)

    # check that .globus is found
    globusdir = Path.home() / ".globus"
    assert igwn_x509.find_credentials() == (
        str(globusdir / "usercert.pem"),
        str(globusdir / "userkey.pem"),
    )


@mock.patch.dict("os.environ")
@mock.patch("igwn_auth_utils.x509.is_valid_certificate", return_value=False)
def test_find_credentials_error(_):
    """Test that a failure in discovering X.509 creds raises the right error
    """
    # clear X509 variables out of the environment
    for suffix in ("PROXY", "CERT", "KEY"):
        os.environ.pop(f"X509_USER_{suffix}", None)

    # check that we can't find any credentials
    with pytest.raises(
        IgwnAuthError,
        match="could not find an RFC-3820 compliant X.509 credential",
    ):
        igwn_x509.find_credentials()
