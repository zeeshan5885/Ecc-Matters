# -*- coding: utf-8 -*-
# Copyright 2021 Cardiff University
# Distributed under the terms of the BSD-3-Clause license

import os
from datetime import datetime
from pathlib import Path

from cryptography.x509 import (
    Certificate,
    load_pem_x509_certificate,
)
from cryptography.hazmat.backends import default_backend

from .error import IgwnAuthError


def load_x509_certificate_file(file, backend=None):
    """Load a PEM-format X.509 certificate from a file, or file path

    Parameters
    ----------
    file : `str`, `pathlib.Path`, `file`
        file object or path to read from

    backend : `module`, optional
        the `cryptography` backend to use

    Returns
    -------
    cert : `cryptography.x509.Certificate`
        the X.509 certificate
    """
    if isinstance(file, (str, bytes, os.PathLike)):
        with open(file, "rb") as fileobj:
            return load_x509_certificate_file(fileobj)
    if backend is None:  # cryptography < 3.1 requires a non-None backend
        backend = default_backend()
    return load_pem_x509_certificate(file.read(), backend=backend)


def validate_certificate(cert, timeleft=600):
    """Validate an X.509 certificate by checking it's expiry time

    Parameters
    ----------
    cert : `cryptography.x509.Certificate`, `str`, `file`
        the certificate object or file (object or path) to validate

    timeleft : `float`, optional
        the minimum required time until expiry (seconds)

    Raises
    ------
    ValueError
        if the certificate has expired or is about to expire
    """
    # load a certificate from a file
    if not isinstance(cert, Certificate):
        cert = load_x509_certificate_file(cert)

    # then validate it
    if _timeleft(cert) < timeleft:
        raise ValueError(
            f"X.509 certificate has less than {timeleft} seconds remaining"
        )


def is_valid_certificate(cert, timeleft=600):
    """Returns True if ``cert`` contains a valid X.509 certificate

    Parameters
    ----------
    cert : `cryptography.x509.Certificate`, `str`, `file`
        the certificate object or file (object or path) to validate

    timeleft : `float`, optional
        the minimum required time until expiry (seconds)

    Returns
    -------
    isvalid : `bool`
        whether the certificate is valid
    """
    try:
        validate_certificate(cert, timeleft=timeleft)
    except (
        OSError,  # file doesn't exist or isn't readable
        ValueError,  # cannot load PEM certificate or expiry looming
    ):
        return False
    return True


def _timeleft(cert):
    """Returns the time remaining (in seconds) for a ``cert``
    """
    return (cert.not_valid_after - datetime.utcnow()).total_seconds()


def _default_cert_path(prefix="x509up_"):
    """Returns the temporary path for a user's X509 certificate

    Examples
    --------
    On Windows:

    >>> _default_cert_path()
    'C:\\Users\\user\\AppData\\Local\\Temp\\x509up_user'

    On Unix:

    >>> _default_cert_path()
    '/tmp/x509up_u1000'
    """
    if os.name == "nt":  # Windows
        tmpdir = Path(os.environ["SYSTEMROOT"]) / "Temp"
        user = os.getlogin()
    else:  # Unix
        tmpdir = "/tmp"  # noqa: S108
        user = "u{}".format(os.getuid())
    return Path(tmpdir) / "{}{}".format(prefix, user)


def find_credentials(timeleft=600):
    """Locate X509 certificate and (optionally) private key files.

    This function checks the following paths in order:

    - ``${X509_USER_CERT}`` and ``${X509_USER_KEY}``
    - ``${X509_USER_PROXY}``
    - ``/tmp/x509up_u${UID}``
    - ``~/.globus/usercert.pem`` and ``~/.globus/userkey.pem``

    Note
    ----
    If the ``X509_USER_{CERT,KEY,PROXY}`` variables are set, their paths
    **are not** validated in any way, but are trusted to point at valid,
    non-expired credentials.
    The default paths in `/tmp` and `~/.globus` **are** validated before
    being returned.

    Parameters
    ----------
    timeleft : `int`
        minimum required time left until expiry (in seconds)
        for a certificate to be considered 'valid'

    Returns
    -------
    cert : `str`
        the path of the certificate file that also contains the
        private key, **OR**

    cert, key : `str`
        the paths of the separate cert and private key files

    Raises
    ------
    ~igwn_auth_utils.IgwnAuthError
        if not certificate files can be found, or if the files found on
        disk cannot be validtted.

    Examples
    --------
    If no environment variables are set, but a short-lived certificate has
    been generated in the default location:

    >>> find_credentials()
    '/tmp/x509up_u1000'

    If a long-lived (grid) certificate has been downloaded:

    >>> find_credentials()
    ('/home/me/.globus/usercert.pem', '/home/me/.globus/userkey.pem')
    """
    # -- check the environment variables (without validation)

    try:
        return os.environ['X509_USER_CERT'], os.environ['X509_USER_KEY']
    except KeyError:
        try:
            return os.environ['X509_USER_PROXY']
        except KeyError:
            pass

    # -- look up some default paths (with validation)

    # 1: /tmp/x509up_u<uid> (cert = key)
    default = str(_default_cert_path())
    if is_valid_certificate(default, timeleft):
        return default

    # 2: ~/.globus/user{cert,key}.pem
    try:
        globusdir = Path.home() / ".globus"
    except RuntimeError:  # pragma: no cover
        # no 'home'
        pass
    else:
        cert = str(globusdir / "usercert.pem")
        key = str(globusdir / "userkey.pem")
        if (
            is_valid_certificate(cert, timeleft)  # validate the cert
            and os.access(key, os.R_OK)  # sanity check the key
        ):
            return cert, key

    raise IgwnAuthError(
        "could not find an RFC-3820 compliant X.509 credential, "
        "please generate one and try again.",
    )
