# -*- coding: utf-8 -*-
# Copyright (C) Scott Koranda (2012-2015)
#               Louisiana State University (2015-2017)
#               Cardiff University (2017-2022)
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Utilities for the GW datafind service.
"""

import os
import warnings

from ligo.segments import segment

from igwn_auth_utils import x509 as igwn_x509


def get_default_host():
    """Return the default host as stored in the ``${GWDATAFIND_SERVER}``.

    Returns
    -------
    host : `str`
        the URL of the default host

    Raises
    ------
    ValueError
        if the ``GWDATAFIND_SERVER`` environment variable is not set
    """
    try:
        return os.environ["GWDATAFIND_SERVER"]
    except KeyError:
        try:
            return os.environ["LIGO_DATAFIND_SERVER"]
        except KeyError:
            raise ValueError(
                "Failed to determine default gwdatafind host, please "
                "pass manually or set the `GWDATAFIND_SERVER` "
                "environment variable",
            )


def validate_proxy(path):
    """Validate an X509 proxy certificate file.

    .. warning::

       This method is deprecated and will be removed in a future release.

    This function tests that the proxy certificate is
    `RFC 3820 <https://www.ietf.org/rfc/rfc3820.txt>` compliant and is
    not expired.

    Parameters
    ----------
    path : `str`
        the path of the X509 file on disk.

    Returns
    -------
    valid : `True`
        if the certificate validiates.

    Raises
    ------
    RuntimeError
        if the certificate cannot be validated.
    """
    warnings.warn(
        "this method is deprecated and will be removed in gwdatafind-3.0.0, "
        "please update your workflow to use "
        "igwn_auth_utils.x509.validate_certificate",
        DeprecationWarning,
    )
    igwn_x509.validate_certificate(path, timeleft=0)
    # return True to indicate validated proxy
    return True


def find_credential():
    """Locate X509 certificate and key files.

    .. warning::

       This method is deprecated and will be removed in a future release.

    This function checks the following paths in order:

    - ``${X509_USER_PROXY}``
    - ``${X509_USER_CERT}`` and ``${X509_USER_KEY}``
    - ``/tmp/x509up_u${UID}``

    Returns
    -------
    cert, key : `str`, `str`
        the paths of the certificate and key files, respectively.

    Raises
    ------
    RuntimeError
        if not certificate files can be found, or if the files found on
        disk cannot be validtted.
    """
    warnings.warn(
        "this method is deprecated and will be removed in gwdatafind-3.0.0, "
        "please update your workflow to use "
        "igwn_auth_utils.x509.find_credentials",
        DeprecationWarning,
    )
    creds = igwn_x509.find_credentials(timeleft=0)
    if isinstance(creds, tuple):  # (cert, private key) pair
        return creds
    # just a cert, so return it as the private key as well
    return creds, creds


# -- LIGO-T050017 filename parsing --------------------------------------------

def filename_metadata(filename):
    """Return metadata parsed from a filename following LIGO-T050017.

    Parameters
    ----------
    filename : `str`
        the path name of a file

    Returns
    -------
    obs : `str`
        the observatory metadata

    tag : `str`
        the file tag

    segment : `ligo.segments.segment`
        the GPS ``[start, stop)`` interval for this file

    Notes
    -----
    `LIGO-T050017 <https://dcc.ligo.org/LIGO-T050017>`__ declares a
    file naming convention that includes documenting the GPS start integer
    and integer duration of a file, see that document for more details.
    """
    obs, desc, start, end = os.path.basename(filename).split('-')
    start = int(start)
    end = int(end.split('.')[0])
    return obs, desc, segment(start, start+end)


def file_segment(filename):
    """Return the data segment for a filename following LIGO-T050017.

    Parameters
    ----------
    filename : `str`
        the path of a file.

    Returns
    -------
    segment : `~ligo.segments.segment`
        the ``[start, stop)`` GPS segment covered by the given file
    """
    return filename_metadata(filename)[2]
