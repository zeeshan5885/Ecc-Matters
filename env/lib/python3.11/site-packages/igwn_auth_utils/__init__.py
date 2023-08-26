# -*- coding: utf-8 -*-
# Copyright 2021-2022 Cardiff University
# Distributed under the terms of the BSD-3-Clause license

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Duncan Brown, Leo Singer"
__license__ = "BSD-3-Clause"

from .error import IgwnAuthError
from .requests import (
    get,
    request,
    HTTPSciTokenAuth,
    Session,
    SessionAuthMixin,
    SessionErrorMixin,
)
from .scitokens import (
    find_token as find_scitoken,
    token_authorization_header as scitoken_authorization_header,
)
from .x509 import (
    find_credentials as find_x509_credentials,
)

try:  # parse version
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = ''
