# -*- coding: utf-8 -*-
# Copyright 2021 Cardiff University
# Distributed under the terms of the BSD-3-Clause license

"""Utilities for the `igwn_auth_utils` test suite
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key

import pytest


@pytest.fixture(scope="session")  # one per suite is fine
def private_key():
    return generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )


@pytest.fixture(scope="session")
def public_key(private_key):
    return private_key.public_key()


@pytest.fixture(scope="session")
def public_pem(public_key):
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )


@pytest.fixture()
def public_pem_path(public_pem, tmp_path):
    pem_path = tmp_path / "key.pem"
    with open(pem_path, "wb") as file:
        file.write(public_pem)
    return pem_path
