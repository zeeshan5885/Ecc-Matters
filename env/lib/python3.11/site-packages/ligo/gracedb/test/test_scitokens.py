import mock
import os
import pytest
import scitokens
import tempfile

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key

from ligo.gracedb.rest import GraceDb

TEST_ISSUER = "local"
TEST_AUDIENCE = ["TEST", "ANY"]
TEST_SCOPE = "gracedb.read"


# -- fixtures ---------------

@pytest.fixture(scope="session")  # one per suite is fine
def private_key():
    return generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend(),
    )


@pytest.fixture
def rtoken(private_key):
    """Create a token
    """
    # configure keycache
    from scitokens.utils import keycache
    kc = keycache.KeyCache.getinstance()
    kc.addkeyinfo(
        TEST_ISSUER,
        "test_key",
        private_key.public_key(),
        cache_timer=60,
    )

    # create token
    token = scitokens.SciToken(key=private_key, key_id="test_key")
    token.update_claims({
        "iss": TEST_ISSUER,
        "aud": TEST_AUDIENCE,
        "scope": TEST_SCOPE,
    })
    serialized_token = token.serialize(issuer=TEST_ISSUER,
                                       lifetime=1000).decode("utf-8")
    return serialized_token


# -- test scitokens ---------

def test_scitokens(rtoken):
    with tempfile.TemporaryDirectory() as tmpdir:
        scitoken_file = os.path.join(tmpdir, "test_scitoken")

        with os.fdopen(os.open(scitoken_file,
                       os.O_RDWR | os.O_CREAT, 0o500), 'w+') as h:
            h.write(rtoken)

        # Initialize client
        with mock.patch.dict(os.environ, {'SCITOKEN_FILE': scitoken_file}):
            g = GraceDb()

        assert 'scitoken' in g.auth_type
