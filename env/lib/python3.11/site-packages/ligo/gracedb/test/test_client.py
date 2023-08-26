import pytest
from unittest import mock

from ligo.gracedb.rest import GraceDb
from igwn_auth_utils.error import IgwnAuthError


def test_provide_x509_cert_and_key():
    """Test client instantiation with provided certificate and key files"""
    # Set up cert and key files
    cert_file = '/tmp/cert_file'
    key_file = '/tmp/key_file'

    load_cert_func = 'ligo.gracedb.cert.load_certificate'
    with mock.patch(load_cert_func):  # noqa: E127
        # Initialize client
        g = GraceDb(cred=(cert_file, key_file))

    # Output credentials:
    creds = g.show_credentials(print_output=False)

    # Check credentials
    assert len(g.cert) == 2
    assert 'x509' in g.auth_type
    assert creds['x509'].get('cert_file') == cert_file
    assert creds['x509'].get('key_file') == key_file


def test_provide_x509_proxy():
    """Test client instantiation with provided combined proxy file"""
    # Set up combined proxy file
    proxy_file = '/tmp/proxy_file'

    exists_func = 'os.path.exists'
    with mock.patch(exists_func) as mock_exists_func:  # noqa: E127
        g = GraceDb(cred=proxy_file)

        mock_exists_func.return_value = True
        # Initialize client

    # Output credentials:
    creds = g.show_credentials(print_output=False)

    # Check credentials
    assert type(g.cert) == str
    assert 'x509' in g.auth_type
    assert creds['x509'].get('cert_file') == proxy_file
    assert creds['x509'].get('key_file') == proxy_file


def test_provide_all_creds():
    """Test providing all credentials to the constructor
       FIXME: add scitokens to this"""
    # Setup
    cert_file = '/tmp/cert_file'
    key_file = '/tmp/key_file'

    # Initialize client
    g = GraceDb(cred=(cert_file, key_file))

    # Check credentials - should prioritze x509 credentials
    creds = g.show_credentials(print_output=False)

    assert len(g.cert) == 2
    assert 'x509' in g.auth_type
    assert creds['x509'].get('cert_file') == cert_file
    assert creds['x509'].get('key_file') == key_file


def test_x509_credentials_lookup():
    """Test lookup of X509 credentials"""
    # Setup
    cert_file = '/tmp/cert_file'
    key_file = '/tmp/key_file'

    g = GraceDb(cred=(cert_file, key_file))

    # the 'auth_type' variable doesn't get set in this case, but it's
    # sort of fictitious when self.cert is being mocked.
    assert len(g.cert) == 2


def test_x509_lookup_cert_key_from_envvars():
    """Test lookup of X509 cert and key from environment variables"""
    # Setup
    cert_file = '/tmp/cert_file'
    key_file = '/tmp/key_file'

    # Initialize client
    fake_creds_dict = {
        'X509_USER_CERT': cert_file,
        'X509_USER_KEY': key_file,
    }
    os_environ_func = 'os.environ'
    with mock.patch.dict(os_environ_func, fake_creds_dict):  # noqa: E127
        g = GraceDb()

    # Check credentials - should prioritze x509 credentials
    creds = g.show_credentials(print_output=False)

    assert len(g.cert) == 2
    assert 'x509' in g.auth_type
    assert creds['x509'].get('cert_file') == cert_file
    assert creds['x509'].get('key_file') == key_file


def test_x509_lookup_proxy_from_envvars():
    """Test lookup of X509 combined provxy file from environment variables"""
    # Setup
    proxy_file = '/tmp/proxy_file'

    # Initialize client
    os_environ_func = 'os.environ'
    mock_environ_dict = {'X509_USER_PROXY': proxy_file}
    with mock.patch.dict(os_environ_func, mock_environ_dict):  # noqa: E127
        g = GraceDb()

    # Check credentials - should prioritze x509 credentials
    creds = g.show_credentials(print_output=False)

    assert type(g.cert) == str
    assert 'x509' in g.auth_type
    assert creds['x509'].get('cert_file') == proxy_file
    assert creds['x509'].get('key_file') == proxy_file


@mock.patch("igwn_auth_utils.requests.find_scitoken", return_value=None)
@mock.patch("igwn_auth_utils.requests.find_x509_credentials",
            return_value=None)
def test_no_credentials(_find_x509, _find_token):
    """Test client instantiation with no credentials at all"""
    # Check credentials
    g = GraceDb(fail_if_noauth=False)
    assert not g.auth_type


@mock.patch.dict("os.environ", {
    "X509_USER_CERT": "cert_file",
    "X509_USER_KEY": "key_file",
})
def test_force_noauth():
    """Test forcing no authentication, even with X509 certs available"""
    # Initialize client
    g = GraceDb(force_noauth=True)

    # Check credentials
    assert not g.auth_type


@mock.patch("igwn_auth_utils.requests.find_scitoken", return_value=None)
@mock.patch("igwn_auth_utils.requests.find_x509_credentials",
            return_value=None)
def test_fail_if_noauth(*patches):
    """Test failing if no authentication credentials are provided"""
    err_str = 'no valid authorisation credentials found'
    with pytest.raises(IgwnAuthError, match=err_str):
        GraceDb(fail_if_noauth=True)


def test_fail_if_noauth_creds():
    """Test fail_if_noauth doesn't error if credentials are provided"""
    cert_file = '/tmp/cert_file'
    key_file = '/tmp/key_file'

    # Initialize client:
    g = GraceDb(cred=(cert_file, key_file), fail_if_noauth=True)

    # Check credentials
    assert g.cert == (cert_file, key_file)


def test_force_noauth_and_fail_if_noauth():
    # Initialize client
    err_str = 'cannot select both force_noauth and fail_if_noauth'
    with pytest.raises(ValueError, match=err_str):
        GraceDb(force_noauth=True, fail_if_noauth=True)


@pytest.mark.parametrize(
    "resource,key",
    [
        ('api_versions', 'api-versions'),
        ('server_version', 'server-version'),
        ('links', 'links'),
        ('templates', 'templates'),
        ('groups', 'groups'),
        ('pipelines', 'pipelines'),
        ('searches', 'searches'),
        ('allowed_labels', 'labels'),
        ('superevent_categories', 'superevent-categories'),
        ('em_groups', 'em-groups'),
        ('voevent_types', 'voevent-types'),
        ('signoff_types', 'signoff-types'),
        ('signoff_statuses', 'signoff-statuses'),
        ('instruments', 'instruments'),
    ]
)
def test_properties_from_api_root(safe_client, resource, key):
    si_prop = 'ligo.gracedb.rest.GraceDb.service_info'
    with mock.patch(si_prop, new_callable=mock.PropertyMock()) as mock_si:
        getattr(safe_client, resource)

    call_args, call_kwargs = mock_si.get.call_args
    assert mock_si.get.call_count == 1
    assert len(call_args) == 1
    assert call_kwargs == {}
    assert call_args[0] == key


@pytest.mark.parametrize("api_version", [1, 1.2, [], (), {}])
def test_bad_api_version(api_version):
    err_msg = 'api_version should be a string'
    with pytest.raises(TypeError, match=err_msg):
        GraceDb(api_version=api_version)


@pytest.mark.parametrize(
    "service_url,api_version",
    [
        ('test', None),
        ('test/', None),
        ('test', 'v1'),
        ('test/', 'v2'),
        ('test/', 'default'),
    ],
)
def test_set_service_url(safe_client, service_url, api_version):
    safe_client._set_service_url(service_url, api_version)

    # Construct expected service urls
    expected_service_url = service_url.rstrip('/') + '/'
    expected_versioned_service_url = expected_service_url

    if api_version and api_version != 'default':
        expected_versioned_service_url += api_version + '/'

    assert safe_client._service_url == expected_service_url
    assert safe_client._versioned_service_url == expected_versioned_service_url


def test_invalid_retries(safe_client):
    retries = -1
    expected_response = 'Invalid value of retries'
    with pytest.raises(ValueError, match=expected_response):
        GraceDb(retries=retries)


def test_invalid_retry_codes(safe_client):
    retry_codes = 503
    expected_response = 'retry_codes must be a list'
    with pytest.raises(ValueError, match=expected_response):
        GraceDb(retry_codes=retry_codes)
