# -*- coding: utf-8 -*-
# Copyright (C) Alexander Pace, Tanner Prestegard,
#               Branson Stephens, Brian Moe (2020)
#
# This file is part of gracedb
#
# gracedb is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# It is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gracedb.  If not, see <http://www.gnu.org/licenses/>.
import sys
import json as json_lib
from datetime import datetime
from pprint import pprint
from requests.adapters import HTTPAdapter, Retry

from .version import __version__
from .adapter import GraceDbCertAdapter
from .utils import hook_response, raise_status_exception

from igwn_auth_utils.requests import Session

# To remove later: python2 compatibility fix:
if sys.version_info[0] > 2:
    from urllib.parse import urlparse
else:
    from urlparse import urlparse

DEFAULT_SERVICE_URL = "https://gracedb.ligo.org/api/"
DEFAULT_RETRY_CODES = [500, 502, 503, 504, 408, 409]


class GraceDBClient(Session):
    """
    url (:obj:`str`, optional): URL of server API
    cred (:obj:`tuple` or :obj:`str, optional): a tuple or list of
        (``/path/to/cert/file``, ``/path/to/key/file) or a single path to
        a combined proxy file (if using an X.509 certificate for
        authentication)
    force_noauth (:obj:`bool`, optional): set to True if you want to skip
        credential lookup and use this client as an unauthenticated user
    fail_if_noauth (:obj:`bool`, optional): set to True if you want the
        constructor to fail if no authentication credentials are provided
        or found
    reload_certificate (:obj:`bool`, optional): if ``True``, your
        certificate will be checked before each request whether it is
        within ``reload_buffer`` seconds of expiration, and if so, it will
        be reloaded. Useful for processes which may live longer than the
        certificate lifetime and have an automated method for certificate
        renewal. The path to the new/renewed certificate **must** be the
        same as for the old certificate.
    reload_buffer (:obj:`int`, optional): buffer (in seconds) for reloading
        a certificate in advance of its expiration. Only used if
        ``reload_certificate`` is ``True``.
    retries (:obj:`int`, optional): the maximum number of retries to
        attempt on an error from the server. Default 5.
    backoff_factor (:obj:`float`, optional): The backoff factor for retrying.
        Default: 0.1. Refer to urllib3 documentation for usage.
    retry_codes (:obj:`list`, optional): List of HTTPError codes to retry
        on. Default: [500, 502, 503, 504, 408, 409].

    Authentication details:
    You can:
        1. Provide a path to an X.509 certificate and key or a single
           combined proxy file
    Or:
        The code will look for a certificate in a default location
            (``/tmp/x509up_u%d``, where ``%d`` is your user ID)
    """

    def __init__(self, url=DEFAULT_SERVICE_URL, cred=None,
                 reload_certificate=False, reload_buffer=300,
                 use_auth='all', retries=5,
                 retry_codes=DEFAULT_RETRY_CODES,
                 backoff_factor=0.1, *args, **kwargs):

        # Set which auth method to use if specified, otherwise use all.
        token = None
        if use_auth == 'x509':
            token = False
        elif use_auth == 'scitoken':
            cred = False

        super().__init__(cert=cred, token=token, *args, **kwargs)

        # Initialize variables:
        self.host = urlparse(url).hostname

        # Used for decoding scitokens in show_credentials.
        u = urlparse(url)
        self.token_audience = [
            "ANY",
            f"{u.scheme}://{u.netloc}",
            f"{u.scheme}://{u.hostname}",
        ]

        # Define auth_type based on.... type of auth.
        # We're not supporting basic auth, so self.auth isn't recognized.
        self.auth_type = {}
        if use_auth != 'x509' and 'Authorization' in self.headers.keys():
            self.auth_type['scitoken'] = None
        if use_auth != 'scitoken' and self.cert:
            self.auth_type['x509'] = None

        # Update session headers:
        self.headers.update(self._update_headers())

        # Adjust the response via a session hook:
        self.hooks = {'response': [hook_response, raise_status_exception]}

        # Add the retrying adaptor:
        # https://stackoverflow.com/a/35504626
        # Sanity check the inputs:

        # 'retries' must be a positive (or zero) integer:
        if (not isinstance(retries, int) or retries < 0):
            raise ValueError('Invalid value of retries')

        # 'retry_codes' must be a list:
        if not isinstance(retry_codes, list):
            raise ValueError('retry_codes must be a list')

        if retries > 0:
            retries = Retry(total=retries,
                            backoff_factor=backoff_factor,
                            status_forcelist=retry_codes)

        if reload_certificate and self.cert:
            self.mount('https://', GraceDbCertAdapter(
                       cert=self.cert,
                       reload_buffer=reload_buffer,
                       max_retries=retries))
        else:
            self.mount('https://', HTTPAdapter(max_retries=retries))

    def _update_headers(self):
        """ Update the sessions' headers:
        """
        new_headers = {}
        # Assign the user agent. This shows up in gracedb's log files.
        new_headers.update({'User-Agent':
                            'gracedb-client/{}'.format(__version__)})
        new_headers.update({'Accept-Encoding':
                            'gzip, deflate'})
        return new_headers

    # hijack 'Session.request':
    # https://2.python-requests.org/en/master/api/#requests.Session.request
    def request(
            self, method, url, params=None, data=None, headers=None,
            cookies=None, files=None, auth=None, timeout=None,
            allow_redirects=True, proxies=None, hooks=None, stream=None,
            verify=None, cert=None, json=None):
        return super().request(
            method, url, params=params, data=data, headers=headers,
            cookies=cookies, files=files, auth=auth,
            timeout=timeout, allow_redirects=True, proxies=proxies,
            hooks=hooks, stream=stream, verify=verify, cert=cert, json=json)

    # Extra definitions to return closed contexts' connections
    # back to the pool:
    # https://stackoverflow.com/questions/48160728/resourcewarning
    # -unclosed-socket-in-python-3-unit-test
    def close(self):
        super().close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # For getting files, return a "raw" file-type object.
    # Automatically decode content.
    def get_file(self, url, **kwargs):
        resp = self.get(url, stream=True, **kwargs)
        resp.raw.decode_content = True
        resp.raw.status_code = resp.status_code
        resp.raw.json = resp.json
        return resp.raw

    # Return client credentials
    def show_credentials(self, print_output=True):
        """ Prints authentication type and credentials info."""
        output = {}
        if not self.auth_type:
            output = {'auth_type(s)': 'No auth type found'}

        token = {}
        cert = {}

        if 'scitoken' in self.auth_type:
            token = {'SciToken subject': self.token['sub'],
                     'SciToken expiration': datetime.utcfromtimestamp(
                     self.token['exp']).isoformat(),
                     'SciToken scope': self.token['scope'],
                     'SciToken audience': self.token['aud']}
            output['scitoken'] = token
        if 'x509' in self.auth_type:
            if isinstance(self.cert, tuple):
                cert = {'cert_file': self.cert[0],
                        'key_file': self.cert[1]}
            elif isinstance(self.cert, str):
                cert = {'cert_file': self.cert,
                        'key_file': self.cert}
            else:
                raise ValueError("Problem reading authentication certificate")
            output['x509'] = cert

        if print_output:
            pprint(output)
        else:
            return output

    def get_user_info(self):
        """Get information from the server about your user account."""
        user_info_link = self.links.get('user-info', None)
        if user_info_link is None:
            raise RuntimeError('Server does not provide a user info endpoint')
        return self.get(user_info_link)

    @classmethod
    def load_json_from_response(cls, response):
        """ Always return a json content response, even when the server
            provides a 204: no content"""
        # Check if response exists:
        if not response:
            raise ValueError("No response object provided")

        # Check if there is response content. If not, create it.
        if response.content == 'No Content':
            response_content = '{}'

        # Some responses send back strings of strings. This iterates
        # until proper dict is returned, or if it doesn't make progress.
        num_tries = 1
        response_content = response.content.decode('utf-8')

        while type(response_content) == str and num_tries < 3:
            response_content = json_lib.loads(response_content)
            num_tries += 1

        if type(response_content) == dict:
            return response_content
        else:
            return ValueError("ERROR: got unexpected content from "
                              "the server: {}".format(response_content))
