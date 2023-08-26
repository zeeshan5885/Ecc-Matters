# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2018-2022)
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

"""Tests for :mod:`gwdatafind.http`.
"""

import json
import warnings
from unittest import mock
from urllib.error import HTTPError

import pytest

from ligo.segments import (segment, segmentlist)

from ..http import (
    HTTPConnection,
    HTTPSConnection,
)
from . import yield_fixture


def fake_response(output, status=200):
    resp = mock.Mock()
    resp.status = int(status)
    if not isinstance(output, (str, bytes)):
        output = json.dumps(output)
    resp.read.return_value = output.encode('utf-8')
    return resp


class TestHTTPConnection(object):
    CONNECTION = HTTPConnection

    @classmethod
    def setup_class(cls):
        cls._create_connection_patch = mock.patch('socket.create_connection')
        cls._create_connection_patch.start()

    @classmethod
    def teardown_class(cls):
        cls._create_connection_patch.stop()

    @classmethod
    @yield_fixture
    def connection(cls):
        with mock.patch.dict(
            "os.environ",
            {"GWDATAFIND_SERVER": "test.gwdatafind.com:123"},
        ):
            with pytest.warns(DeprecationWarning):
                yield cls.CONNECTION()

    def test_init(self, connection):
        assert connection.host == 'test.gwdatafind.com'
        assert connection.port == 123

    def test_get_json(self, response, connection):
        response.return_value = fake_response({'test': 1})
        jdata = connection.get_json('something')
        assert jdata['test'] == 1

    def test_ping(self, response, connection):
        response.return_value = fake_response('')
        assert connection.ping() == 0
        response.return_value = fake_response('', 500)
        with pytest.raises(HTTPError):
            connection.ping()

    @pytest.mark.parametrize('match, out', [
        (None, ['A', 'B', 'C', 'D', 'ABCD']),
        ('B', ['B', 'ABCD']),
    ])
    def test_find_observatories(self, response, connection, match, out):
        response.return_value = fake_response(['A', 'B', 'C', 'D', 'ABCD'])
        assert sorted(connection.find_observatories(match=match)) == (
            sorted(out))

    @pytest.mark.parametrize('site, match, out', [
        (None, None, ['A', 'B', 'C', 'D', 'ABCD']),
        ('X', 'B', ['B', 'ABCD']),
    ])
    def test_find_types(self, response, connection, site, match, out):
        response.return_value = fake_response(['A', 'B', 'C', 'D', 'ABCD'])
        assert sorted(connection.find_types(match=match)) == (
            sorted(out))

    def test_find_times(self, response, connection):
        segs = [(0, 1), (1, 2), (3, 4)]
        response.return_value = fake_response(segs)
        times = connection.find_times('X', 'test')
        assert isinstance(times, segmentlist)
        assert isinstance(times[0], segment)
        assert times == segs

        # check keywords
        times = connection.find_times('X', 'test', 0, 10)
        assert times == segs

    def test_find_url(self, response, connection):
        out = ['file:///tmp/X-test-0-10.gwf']
        response.return_value = fake_response(out)
        url = connection.find_url('X-test-0-10.gwf')
        assert url == out

        response.return_value = fake_response([])
        with pytest.raises(RuntimeError):
            connection.find_url('X-test-0-10.gwf')
        with pytest.warns(UserWarning):
            url = connection.find_url('X-test-0-10.gwf', on_missing='warn')
            assert url == []
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            url = connection.find_url('X-test-0-10.gwf', on_missing='ignore')
        assert url == []

    def test_find_frame(self, response, connection):
        out = ['file:///tmp/X-test-0-10.gwf']
        response.return_value = fake_response(out)
        with pytest.warns(DeprecationWarning):
            url = connection.find_frame('X-test-0-10.gwf')
        assert url == out

    def test_find_latest(self, response, connection):
        out = ['file:///tmp/X-test-0-10.gwf']
        response.return_value = fake_response(out)
        url = connection.find_latest('X', 'test')
        assert url == out

        response.return_value = fake_response([])
        with pytest.raises(RuntimeError):
            connection.find_latest('X', 'test')
        with pytest.warns(UserWarning):
            url = connection.find_latest('X', 'test', on_missing='warn')
            assert url == []
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            url = connection.find_latest('X', 'test', on_missing='ignore')
        assert url == []

    def test_find_urls(self, response, connection):
        files = [
            'file:///tmp/X-test-0-10.gwf',
            'file:///tmp/X-test-10-10.gwf',
            'file:///tmp/X-test-20-10.gwf',
        ]
        response.return_value = fake_response(files)
        urls = connection.find_urls('X', 'test', 0, 30, match='anything')
        assert urls == files

        # check gaps
        with pytest.raises(RuntimeError):
            connection.find_urls('X', 'test', 0, 40, on_gaps='error')
        with pytest.warns(UserWarning):
            urls = connection.find_urls('X', 'test', 0, 40, on_gaps='warn')
        assert urls == files
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            urls = connection.find_urls('X', 'test', 0, 40, on_gaps='ignore')
        assert urls == files

    def test_find_frame_urls(self, response, connection):
        files = [
            'file:///tmp/X-test-0-10.gwf',
            'file:///tmp/X-test-10-10.gwf',
            'file:///tmp/X-test-20-10.gwf',
        ]
        response.return_value = fake_response(files)
        with pytest.warns(DeprecationWarning):
            urls = connection.find_frame_urls('X', 'test', 0, 30)
        assert urls == files


class TestHTTPSConnection(TestHTTPConnection):
    CONNECTION = HTTPSConnection
