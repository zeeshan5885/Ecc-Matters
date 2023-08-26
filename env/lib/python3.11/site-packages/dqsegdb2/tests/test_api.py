# DQSEGDB2
# Copyright (C) 2018  Duncan Macleod
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for :mod:`dqsegdb.api`
"""

from .. import api


def test_ifos_path():
    assert api.ifos_path() == "/dq"


def test_flags_path():
    assert api.flags_path("X1") == "/dq/X1"


def test_versions_path():
    assert api.versions_path("X1", "TEST") == "/dq/X1/TEST"


def test_resources_path():
    assert api.resources_path("X1", "TEST", 9) == "/dq/X1/TEST/9"


def test_resources_path_params():
    assert api.resources_path(
        "X1",
        "TEST",
        9,
        s=1234,
        e=5678,
    ) == "/dq/X1/TEST/9?s=1234&e=5678"


def test_metadata_path():
    assert api.metadata_path("X1", "TEST", 9) == "/dq/X1/TEST/9/metadata"


def test_metadata_path_params():
    assert api.metadata_path(
        "X1",
        "TEST",
        9,
        s=1234,
        e=5678,
    ) == "/dq/X1/TEST/9/metadata?s=1234&e=5678"


def test_active_path():
    assert api.active_path("X1", "TEST", 9) == "/dq/X1/TEST/9/active"


def test_active_path_params():
    assert api.active_path(
        "X1",
        "TEST",
        9,
        s=1234,
        e=5678,
    ) == "/dq/X1/TEST/9/active?s=1234&e=5678"


def test_known_path():
    assert api.known_path("X1", "TEST", 9) == "/dq/X1/TEST/9/known"


def test_known_path_params():
    assert api.known_path(
        "X1",
        "TEST",
        9,
        s=1234,
        e=5678,
    ) == "/dq/X1/TEST/9/known?s=1234&e=5678"


def test_insert_history_path():
    assert (
        api.insert_history_path("X1", "TEST", 9)
        == "/dq/X1/TEST/9/insert_history"
    )


def test_insert_history_path_params():
    assert api.insert_history_path(
        "X1",
        "TEST",
        9,
        s=1234,
        e=5678,
    ) == "/dq/X1/TEST/9/insert_history?s=1234&e=5678"


def test_report_path():
    assert api.report_path() == "/report"


def test_report_flags_path():
    assert api.report_flags_path() == "/report/flags"


def test_report_coverage_path():
    assert api.report_coverage_path() == "/report/coverage"


def test_report_db_path():
    assert api.report_db_path() == "/report/db"


def test_report_process_path():
    assert api.report_process_path() == "/report/process"


def test_report_known_path():
    assert api.report_known_path() == "/report/known"


def test_report_known_path_params():
    assert api.report_known_path(
        s=1234,
        e=5678,
    ) == "/report/known?s=1234&e=5678"


def test_report_active_path():
    assert api.report_active_path(
        s=1234,
        e=5678,
    ) == "/report/active?s=1234&e=5678"
