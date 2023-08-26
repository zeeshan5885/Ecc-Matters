# -*- coding: utf-8 -*-
# DQSEGDB2
# Copyright (C) 2022 Cardiff University
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

"""Utilities for dqsegdb2.
"""

import os

DEFAULT_SEGMENT_SERVER_ENV = "DEFAULT_SEGMENT_SERVER"

# set initial default
DEFAULT_SEGMENT_SERVER = os.environ.setdefault(
    DEFAULT_SEGMENT_SERVER_ENV,
    "https://segments.ligo.org",
)


def get_default_host():
    """Return the default host as stored in the ``${DEFAULT_SEGMENT_SERVER}``
    environment variable.
    """
    return os.environ[DEFAULT_SEGMENT_SERVER_ENV]
