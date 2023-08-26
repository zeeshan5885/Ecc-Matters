# -*- coding: utf-8 -*-
# DQSEGDB2
# Copyright (C) 2018,2020  Duncan Macleod
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

"""A simplified Python implementation of the DQSEGDB API.
"""

from .query import (
    query_ifos,
    query_names,
    query_versions,
    query_segments,
)
from .requests import Session

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # development mode
    __version__ = 'dev'

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Ryan Fisher, Gary Hemming'
