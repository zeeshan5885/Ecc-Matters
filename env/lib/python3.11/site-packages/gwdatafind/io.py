# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2022)
#
# This file is part of GWDataFind.
#
# GWDataFind is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWDataFind is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWDataFind.  If not, see <http://www.gnu.org/licenses/>.

"""I/O (mainly O) routines for GWDataFind.
"""

import os.path
from collections import namedtuple
from operator import attrgetter
from urllib.parse import urlparse

from .utils import filename_metadata

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class LalCacheEntry(
    namedtuple('CacheEntry', ('obs', 'tag', 'segment', 'url')),
):
    """Simplified version of `lal.utils.CacheEntry`.

    This is provided so that we don't have to depend on lalsuite.
    """
    def __str__(self):
        seg = self.segment
        return " ".join(map(str, (
            self.obs,
            self.tag,
            seg[0],
            abs(seg),
            self.url,
        )))

    @classmethod
    def from_url(cls, url, **kwargs):
        """Create a new `LalCacheEntry` from a URL that follows LIGO-T050017.
        """
        obs, tag, seg = filename_metadata(url)
        return cls(obs, tag, seg, url)


def lal_cache(urls):
    """Convert a list of URLs into a LAL cache.

    Returns
    -------
    cache : `list` of `LalCacheEntry`
    """
    return [LalCacheEntry.from_url(url) for url in urls]


class OmegaCacheEntry(namedtuple(
    'OmegaCacheEntry',
    ('obs', 'tag', 'segment', 'duration', 'url')
)):
    """CacheEntry for an omega-style cache.

    Omega-style cache files contain one entry per contiguous directory of
    the form:

        <obs> <tag> <dir-start> <dir-end> <file-duration> <directory>
    """
    def __str__(self):
        return " ".join(map(str, (
            self.obs,
            self.tag,
            self.segment[0],
            self.segment[1],
            self.duration,
            self.url,
        )))


def omega_cache(cache):
    """Convert a list of `LalCacheEntry` into a list of `OmegaCacheEntry`.

    Returns
    -------
    cache : `list` of `OmegaCacheEntry`
    """
    wcache = []
    append = wcache.append
    wentry = None
    for entry in sorted(
        cache,
        key=attrgetter('obs', 'tag', 'segment'),
    ):
        dir_ = os.path.dirname(entry.url)

        # if this file has the same attributes, goes into the same directory,
        # has the same duration, and overlaps with or is contiguous with
        # the last file, just add its segment to the last one:
        if wcache and (
                entry.obs == wentry.obs
                and entry.tag == wentry.tag
                and dir_ == wentry.url
                and abs(entry.segment) == wentry.duration
                and (entry.segment.connects(wentry.segment)
                     or entry.segment.intersects(wentry.segment))
        ):
            wcache[-1] = wentry = OmegaCacheEntry(
                wentry.obs,
                wentry.tag,
                wentry.segment | entry.segment,
                wentry.duration,
                wentry.url,
            )
            continue

        # otherwise create a new entry in the omega wcache
        wentry = OmegaCacheEntry(
            entry.obs,
            entry.tag,
            entry.segment,
            abs(entry.segment),
            dir_,
        )
        append(wentry)
    return wcache


def format_cache(cache, fmt):
    """Format a list of `LalCacheEntry` into a different format.

    Valid formats:

    - ``omega`` - return a list of Omega-format cache entries
    """
    if fmt == "lal":
        return cache

    if fmt == "urls":
        return [e.url for e in cache]

    if fmt == "names":
        return [urlparse(e.url).path for e in cache]

    if fmt == "omega":
        return omega_cache(cache)

    raise ValueError(
        f"invalid format '{fmt}'",
    )
