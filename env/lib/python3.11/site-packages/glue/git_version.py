# This file is part of the Grid LSC User Environment (GLUE)
#
# GLUE is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

version = '3.0.1'

# git information
id = 'ad4229f87c381ec0332f9c7f08b254275adc9442'
date = '2022-01-22 17:58:12 +0000'
branch = 'glue_3.0.1'
tag = 'glue-release-3.0.1'
author = 'Robert Bruntz <robert.bruntz@ligo.org>'
builder = 'Robert Bruntz <robert.bruntz@ligo.org>'
committer = 'Robert Bruntz <robert.bruntz@ligo.org>'
status = 'CLEAN: All modifications committed'
verbose_msg = """Branch: glue_3.0.1
Tag: glue-release-3.0.1
Id: ad4229f87c381ec0332f9c7f08b254275adc9442

Builder: Robert Bruntz <robert.bruntz@ligo.org>
Build date: 2022-01-22 18:41:47 +0000
Repository status: CLEAN: All modifications committed"""

import warnings

class VersionMismatchError(ValueError):
    pass

def check_match(foreign_id, onmismatch="raise"):
    """
    If foreign_id != id, perform an action specified by the onmismatch
    kwarg. This can be useful for validating input files.

    onmismatch actions:
      "raise": raise a VersionMismatchError, stating both versions involved
      "warn": emit a warning, stating both versions involved
    """
    if onmismatch not in ("raise", "warn"):
        raise ValueError(onmismatch + " is an unrecognized value of onmismatch")
    if foreign_id == 'ad4229f87c381ec0332f9c7f08b254275adc9442':
        return
    msg = "Program id (ad4229f87c381ec0332f9c7f08b254275adc9442 does not match given id (%s)." % foreign_id
    if onmismatch == "raise":
        raise VersionMismatchError(msg)

    # in the backtrace, show calling code
    warnings.warn(msg, UserWarning)

