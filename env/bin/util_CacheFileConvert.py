#!/Users/mzeeshan/Desktop/codes/ms-project/env/bin/python3.11

from __future__ import print_function

from glue.lal import Cache
from glue.lal import CacheEntry
import sys

for line in sys.stdin:
    c = CacheEntry.from_T050017(line)
    print(str(c), end=' ')

