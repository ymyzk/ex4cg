#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import print_function, unicode_literals
import json
from pprint import pprint
import sys


def analyze(files):
    profile = {
        'num_files': len(files),
        'vrml': 0.0,
        'prepare': 0.0,
        'draw': 0.0,
        'ppm': 0.0,
        'total': 0.0
    }
    print('{0:40s} {1:>8s} {2:>8s} {3:>9s} {4:>7s} {5:>9s}'.format(
        'File', 'VRML', 'Prepare', 'Draw', 'PPM', 'Total'))
    print('=' * 86)
    for file in files:
        with open(file) as f:
            d = json.load(f)
            print(('{0:40s} {vrml:8.2f} {prepare:8.2f} {draw:9.2f} {ppm:7.2f} '
                   '{total:9.2f}').format(file, **d))
            for k in d:
                if k in profile:
                    profile[k] += d[k]
    print('-' * 86)
    print(('{0:40s} {vrml:8.2f} {prepare:8.2f} {draw:9.2f} {ppm:7.2f} '
           '{total:9.2f}').format('Total', **profile))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(1)
    analyze(sys.argv[1:])
    sys.exit(0)