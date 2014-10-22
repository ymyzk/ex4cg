#cython: language_level=3, boundscheck=False, cdivision=True
# -*- coding: utf-8 -*-

from random import randint

import numpy as np
cimport numpy as np


DOUBLE = np.float64
ctypedef np.float64_t DOUBLE_t

def random_color(DOUBLE_t[:] color, int depth):
    """ランダムな色を生成する処理"""
    cdef int c
    c = 2 ** depth - 1
    _color = np.array([randint(0, c) for _ in range(3)], dtype=DOUBLE)
    color[0] = _color[0]
    color[1] = _color[1]
    color[2] = _color[2]