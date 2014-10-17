#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint

import numpy as np


def random_color(depth):
    """ランダムな色を生成する処理

    :rtype: numpy.ndarray
    """
    color = 2 ** depth - 1
    return np.array([randint(0, color) for _ in range(3)], dtype=np.float64)


def random_point():
    """ランダムな座標を生成する処理

    :rtype: numpy.ndarray
    """
    return np.array((randint(-20, 20), randint(-20, 20), randint(15, 50)),
                    dtype=np.float64)


def random_points(n):
    """ランダムな座標のリストを生成する処理

    :rtype: list
    """
    return [random_point() for _ in range(n)]


def random_polygons(n):
    """ランダムな座標とポリゴンのリストを生成する処理

    :rtype: tuple
    """
    points = random_points(3 * n)
    indexes = [tuple(range(3 * i, 3 * i + 3)) for i in range(n)]
    return points, indexes