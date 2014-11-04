#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint, random

import numpy as np


def random_color():
    """ランダムな色を生成する処理

    :rtype: numpy.ndarray
    """
    return np.array([random() for _ in range(3)], dtype=np.float64)


def random_point():
    """ランダムな座標を生成する処理

    :rtype: tuple
    """
    return randint(-20, 20), randint(-20, 20), randint(15, 50)


def random_points(n):
    """ランダムな座標のリストを生成する処理

    :rtype: list
    """
    return [random_point() for _ in range(n)]


def random_polygons(n):
    """ランダムな座標とポリゴンのリストを生成する処理

    :rtype: tuple
    """
    points = np.array(random_points(3 * n), dtype=np.float64)
    indexes = np.array([tuple(range(3 * i, 3 * i + 3)) for i in range(n)],
                       dtype=np.uint64)
    return points, indexes