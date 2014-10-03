#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import randint

import numpy as np


def random_color(depth):
    """ランダムな色を生成する処理"""
    return [randint(0, 2 ** depth - 1) for _ in range(3)]


def random_point():
    """ランダムな座標を生成する処理"""
    return np.array((randint(-20, 20), randint(-20, 20), randint(15, 50)))


def random_points(n):
    """ランダムな座標のリストを生成する処理"""
    return [random_point() for _ in range(n)]


def random_polygons(n):
    """ランダムな座標とポリゴンのリストを生成する処理"""
    points = random_points(3 * n)
    polygons = [tuple(range(3 * i, 3 * (i + 1))) for i in range(n)]
    return points, polygons


def sample_polygons_1():
    """サンプルデータ1"""
    points = (
        (0.0, 0.0, 20.0),
        (6.0, 3.0, 17.0),
        (3.0, 6.0, 23.0),
        (-6.0, 3.0, 23.0),
        (-3.0, 6.0, 17.0),
        (-6.0, -3.0, 17.0),
        (-3.0, -6.0, 23.0),
        (6.0, -3.0, 23.0),
        (3.0, -6.0, 17.0)
    )
    points = list(map(np.array, points))

    polygons = (
        (0, 1, 2, -1),
        (0, 3, 4, -1),
        (0, 5, 6, -1),
        (0, 7, 8, -1)
    )

    return points, polygons