#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Camera(object):
    def __init__(self, positon, angle, focus, width, height):
        self.positon = positon
        self.angle = angle
        self.focus = focus
        self.width = width
        self.height = height
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        # カメラ座標系 -> 画像平面の変換行列
        self.array = np.array((
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0 / self.focus, 0.0)
        ))

    def convert_point(self, point):
        """カメラ座標系の座標を画像平面上の座標に変換する処理"""
        vertex = np.append(point, 1)
        p = np.dot(self.array, vertex)
        return np.nan_to_num((self.focus / point[2]) * p)

    def rasterize(self, points):
        """ラスタライズ処理

        ポリゴンをラスタライズして, 点を順に返すイタレータ
        :param points: ポリゴンの3点の座標のリスト
        """

        def to_int(scalar):
            return round(np.asscalar(scalar))

        def make_range(x1, x2):
            if x2 < x1:
                x1, x2 = x2, x1
            return to_int(x1), to_int(x2)

        def make_range_x(x1, x2):
            x1, x2 = make_range(x1, x2)
            return range(max(x1, -self.half_width),
                         min(x2, self.half_width - 1) + 1)

        def make_range_y(y1, y2):
            y1, y2 = make_range(y1, y2)
            return range(max(y1, 1 - self.half_height),
                         min(y2, self.half_height) + 1)

        # 3点を座標変換し, y でソート
        a, b, c = sorted(map(self.convert_point, points), key=lambda p: p[1])
        # d の座標を求める
        r = (b[1] - a[1]) / (c[1] - a[1])
        d = (1 - r) * a + r * c

        for y in make_range_y(a[1], c[1]):
            if y <= to_int(d[1]):
                # 上 (a -> bd)
                s = np.nan_to_num((y - a[1]) / (d[1] - a[1]))
                p = ((1 - s) * a + s * d)
                q = ((1 - s) * a + s * b)
            else:
                # 下 (bd -> c)
                s = (y - c[1]) / (d[1] - c[1])
                p = ((1 - s) * c + s * d)
                q = ((1 - s) * c + s * b)
            for x in make_range_x(p[0], q[0]):
                yield (x, y)