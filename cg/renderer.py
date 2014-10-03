#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from cg.utils import random_color


class Renderer(object):
    def __init__(self, camera, width, height, depth=8):
        """
        :param cg.camera.Camera camera: カメラ
        """
        self.camera = camera
        self.depth = depth
        self.width = width
        self.height = height

        self.data = np.zeros((self.height, self.width * 3),
                             dtype=np.int8)
        self.half_width = self.width // 2
        self.half_height = self.height // 2

    def convert_point(self, point):
        """カメラ座標系の座標を画像平面上の座標に変換する処理"""
        vertex = np.append(point, 1)
        p = np.dot(self.camera.array, vertex)
        return np.nan_to_num((self.camera.focus / point[2]) * p)

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

    def draw_polygon(self, polygon):
        color = random_color(self.depth)
        for point in self.rasterize(polygon[:3]):
            # X: -128 ~ 127 -> (x + 128) -> 0 ~ 255
            # Y: -127 ~ 128 -> (128 - y) -> 0 ~ 255
            data_x = 3 * (self.half_width + point[0])
            data_y = self.half_height - point[1]
            self.data[data_y][data_x:data_x+3] = color