#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from cg.utils import random_color


class Renderer(object):
    def __init__(self, camera, width, height, depth=8, shader=None):
        """
        :param cg.camera.Camera camera: カメラ
        """
        self.camera = camera
        self.shader = shader
        self.depth = depth
        self.width = width
        self.height = height

        self.data = np.zeros((self.height, self.width * 3), dtype=np.int8)
        self.zbuffer = np.empty((self.height, self.width), dtype=np.float64)
        self.zbuffer.fill(float('inf'))
        self.half_width = self.width // 2
        self.half_height = self.height // 2

    def convert_point(self, point):
        """カメラ座標系の座標を画像平面上の座標に変換する処理

        画像平面の x, y, z + 元の座標の z
        """
        vertex = np.append(point, 1)
        p = np.dot(self.camera.array, vertex)
        converted = (self.camera.focus / point[2]) * p
        converted[3] = point[2]
        return converted

    def rasterize(self, polygon):
        """ラスタライズ処理

        ポリゴンをラスタライズして, 点を順に返すイタレータ
        :param polygon: ポリゴン
        """

        def make_range(r1, r2):
            if r2 < r1:
                r1, r2 = r2, r1
            return math.ceil(np.asscalar(r1)), math.floor(np.asscalar(r2))

        def make_range_xz(p1, p2):
            if p2[0] < p1[0]:
                p1, p2 = p2, p1
            x1, x2 = make_range(p1[0], p2[0])
            z1, z2 = p1[3], p2[3]
            if x1 == x2:
                return x1, z1
            for x in range(max(x1, -self.half_width),
                           min(x2, self.half_width - 1) + 1):
                yield x, z1 + (x - p1[0]) / (x2 - p1[0]) * (z2 - z1)

        def make_range_y(p1, p2):
            y1, y2 = make_range(p1[1], p2[1])
            return range(max(y1, 1 - self.half_height),
                         min(y2, self.half_height) + 1)

        # 3点を座標変換し, y でソート
        a, b, c = sorted(map(self.convert_point, polygon), key=lambda p: p[1])

        # 3点の y 座標が同じであれば処理終了
        if a[1] == c[1]:
            return

        # d の座標を求める
        r = (b[1] - a[1]) / (c[1] - a[1])
        d = (1 - r) * a + r * c

        for y in make_range_y(a, c):
            # x の左右を探す:
            if y <= b[1]:
                # a -> bd
                if a[1] == b[1]:
                    continue
                s = (y - a[1]) / (b[1] - a[1])
                p = (1 - s) * a + s * b
                q = (1 - s) * a + s * d
            else:
                # 下 bd -> c
                if b[1] == c[1]:
                    continue
                s = (y - c[1]) / (b[1] - c[1])
                p = (1 - s) * c + s * b
                q = (1 - s) * c + s * d
            # x についてループ
            for x, z in make_range_xz(p, q):
                yield x, y, z

    def rasterize_b(self, polygon):
        """Bresenham's algorithm test"""
        def to_int(scalar):
            return round(np.asscalar(scalar))

        vt1, vt2, vt3 = tuple(map(self.convert_point, polygon))

        min_x = to_int(min(vt1[0], vt2[0], vt3[0]))
        max_x = to_int(max(vt1[0], vt2[0], vt3[0]))
        min_y = to_int(min(vt1[1], vt2[1], vt3[1]))
        max_y = to_int(max(vt1[1], vt2[1], vt3[1]))

        vs1 = vt2[:2] - vt1[:2]
        vs2 = vt3[:2] - vt1[:2]

        for y in range(max(min_y, 1 - self.half_height),
                       min(max_y, self.half_height) + 1):
            for x in range(max(min_x, -self.half_width),
                           min(max_x, self.half_width - 1) + 1):
                q = np.array((x - vt1[0], y - vt1[1]))
                s = np.cross(q, vs2) / np.cross(vs1, vs2)
                t = np.cross(vs1, q) / np.cross(vs1, vs2)
                if 0 <= s and 0 <= t and s + t <= 1:
                    yield x, y, 1.0

    def draw_polygon(self, polygon):
        if self.shader is None:
            color = random_color(8)
        else:
            color = self.shader.calc(polygon)
        for x, y, z in self.rasterize(polygon):
            # Z バッファでテスト
            if z <= self.zbuffer[y][x]:
                # X: -128 ~ 127 -> (x + 128) -> 0 ~ 255
                # Y: -127 ~ 128 -> (128 - y) -> 0 ~ 255
                data_x = 3 * (self.half_width + x)
                data_y = self.half_height - y
                self.data[data_y][data_x:data_x + 3] = color
                self.zbuffer[y][x] = z