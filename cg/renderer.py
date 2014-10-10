#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np


class Renderer(object):
    def __init__(self, camera, width, height, zbuffering=True, depth=8,
                 shaders=[]):
        """
        :param cg.camera.Camera camera: カメラ
        :param bool zbuffering: Z バッファを有効にするかどうか
        """
        self.camera = camera
        self.shaders = shaders
        self.depth = depth
        self.width = width
        self.height = height
        self.zbuffering = zbuffering

        self.data = np.zeros((self.height, self.width * 3), dtype=np.uint8)
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

        def calc_z(z1, z2, r):
            return 1 / (r / z1 + (1 - r) / z2)

        def make_range_xz(x1, x2, z1, z2):
            if x2 < x1:
                x1, x2 = x2, x1
            x1, x2 = math.ceil(np.asscalar(x1)), math.floor(np.asscalar(x2))
            if x1 == x2:
                yield x1, z1
                return
            for x in range(max(x1, -self.half_width),
                           min(x2, self.half_width - 1) + 1):
                yield x, calc_z(z1, z2, (x - x1) / (x2 - x1))

        def make_range_y(y1, y2):
            if y2 < y1:
                y1, y2 = y2, y1
            y1 = math.ceil(np.asscalar(y1))
            y2 = math.floor(np.asscalar(y2))
            return range(max(y1, 1 - self.half_height),
                         min(y2, self.half_height) + 1)

        # ポリゴンの3点を座標変換
        a, b, c = map(self.convert_point, polygon)
        # ポリゴンの3点を y でそーと
        if a[1] > b[1]:
            a, b = b, a
        if b[1] > c[1]:
            b, c = c, b
        if a[1] > b[1]:
            a, b = b, a

        # 3点の y 座標が同じであれば処理終了
        if a[1] == c[1]:
            return

        # d の座標を求める
        r = (b[1] - a[1]) / (c[1] - a[1])
        d = (1 - r) * a + r * c

        for y in make_range_y(a[1], c[1]):
            # x の左右を探す:
            if y <= b[1]:
                # a -> bd
                if a[1] == b[1]:
                    continue
                s = (y - a[1]) / (b[1] - a[1])
                p = (1 - s) * a + s * b
                q = (1 - s) * a + s * d
                pz = calc_z(a[3], b[3], 1 - s)
                qz = calc_z(a[3], d[3], 1 - s)
            else:
                # 下 bd -> c
                if b[1] == c[1]:
                    continue
                s = (y - c[1]) / (b[1] - c[1])
                p = (1 - s) * c + s * b
                q = (1 - s) * c + s * d
                pz = calc_z(c[3], b[3], 1 - s)
                qz = calc_z(c[3], d[3], 1 - s)
            # x についてループ
            for x, z in make_range_xz(p[0], q[0], pz, qz):
                yield x, y, z

    def _draw_polygon(self, polygon):
        # TODO: 飽和演算の実装の改善
        color = np.zeros(3, dtype=np.float)
        for shader in self.shaders:
            color += shader.calc(polygon)
        if 255 < color[0]:
            color[0] = 255
        if 255 < color[1]:
            color[1] = 255
        if 255 < color[2]:
            color[2] = 255
        color = color.astype(np.uint8)
        for x, y, z in self.rasterize(polygon):
            # Z バッファでテスト
            if not self.zbuffering or z <= self.zbuffer[y][x]:
                # X: -128 ~ 127 -> (x + 128) -> 0 ~ 255
                # Y: -127 ~ 128 -> (128 - y) -> 0 ~ 255
                # TODO: サンプル画像がおかしいので X を反転して表示
                data_x = 3 * (self.width - 1 - (self.half_width + x))
                data_y = self.half_height - y
                self.data[data_y][data_x:data_x + 3] = color
                self.zbuffer[y][x] = z

    def draw_polygons(self, points, indexes):
        for index in indexes:
            self._draw_polygon(tuple((map(lambda i: points[i], index))))