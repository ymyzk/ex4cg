#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum
import math
import numpy as np


class ShadingMode(Enum):
    flat = 0
    gouraud = 1


class Renderer(object):
    def __init__(self, camera, width, height, zbuffering=True, depth=8,
                 shaders=tuple(), shading_mode=ShadingMode.flat):
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
        self.shading_mode = shading_mode

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
        # ポリゴンの3点を y でソート
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

    def _draw_polygon(self, polygon, normals):
        # TODO: 飽和演算の実装の改善
        color = np.zeros(3, dtype=np.float)
        if self.shading_mode is ShadingMode.flat:
            for shader in self.shaders:
                color += shader.calc_flat(polygon, normals[0])
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

    def _polygons_normal(self, points, indexes):
        """ポリゴンの法線ベクトルのリストを作成する処理"""
        normals = []
        for index in indexes:
            polygon = list((map(lambda i: points[i], index)))
            polygon.append(index)
            # 直交ベクトル (時計回りを表)
            cross = np.cross(polygon[2] - polygon[1], polygon[1] - polygon[0])
            # 直交ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
            if np.count_nonzero(cross) == 0:
                normals.append(np.zeros(3))
            else:
                # 法線ベクトル
                normals.append(cross / np.linalg.norm(cross))

        return normals

    def draw_polygons(self, points, indexes):
        # ポリゴンの法線ベクトルを求める
        polygon_normals = self._polygons_normal(points, indexes)

        if self.shading_mode is ShadingMode.flat:
            for index, normal in zip(indexes, polygon_normals):
                # ポリゴンの3点の座標
                verticies = tuple((map(lambda i: points[i], index)))
                # ポリゴンの面の法線ベクトル
                normals = (normal,)
                self._draw_polygon(verticies, normals)
        elif self.shading_mode is ShadingMode.gouraud:
            # 各頂点の法線ベクトルを集計
            verticies = [[] for _ in range(len(points))]
            for i, index in enumerate(indexes):
                normal = polygon_normals[i]
                verticies[index[0]].append(normal)
                verticies[index[1]].append(normal)
                verticies[index[2]].append(normal)

            # 各頂点の法線ベクトルを, 面法線ベクトルの平均として求める
            vertex_normals = []
            for vertex in verticies:
                vertex_normals.append(np.array(sum(vertex) / len(vertex)))

            for index in indexes:
                # ポリゴンの3点の座標
                verticies = tuple((map(lambda i: points[i], index)))
                # ポリゴンの3点の法線ベクトル
                normals = tuple((map(lambda i: vertex_normals[i], index)))
                self._draw_polygon(verticies, normals)