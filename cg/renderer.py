#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math

import numpy as np

from cg.shader import ShadingMode


DOUBLE = np.float64


class Renderer(object):
    def __init__(self, camera, width, height, z_buffering=True, depth=8,
                 shaders=tuple(), shading_mode=ShadingMode.flat):
        """
        :param cg.camera.Camera camera: カメラ
        :param bool z_buffering: Z バッファを有効にするかどうか
        """
        self.camera = camera
        self.shaders = shaders
        self.depth = depth
        self.width = width
        self.height = height
        self.z_buffering = z_buffering
        self.shading_mode = shading_mode

        self.data = np.zeros((self.height, self.width * 3), dtype=np.uint8)
        self.z_buffer = np.empty((self.height, self.width), dtype=DOUBLE)
        self.z_buffer.fill(float('inf'))
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._depth = 2 ** depth - 1

    def _convert_point(self, point):
        """カメラ座標系の座標を画像平面上の座標に変換する処理

        画像平面の x, y, z + 元の座標の z
        """
        p = np.dot(self.camera.array, point)
        converted = (self.camera.focus / point[2]) * p
        converted[2] = point[2]
        return converted

    def _shade_vertex(self, polygon, normal):
        """シェーディング処理"""
        return sum([s.calc(polygon, normal) for s in self.shaders])

    def make_range_x(self, x1, x2):
        x1 = int(math.ceil(x1))
        x2 = int(math.floor(x2))
        return range(max(x1, -self.half_width),
                     min(x2, self.half_width - 1) + 1)

    def make_range_y(self, y1, y2):
        y1 = int(math.ceil(y1))
        y2 = int(math.floor(y2))
        return range(max(y1, 1 - self.half_height),
                     min(y2, self.half_height) + 1)

    def draw_pixel(self, x, y, z, cl):
        """画素を描画する処理"""
        # Z バッファでテスト
        if not self.z_buffering or z <= self.z_buffer[y][x]:
            # 飽和
            if 1.0 < cl[0]:
                cl[0] = 1.0
            if 1.0 < cl[1]:
                cl[1] = 1.0
            if 1.0 < cl[2]:
                cl[2] = 1.0
            cl = (cl * self._depth).astype(np.uint8)

            # X: -128 ~ 127 -> (x + 128) -> 0 ~ 255
            # Y: -127 ~ 128 -> (128 - y) -> 0 ~ 255
            # NOTE: サンプル画像がおかしいので X を反転して表示している
            # data_x = 3 * (self.half_width + x) # 反転させないコード
            data_x = 3 * (self.half_width - x - 1)
            data_y = self.half_height - y
            self.data[data_y][data_x:data_x + 3] = cl
            self.z_buffer[y][x] = z

    def _draw_polygon_flat(self, polygon, normal):
        """ポリゴンを描画する処理
        フラット (コンスタント) シェーディング

        :param np.ndarray polygon: ポリゴン 3x3
        :param np.ndarray normal: 法線ベクトル
        """
        # ポリゴンの3点を座標変換
        a = self._convert_point(polygon[0])
        b = self._convert_point(polygon[1])
        c = self._convert_point(polygon[2])

        # ポリゴンの3点を y でソート
        if a[1] < b[1]:
            if c[1] < a[1]:
                a, c = c, a
        else:
            if b[1] < c[1]:
                a, b = b, a
            else:
                a, c = c, a
        if c[1] < b[1]:
            b, c = c, b

        # 3点の y 座標が同じであれば処理終了
        if a[1] == c[1]:
            return

        # d の座標を求める
        r = (b[1] - a[1]) / (c[1] - a[1])
        d = (1 - r) * a + r * c
        d[2] = a[2] * c[2] / (r * a[2] + (1 - r) * c[2])

        # ポリゴン全体を1色でシェーディング
        color = self._shade_vertex(polygon, normal)

        for y in self.make_range_y(a[1], c[1]):
            # x の左右を探す:
            if y <= b[1]:
                # a -> bd
                if a[1] == b[1]:
                    continue
                s = (y - a[1]) / (b[1] - a[1])
                px = ((1 - s) * a[0] + s * b[0])
                qx = ((1 - s) * a[0] + s * d[0])
                pz = a[2] * b[2] / (s * a[2] + (1 - s) * b[2])
                qz = a[2] * d[2] / (s * a[2] + (1 - s) * d[2])
            else:
                # bd -> c
                if b[1] == c[1]:
                    continue
                s = (y - c[1]) / (b[1] - c[1])
                px = ((1 - s) * c[0] + s * b[0])
                qx = ((1 - s) * c[0] + s * d[0])
                pz = c[2] * b[2] / (s * c[2] + (1 - s) * b[2])
                qz = c[2] * d[2] / (s * c[2] + (1 - s) * d[2])
            # x についてループ
            if px == qx:
                # x が同じの時はすぐに終了
                self.draw_pixel(int(px), y, pz, color)
                continue
            elif px > qx:
                # x についてソート
                px, qx = qx, px
            for x in self.make_range_x(px, qx):
                r = (x - px) / (qx - px)
                self.draw_pixel(x, y, pz * qz / (r * qz + (1 - r) * pz), color)

    def _draw_polygon_gouraud(self, polygon, normals):
        """ポリゴンを描画する処理
        グーローシェーディング

        :param np.ndarray polygon: ポリゴン 3x3
        :param np.ndarray normals: 法線ベクトル 3x3
        """
        # ポリゴンの3点を座標変換
        a = self._convert_point(polygon[0])
        b = self._convert_point(polygon[1])
        c = self._convert_point(polygon[2])
        an, bn, cn = normals

        # ポリゴンの3点を y でソート
        if a[1] < b[1]:
            if c[1] < a[1]:
                a, c = c, a
                an, cn = cn, an
        else:
            if b[1] < c[1]:
                a, b = b, a
                an, bn = bn, an
            else:
                a, c = c, a
                an, cn = cn, an
        if c[1] < b[1]:
            b, c = c, b
            bn, cn = cn, bn

        # 3点の y 座標が同じであれば処理終了
        if a[1] == c[1]:
            return

        # d の座標を求める
        r = (b[1] - a[1]) / (c[1] - a[1])
        d = (1 - r) * a + r * c
        d[2] = a[2] * c[2] / (r * a[2] + (1 - r) * c[2])
        dn = (1 - r) * an + r * cn

        # 頂点をそれぞれの法線ベクトルでシェーディング
        ac = self._shade_vertex(polygon, an)
        bc = self._shade_vertex(polygon, bn)
        cc = self._shade_vertex(polygon, cn)
        dc = self._shade_vertex(polygon, dn)
        for y in self.make_range_y(a[1], c[1]):
            # x の左右を探す:
            if y <= b[1]:
                # a -> bd
                if a[1] == b[1]:
                    continue
                s = (y - a[1]) / (b[1] - a[1])
                px = ((1 - s) * a[0] + s * b[0])
                qx = ((1 - s) * a[0] + s * d[0])
                pc = ((1 - s) * ac + s * bc)
                qc = ((1 - s) * ac + s * dc)
                pz = a[2] * b[2] / (s * a[2] + (1 - s) * b[2])
                qz = a[2] * d[2] / (s * a[2] + (1 - s) * d[2])
            else:
                # 下 bd -> c
                if b[1] == c[1]:
                    continue
                s = (y - c[1]) / (b[1] - c[1])
                px = ((1 - s) * c[0] + s * b[0])
                qx = ((1 - s) * c[0] + s * d[0])
                pc = ((1 - s) * cc + s * bc)
                qc = ((1 - s) * cc + s * dc)
                pz = c[2] * b[2] / (s * c[2] + (1 - s) * b[2])
                qz = c[2] * d[2] / (s * c[2] + (1 - s) * d[2])
            # x についてループ
            if px == qx:
                # x が同じの時はすぐに終了
                self.draw_pixel(int(px), y, pz, pc)
                continue
            elif px > qx:
                # x についてソート
                pc, qc = qc, pc
                px, qx = qx, px
            for x in self.make_range_x(px, qx):
                r = (x - px) / (qx - px)
                rc = ((1 - r) * pc + r * qc)
                self.draw_pixel(x, y, pz * qz / (r * qz + (1 - r) * pz), rc)

    def _draw_polygon_phong(self, polygon, normals):
        """ポリゴンを描画する処理
        フォンシェーディング

        :param np.ndarray polygon: ポリゴン 3x3
        :param np.ndarray normals: 法線ベクトル 3x3
        """
        # ポリゴンの3点を座標変換
        a = self._convert_point(polygon[0])
        b = self._convert_point(polygon[1])
        c = self._convert_point(polygon[2])
        an, bn, cn = normals

        # ポリゴンの3点を y でソート
        if a[1] < b[1]:
            if c[1] < a[1]:
                a, c = c, a
                an, cn = cn, an
        else:
            if b[1] < c[1]:
                a, b = b, a
                an, bn = bn, an
            else:
                a, c = c, a
                an, cn = cn, an
        if c[1] < b[1]:
            b, c = c, b
            bn, cn = cn, bn

        # 3点の y 座標が同じであれば処理終了
        if a[1] == c[1]:
            return

        # d の座標を求める
        r = (b[1] - a[1]) / (c[1] - a[1])
        d = (1 - r) * a + r * c
        d[2] = a[2] * c[2] / (r * a[2] + (1 - r) * c[2])
        dn = (1 - r) * an + r * cn

        for y in self.make_range_y(a[1], c[1]):
            # x の左右を探す:
            if y <= b[1]:
                # a -> bd
                if a[1] == b[1]:
                    continue
                s = (y - a[1]) / (b[1] - a[1])
                px = ((1 - s) * a[0] + s * b[0])
                qx = ((1 - s) * a[0] + s * d[0])
                pn = ((1 - s) * an + s * bn)
                qn = ((1 - s) * an + s * dn)
                pz = a[2] * b[2] / (s * a[2] + (1 - s) * b[2])
                qz = a[2] * d[2] / (s * a[2] + (1 - s) * d[2])
            else:
                # 下 bd -> c
                if b[1] == c[1]:
                    continue
                s = (y - c[1]) / (b[1] - c[1])
                px = ((1 - s) * c[0] + s * b[0])
                qx = ((1 - s) * c[0] + s * d[0])
                pn = ((1 - s) * cn + s * bn)
                qn = ((1 - s) * cn + s * dn)
                pz = c[2] * b[2] / (s * c[2] + (1 - s) * b[2])
                qz = c[2] * d[2] / (s * c[2] + (1 - s) * d[2])
            # x についてループ
            if px == qx:
                # x が同じの時はすぐに終了
                self.draw_pixel(int(px), y, pz,
                                self._shade_vertex(polygon, pn))
                continue
            elif px > qx:
                # x についてソート
                pn, qn = qn, pn
                px, qx = qx, px
            for x in self.make_range_x(px, qx):
                r = (x - px) / (qx - px)
                rn = ((1 - r) * pn + r * qn)
                z = pz * qz / (r * qz + (1 - r) * pz)
                self.draw_pixel(x, y, z, self._shade_vertex(polygon, rn))

    def draw_polygons(self, points, indexes):
        # ポリゴンのリストを作成
        polygons = np.array([[points[i] for i in j] for j in indexes],
                            dtype=DOUBLE)

        def calc_normal(polygon):
            """ポリゴンの面の法線ベクトルを求める処理"""
            # 直交ベクトル (時計回りを表)
            a = polygon[2] - polygon[1]
            b = polygon[1] - polygon[0]
            cross = np.array((
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]
            ), dtype=DOUBLE)
            # 直交ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
            if np.count_nonzero(cross) == 0:
                return np.zeros(3, dtype=DOUBLE)
            else:
                # 法線ベクトル
                return cross / np.linalg.norm(cross)

        # すべてのポリゴンの面の法線ベクトルを求める
        polygon_normals = [calc_normal(p) for p in polygons]

        if self.shading_mode is ShadingMode.flat:
            for i in range(len(polygons)):
                self._draw_polygon_flat(polygons[i], polygon_normals[i])
        else:
            # 各頂点の法線ベクトルのリストを作成
            vertexes = [[] for _ in range(len(points))]
            for i, index in enumerate(indexes):
                normal = polygon_normals[i]
                vertexes[index[0]].append(normal)
                vertexes[index[1]].append(normal)
                vertexes[index[2]].append(normal)

            # 各頂点の法線ベクトルを, 面法線ベクトルの平均として求める
            def mean(vertex):
                if 0 < len(vertex):
                    return np.array(sum(vertex) / len(vertex), dtype=DOUBLE)
                else:
                    return np.zeros(3, dtype=DOUBLE)
            vertex_normals = [mean(vertex) for vertex in vertexes]

            # ポリゴンの各頂点の法線ベクトルのリストを作成
            polygon_vertex_normals = [[vertex_normals[i] for i in j]
                                      for j in indexes]

            # ポリゴンを描画
            if self.shading_mode is ShadingMode.gouraud:
                for i in range(len(polygons)):
                    self._draw_polygon_gouraud(polygons[i],
                                               polygon_vertex_normals[i])
            elif self.shading_mode is ShadingMode.phong:
                for i in range(len(polygons)):
                    self._draw_polygon_phong(polygons[i],
                                             polygon_vertex_normals[i])