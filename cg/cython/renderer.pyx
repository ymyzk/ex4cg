#cython: language_level=3, boundscheck=False, cdivision=True
# -*- coding: utf-8 -*-

from libc.math cimport ceil, floor, sqrt
import numpy as np
cimport numpy as np

from cg.shader import ShadingMode


DOUBLE = np.float64
UINT = np.uint8
ctypedef np.float64_t DOUBLE_t
ctypedef np.uint8_t UINT_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

cdef class Renderer:
    cdef object camera, shaders, shading_mode
    cdef int depth, width, height, half_width, half_height, z_buffering
    cdef readonly np.ndarray data
    cdef np.ndarray z_buffer
    cdef DOUBLE_t focus

    def __init__(self, camera, int width, int height, z_buffering=True,
                 int depth=8, shaders=tuple(), shading_mode=ShadingMode.flat):
        """
        :param cg.camera.Camera camera: カメラ
        :param bool z_buffering: Z バッファを有効にするかどうか
        """
        self.camera = camera
        self.shaders = shaders
        # self.shaders = [s.calc for s in shaders]
        self.depth = depth
        self.width = width
        self.height = height
        self.z_buffering = z_buffering
        self.shading_mode = shading_mode

        self.data = np.zeros((self.height, self.width * 3), dtype=UINT)
        self.z_buffer = np.empty((self.height, self.width), dtype=DOUBLE)
        self.z_buffer.fill(float('inf'))
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self.focus = self.camera.focus

    cdef void _convert_point(self, DOUBLE_t[:] point):
        """カメラ座標系の座標を画像平面上の座標に変換する処理

        画像平面の x, y, z + 元の座標の z
        """
        cdef DOUBLE_t z_ip
        z_ip = self.focus / point[2]
        point[0] = z_ip * point[0]
        point[1] = z_ip * point[1]

    @cython.profile(True)
    cdef void _shade_vertex_internal(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                     DOUBLE_t[:] c, DOUBLE_t[:] n,
                                     DOUBLE_t[:] color):
        """シェーディング処理"""
        cdef int i
        cdef DOUBLE_t _cl[3]
        cdef DOUBLE_t[:] cl = _cl

        color[0] = 0.0
        color[1] = 0.0
        color[2] = 0.0

        # シェーディング
        for i in range(len(self.shaders)):
            self.shaders[i].calc(a, b, c, n, cl)
            # self.shaders[i](a, b, c, n, cl)
            color[0] += cl[0]
            color[1] += cl[1]
            color[2] += cl[2]

        return

    cdef void _draw_pixel(self, int x, int y, DOUBLE_t z, DOUBLE_t[:] cl):
        """画素を描画する処理"""
        cdef int data_x, data_y

        # Z バッファでテスト
        if not self.z_buffering or z <= self.z_buffer[y,x]:
            # 飽和
            if 255 < cl[0]:
                cl[0] = 255
            if 255 < cl[1]:
                cl[1] = 255
            if 255 < cl[2]:
                cl[2] = 255

            # X: -128 ~ 127 -> (x + 128) -> 0 ~ 255
            # Y: -127 ~ 128 -> (128 - y) -> 0 ~ 255
            # NOTE: サンプル画像がおかしいので X を反転して表示している
            # data_x = 3 * (self.half_width + x) # 反転させないコード
            data_x = 3 * (self.half_width - x - 1)
            data_y = self.half_height - y
            self.data[data_y][data_x+0] = <UINT_t>cl[0]
            self.data[data_y][data_x+1] = <UINT_t>cl[1]
            self.data[data_y][data_x+2] = <UINT_t>cl[2]
            self.z_buffer[y][x] = z

    cdef void _draw_polygon_flat(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                  DOUBLE_t[:] c, DOUBLE_t[:] n):
        cdef int x, y
        cdef DOUBLE_t [3] d
        cdef DOUBLE_t px, qx, pz, qz, r, s
        cdef DOUBLE_t [3] color

        # ポリゴンの3点を座標変換
        self._convert_point(a)
        self._convert_point(b)
        self._convert_point(c)

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

        # ポリゴン全体を1色でシェーディング
        self._shade_vertex_internal(a, b, c, n, color)

        # d の座標を求める
        r = (b[1] - a[1]) / (c[1] - a[1])
        d[0] = (1 - r) * a[0] + r * c[0]
        d[1] = (1 - r) * a[1] + r * c[1]
        d[2] = (1 - r) * a[2] + r * c[2]

        for y in range(int_max(<int>ceil(a[1]), 1 - self.half_height),
                       int_min(<int>floor(c[1]), self.half_height) + 1):
            # x の左右を探す:
            if y <= b[1]:
                # a -> bd
                if a[1] == b[1]:
                    continue
                s = (y - a[1]) / (b[1] - a[1])
                px = ((1 - s) * a[0] + s * b[0])
                qx = ((1 - s) * a[0] + s * d[0])
                pz = 1 / ((1 - s) / a[2] + s / b[2])
                qz = 1 / ((1 - s) / a[2] + s / d[2])
            else:
                # bd -> c
                if b[1] == c[1]:
                    continue
                s = (y - c[1]) / (b[1] - c[1])
                px = ((1 - s) * c[0] + s * b[0])
                qx = ((1 - s) * c[0] + s * d[0])
                pz = 1 / ((1 - s) / c[2] + s / b[2])
                qz = 1 / ((1 - s) / c[2] + s / d[2])
            # x についてループ
            if px == qx:
                # x が同じの時はすぐに終了
                self._draw_pixel(<int>px, y, pz, color)
                continue
            elif px > qx:
                # x についてソート
                px, qx = qx, px
            for x in range(int_max(<int>ceil(px), 1 - self.half_width),
                           int_min(<int>floor(qx), self.half_width) + 1):
                r = (x - px) / (qx - px)
                self._draw_pixel(x, y, 1 / (r / pz + (1 - r) / qz), color)

    cdef void __draw_polygon_gouraud(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                     DOUBLE_t[:] c, DOUBLE_t[:] an,
                                     DOUBLE_t[:] bn, DOUBLE_t[:] cn):
        cdef int x, y
        cdef DOUBLE_t[3] d, dn
        cdef DOUBLE_t px, qx, pz, qz, r, s
        cdef DOUBLE_t [3] ac, bc, cc, dc, pc, qc, rc

        # ポリゴンの3点を座標変換
        self._convert_point(a)
        self._convert_point(b)
        self._convert_point(c)

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
        d[0] = (1 - r) * a[0] + r * c[0]
        d[1] = (1 - r) * a[1] + r * c[1]
        d[2] = (1 - r) * a[2] + r * c[2]
        dn[0] = (1 - r) * an[0] + r * cn[0]
        dn[1] = (1 - r) * an[1] + r * cn[1]
        dn[2] = (1 - r) * an[2] + r * cn[2]

        # 頂点をそれぞれの法線ベクトルでシェーディング
        self._shade_vertex_internal(a, b, c, an, ac)
        self._shade_vertex_internal(a, b, c, bn, bc)
        self._shade_vertex_internal(a, b, c, cn, cc)
        self._shade_vertex_internal(a, b, c, dn, dc)

        for y in range(int_max(<int>ceil(a[1]), 1 - self.half_height),
                       int_min(<int>floor(c[1]), self.half_height) + 1):
            # x の左右を探す:
            if y <= b[1]:
                # a -> bd
                if a[1] == b[1]:
                    continue
                s = (y - a[1]) / (b[1] - a[1])
                px = ((1 - s) * a[0] + s * b[0])
                qx = ((1 - s) * a[0] + s * d[0])
                pc[0] = ((1 - s) * ac[0] + s * bc[0])
                pc[1] = ((1 - s) * ac[1] + s * bc[1])
                pc[2] = ((1 - s) * ac[2] + s * bc[2])
                qc[0] = ((1 - s) * ac[0] + s * dc[0])
                qc[1] = ((1 - s) * ac[1] + s * dc[1])
                qc[2] = ((1 - s) * ac[2] + s * dc[2])
                pz = 1 / ((1 - s) / a[2] + s / b[2])
                qz = 1 / ((1 - s) / a[2] + s / d[2])
            else:
                # bd -> c
                if b[1] == c[1]:
                    continue
                s = (y - c[1]) / (b[1] - c[1])
                px = ((1 - s) * c[0] + s * b[0])
                qx = ((1 - s) * c[0] + s * d[0])
                pc[0] = ((1 - s) * cc[0] + s * bc[0])
                pc[1] = ((1 - s) * cc[1] + s * bc[1])
                pc[2] = ((1 - s) * cc[2] + s * bc[2])
                qc[0] = ((1 - s) * cc[0] + s * dc[0])
                qc[1] = ((1 - s) * cc[1] + s * dc[1])
                qc[2] = ((1 - s) * cc[2] + s * dc[2])
                pz = 1 / ((1 - s) / c[2] + s / b[2])
                qz = 1 / ((1 - s) / c[2] + s / d[2])
            # x についてループ
            if px == qx:
                # x が同じの時はすぐに終了
                self._draw_pixel(<int>px, y, pz, pc)
                continue
            elif px < qx:
                for x in range(int_max(<int>ceil(px), 1 - self.half_width),
                               int_min(<int>floor(qx), self.half_width) + 1):
                    r = (x - px) / (qx - px)
                    rc[0] = ((1 - r) * pc[0] + r * qc[0])
                    rc[1] = ((1 - r) * pc[1] + r * qc[1])
                    rc[2] = ((1 - r) * pc[2] + r * qc[2])
                    self._draw_pixel(x, y, 1 / (r / pz + (1 - r) / qz), rc)
            else:
                for x in range(int_max(<int>ceil(qx), 1 - self.half_width),
                               int_min(<int>floor(px), self.half_width) + 1):
                    r = (x - qx) / (px - qx)
                    rc[0] = ((1 - r) * qc[0] + r * pc[0])
                    rc[1] = ((1 - r) * qc[1] + r * pc[1])
                    rc[2] = ((1 - r) * qc[2] + r * pc[2])
                    self._draw_pixel(x, y, 1 / (r / pz + (1 - r) / qz), rc)

    def _draw_polygon_gouraud(self, np.ndarray[DOUBLE_t, ndim=2] polygon,
                              np.ndarray[DOUBLE_t, ndim=2] normals):
        """ポリゴンを描画する処理
        グーローシェーディング

        :param np.ndarray polygon: ポリゴン 3x3
        :param np.ndarray normals: 法線ベクトル 3x3
        """
        cdef DOUBLE_t[:] a = polygon[0]
        cdef DOUBLE_t[:] b = polygon[1]
        cdef DOUBLE_t[:] c = polygon[2]
        cdef DOUBLE_t[:] an = normals[0]
        cdef DOUBLE_t[:] bn = normals[1]
        cdef DOUBLE_t[:] cn = normals[2]

        self.__draw_polygon_gouraud(a, b, c, an, bn, cn)

    cdef void __draw_polygon_phong(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                   DOUBLE_t[:] c, DOUBLE_t[:] an,
                                   DOUBLE_t[:] bn, DOUBLE_t[:] cn):
        cdef int x, y
        cdef DOUBLE_t[3] d, dn, pn, qn, rn
        cdef DOUBLE_t px, qx, pz, qz, r, s, z
        cdef DOUBLE_t[3] color

        # ポリゴンの3点を座標変換
        self._convert_point(a)
        self._convert_point(b)
        self._convert_point(c)

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
        d[0] = (1 - r) * a[0] + r * c[0]
        d[1] = (1 - r) * a[1] + r * c[1]
        d[2] = (1 - r) * a[2] + r * c[2]
        dn[0] = (1 - r) * an[0] + r * cn[0]
        dn[1] = (1 - r) * an[1] + r * cn[1]
        dn[2] = (1 - r) * an[2] + r * cn[2]

        for y in range(int_max(<int>ceil(a[1]), 1 - self.half_height),
                       int_min(<int>floor(c[1]), self.half_height) + 1):
            # x の左右を探す:
            if y <= b[1]:
                # a -> bd
                if a[1] == b[1]:
                    continue
                s = (y - a[1]) / (b[1] - a[1])
                px = ((1 - s) * a[0] + s * b[0])
                qx = ((1 - s) * a[0] + s * d[0])
                pn[0] = ((1 - s) * an[0] + s * bn[0])
                pn[1] = ((1 - s) * an[1] + s * bn[1])
                pn[2] = ((1 - s) * an[2] + s * bn[2])
                qn[0] = ((1 - s) * an[0] + s * dn[0])
                qn[1] = ((1 - s) * an[1] + s * dn[1])
                qn[2] = ((1 - s) * an[2] + s * dn[2])
                pz = 1 / ((1 - s) / a[2] + s / b[2])
                qz = 1 / ((1 - s) / a[2] + s / d[2])
            else:
                # bd -> c
                if b[1] == c[1]:
                    continue
                s = (y - c[1]) / (b[1] - c[1])
                px = ((1 - s) * c[0] + s * b[0])
                qx = ((1 - s) * c[0] + s * d[0])
                pn[0] = ((1 - s) * cn[0] + s * bn[0])
                pn[1] = ((1 - s) * cn[1] + s * bn[1])
                pn[2] = ((1 - s) * cn[2] + s * bn[2])
                qn[0] = ((1 - s) * cn[0] + s * dn[0])
                qn[1] = ((1 - s) * cn[1] + s * dn[1])
                qn[2] = ((1 - s) * cn[2] + s * dn[2])
                pz = 1 / ((1 - s) / c[2] + s / b[2])
                qz = 1 / ((1 - s) / c[2] + s / d[2])
            # x についてループ
            if px == qx:
                # x が同じの時はすぐに終了
                self._shade_vertex_internal(a, b, c, pn, color)
                self._draw_pixel(<int>px, y, pz, color)
                continue
            elif px < qx:
                for x in range(int_max(<int>ceil(px), 1 - self.half_width),
                               int_min(<int>floor(qx), self.half_width) + 1):
                    r = (x - px) / (qx - px)
                    rn[0] = ((1 - r) * pn[0] + r * qn[0])
                    rn[1] = ((1 - r) * pn[1] + r * qn[1])
                    rn[2] = ((1 - r) * pn[2] + r * qn[2])
                    z = 1 / (r / pz + (1 - r) / qz)
                    self._shade_vertex_internal(a, b, c, rn, color)
                    self._draw_pixel(x, y, z, color)
            else:
                for x in range(int_max(<int>ceil(qx), 1 - self.half_width),
                               int_min(<int>floor(px), self.half_width) + 1):
                    r = (x - qx) / (px - qx)
                    rn[0] = ((1 - r) * qn[0] + r * pn[0])
                    rn[1] = ((1 - r) * qn[1] + r * pn[1])
                    rn[2] = ((1 - r) * qn[2] + r * pn[2])
                    z = 1 / (r / pz + (1 - r) / qz)
                    self._shade_vertex_internal(a, b, c, rn, color)
                    self._draw_pixel(x, y, z, color)

    def _draw_polygon_phong(self, np.ndarray[DOUBLE_t, ndim=2] polygon,
                            np.ndarray[DOUBLE_t, ndim=2] normals):
        """ポリゴンを描画する処理
        フォンシェーディング

        :param np.ndarray polygon: ポリゴン 3x3
        :param np.ndarray normals: 法線ベクトル 3x3
        """
        cdef DOUBLE_t[:] a = polygon[0]
        cdef DOUBLE_t[:] b = polygon[1]
        cdef DOUBLE_t[:] c = polygon[2]
        cdef DOUBLE_t[:] an = normals[0]
        cdef DOUBLE_t[:] bn = normals[1]
        cdef DOUBLE_t[:] cn = normals[2]

        self.__draw_polygon_phong(a, b, c, an, bn, cn)

    cdef calc_normal(self, np.ndarray[DOUBLE_t] a, np.ndarray[DOUBLE_t] b,
                     np.ndarray[DOUBLE_t] c, np.ndarray[DOUBLE_t] n):
        """ポリゴンの面の法線ベクトルを求める処理"""
        cdef DOUBLE_t[3] ab, bc, cr

        ab[0] = b[0] - a[0]
        ab[1] = b[1] - a[1]
        ab[2] = b[2] - a[2]
        bc[0] = c[0] - b[0]
        bc[1] = c[1] - b[1]
        bc[2] = c[2] - b[2]

        # 直交ベクトル (時計回りを表)
        cr[0] = bc[1] * ab[2] - bc[2] * ab[1]
        cr[1] = bc[2] * ab[0] - bc[0] * ab[2]
        cr[2] = bc[0] * ab[1] - bc[1] * ab[0]

        # 直交ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        if cr[0] == 0.0 and cr[1] == 0.0 and cr[2] == 0.0:
            n[0] = 0.0
            n[1] = 0.0
            n[2] = 0.0
        else:
            # 法線ベクトル (単位ベクトル化)
            norm = sqrt(cr[0] ** 2 + cr[1] ** 2 + cr[2] ** 2)
            n[0] = cr[0] / norm
            n[1] = cr[1] / norm
            n[2] = cr[2] / norm

    def draw_polygonsa(self, list points, list indexes):
        cdef np.ndarray normals, polygons, polygon_vertex_normals
        cdef np.ndarray[DOUBLE_t, ndim=2] polygon
        cdef np.ndarray[DOUBLE_t] normal
        cdef list vertexes, vertex_normals
        cdef int i, l


        # ポリゴンのリストを作成
        polygons = np.array([[points[i] for i in j] for j in indexes],
                            dtype=DOUBLE)

        # ポリゴンの数
        l = polygons.shape[0]

        # すべてのポリゴンの面の法線ベクトルを求める
        polygon_normals = np.empty((l, 3))
        for i in range(l):
            polygon = polygons[i]
            normal = polygon_normals[i]
            self.calc_normal(polygon[0], polygon[1], polygon[2], normal)

        if self.shading_mode is ShadingMode.flat:
            for i in range(len(polygons)):
                self._draw_polygon_flat(polygons[i][0], polygons[i][1],
                                        polygons[i][2], polygon_normals[i])
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
            polygon_vertex_normals = np.array(
                [[vertex_normals[i] for i in j] for j in indexes],
                dtype=DOUBLE)

            # ポリゴンを描画
            if self.shading_mode is ShadingMode.gouraud:
                for i in range(len(polygons)):
                    self.__draw_polygon_gouraud(polygons[i][0], polygons[i][1], polygons[i][2],
                                               polygon_vertex_normals[i][0],
                                               polygon_vertex_normals[i][1],
                                               polygon_vertex_normals[i][2])
            elif self.shading_mode is ShadingMode.phong:
                for i in range(len(polygons)):
                    self._draw_polygon_phong(polygons[i],
                                             polygon_vertex_normals[i])

    def draw_polygons(self, list points, list indexes):
        cdef np.ndarray normal, normals, polygons, polygon_vertex_normals
        cdef list vertexes, polygon_normals, vertex_normals
        cdef int i

        # ポリゴンのリストを作成
        polygons = np.array([[points[i] for i in j] for j in indexes],
                            dtype=DOUBLE)

        def calc_normal(np.ndarray[DOUBLE_t, ndim=2] polygon):
            """ポリゴンの面の法線ベクトルを求める処理"""
            cdef np.ndarray[DOUBLE_t] a, b, cross

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
                # 法線ベクトル (単位ベクトル化)
                norm = sqrt(cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2)
                cross[0] /= norm
                cross[1] /= norm
                cross[2] /= norm
                return cross

        # すべてのポリゴンの面の法線ベクトルを求める
        polygon_normals = [calc_normal(p) for p in polygons]

        if self.shading_mode is ShadingMode.flat:
            for i in range(len(polygons)):
                self._draw_polygon_flat(polygons[i][0], polygons[i][1],
                                        polygons[i][2], polygon_normals[i])
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
            polygon_vertex_normals = np.array(
                [[vertex_normals[i] for i in j] for j in indexes],
                dtype=DOUBLE)

            # ポリゴンを描画
            if self.shading_mode is ShadingMode.gouraud:
                for i in range(len(polygons)):
                    self.__draw_polygon_gouraud(polygons[i][0], polygons[i][1], polygons[i][2],
                                               polygon_vertex_normals[i][0],
                                               polygon_vertex_normals[i][1],
                                               polygon_vertex_normals[i][2])
            elif self.shading_mode is ShadingMode.phong:
                for i in range(len(polygons)):
                    self._draw_polygon_phong(polygons[i],
                                             polygon_vertex_normals[i])