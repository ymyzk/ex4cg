#cython: language_level=3, boundscheck=False, cdivision=True
# -*- coding: utf-8 -*-

from random import random

from libc.math cimport ceil, floor, sqrt
from libc.math cimport INFINITY
from libc.stdlib cimport free, malloc
import numpy as np
cimport numpy as np

from cg.shader import (AmbientShader, DiffuseShader, RandomColorShader,
                       ShadingMode, SpecularShader)


DOUBLE = np.float64
UINT8 = np.uint8
UINT64 = np.uint64
ctypedef np.float64_t DOUBLE_t
ctypedef np.uint8_t UINT8_t
ctypedef np.uint64_t UINT64_t


cdef inline int int_max(int a, int b) nogil:
    return a if a >= b else b


cdef inline int int_min(int a, int b) nogil:
    return a if a <= b else b


cdef void calc_polygon_normals(DOUBLE_t[:,:,:] polygons,
                               DOUBLE_t *polygon_normals) nogil:
    """ポリゴンの面の法線ベクトルを求める処理

    :param polygons: ポリゴンの配列 (n x 3 x 3)
    :param polygon_normals: ポリゴンの面の法線ベクトルを格納する配列 (n x 3)
    """
    cdef DOUBLE_t[:,:] polygon
    cdef DOUBLE_t[:] a, b, c
    cdef DOUBLE_t[3] ab, bc, cr
    cdef DOUBLE_t norm
    cdef int i, j, l

    l = polygons.shape[0]

    for i in range(l):
        polygon = polygons[i]
        a = polygon[0]
        b = polygon[1]
        c = polygon[2]
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
        # if cr[0] == 0 and cr[1] == 0 and cr[2] == 0:
        #     return np.zeros(3, dtype=DOUBLE)

        # 法線ベクトル (単位ベクトル化)
        norm = sqrt(cr[0] ** 2 + cr[1] ** 2 + cr[2] ** 2)
        j = 3 * i
        polygon_normals[j + 0] = cr[0] / norm
        polygon_normals[j + 1] = cr[1] / norm
        polygon_normals[j + 2] = cr[2] / norm


cdef void calc_vertex_normals(UINT64_t[:,:] indexes,
                              DOUBLE_t *polygon_normals,
                              DOUBLE_t *vertex_normals, int num) nogil:
    """各頂点の法線ベクトルを計算する処理

    各頂点の法線ベクトルは, その頂点を含む全ての面の法線ベクトルの平均である

    :param indexes: ポリゴンのインデックス
    :param polygon_normals: ポリゴンの面の法線ベクトルの配列 (n x 3)
    :param vertex_normals: 頂点の法線ベクトルの配列 (m x 3)
    :param num: vertex_normals の数
    """
    cdef DOUBLE_t[:] vertex
    cdef DOUBLE_t *vertexes
    cdef UINT64_t[:] index
    cdef UINT64_t *vertexes_n
    cdef UINT64_t vertex_n
    cdef int i, j, k, l

    l = num

    # メモリ確保
    vertexes = <DOUBLE_t *>malloc(sizeof(DOUBLE_t) * l * 3)
    vertexes_n = <UINT64_t *>malloc(sizeof(UINT64_t) * l)
    for i in range(l * 3):
        vertexes[i] = 0.0
    for i in range(l):
        vertexes_n[i] = 0

    # 各頂点を含む面の法線ベクトルの和を求める
    l = indexes.shape[0]
    for i in range(l):
        index = indexes[i]
        k = 3 * i

        j = 3 * index[0]
        vertexes[j + 0] += polygon_normals[k + 0]
        vertexes[j + 1] += polygon_normals[k + 1]
        vertexes[j + 2] += polygon_normals[k + 2]
        vertexes_n[index[0]] += 1

        j = 3 * index[1]
        vertexes[j + 0] += polygon_normals[k + 0]
        vertexes[j + 1] += polygon_normals[k + 1]
        vertexes[j + 2] += polygon_normals[k + 2]
        vertexes_n[index[1]] += 1

        j = 3 * index[2]
        vertexes[j + 0] += polygon_normals[k + 0]
        vertexes[j + 1] += polygon_normals[k + 1]
        vertexes[j + 2] += polygon_normals[k + 2]
        vertexes_n[index[2]] += 1

    # 各頂点の法線ベクトルの平均値を求める
    l = num
    for i in range(l):
        j = 3 * i
        vertex_n = vertexes_n[i]
        if 0 < vertex_n:
            vertex_normals[j + 0] = vertexes[j + 0] / vertex_n
            vertex_normals[j + 1] = vertexes[j + 1] / vertex_n
            vertex_normals[j + 2] = vertexes[j + 2] / vertex_n
        else:
            vertex_normals[j + 0] = 0.0
            vertex_normals[j + 1] = 0.0
            vertex_normals[j + 2] = 0.0

    # メモリ解放
    free(vertexes)
    free(vertexes_n)


cdef class Renderer:
    cdef public object shading_mode
    cdef object camera, shaders
    cdef int depth, width, height, half_width, half_height, z_buffering
    cdef int _depth
    cdef readonly np.ndarray data
    cdef UINT8_t[:,:] _data
    cdef DOUBLE_t *_z_buffer
    cdef DOUBLE_t camera_array[3][4]
    cdef DOUBLE_t camera_position[3]
    cdef DOUBLE_t focus

    # Shading - Ambient
    cdef int _is_ambient_shader_enabled
    cdef DOUBLE_t[3] _ambient_shade
    # Shading - Diffuse
    cdef int _is_diffuse_shader_enabled
    cdef DOUBLE_t[3] _diffuse_direction, _diffuse_pre_shade
    # Shading - Random
    cdef int _is_random_shader_enabled
    # Shading - Specular
    cdef int _is_specular_shader_enabled
    cdef DOUBLE_t _specular_shininess
    cdef DOUBLE_t[3] _specular_direction, _specular_pre_shade

    # ポリゴンのキャッシュ用
    cdef DOUBLE_t[:,:,:] _polygons
    cdef DOUBLE_t[:,:] _points
    cdef DOUBLE_t *_polygon_normals
    cdef DOUBLE_t *_vertex_normals
    cdef UINT64_t[:,:] _indexes

    def __init__(self, int width, int height, z_buffering=True,
                 int depth=8, shading_mode=ShadingMode.flat):
        """
        :param bool z_buffering: Z バッファを有効にするかどうか
        """
        cdef np.ndarray z_buffer

        self.depth = depth
        self.width = width
        self.height = height
        self.z_buffering = z_buffering
        self.shading_mode = shading_mode

        self.data = np.zeros((self.height, self.width * 3), dtype=UINT8)
        self._data = self.data
        self._z_buffer = <DOUBLE_t *>malloc(sizeof(DOUBLE_t) *
                                            self.height * self.width)
        for i in range(self.height * self.width):
            self._z_buffer[i] = INFINITY
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._depth = 2 ** depth - 1

    def __del__(self):
        free(self._z_buffer)
        free(self._polygon_normals)
        free(self._vertex_normals)

    property shaders:
        def __get__(self):
            return self.shaders

        def __set__(self, value):
            cdef int i

            self.shaders = value

            # Python で書かれたシェーダーから必要な値を取り出す
            for shader in self.shaders:
                if isinstance(shader, AmbientShader):
                    self._is_ambient_shader_enabled = 1
                    for i in range(3):
                        self._ambient_shade[i] = shader.shade[i]
                elif isinstance(shader, DiffuseShader):
                    self._is_diffuse_shader_enabled = 1
                    for i in range(3):
                        self._diffuse_direction[i] = shader.direction[i]
                        self._diffuse_pre_shade[i] = shader.pre_shade[i]
                elif isinstance(shader, RandomColorShader):
                    self._is_random_shader_enabled = 1
                elif isinstance(shader, SpecularShader):
                    self._is_specular_shader_enabled = 1
                    self._specular_shininess = shader.shininess
                    for i in range(3):
                        self._specular_direction[i] = shader.direction[i]
                        self._specular_pre_shade[i] = shader.pre_shade[i]

    property camera:
        def __get__(self):
            return self.camera

        def __set__(self, value):
            cdef int i, j

            self.camera = value
            self.focus = self.camera.focus
            for i in range(3):
                for j in range(4):
                    self.camera_array[i][j] = self.camera.array[i][j]
                self.camera_position[i] = self.camera.position[i]

    cdef void _convert_point(self, DOUBLE_t[:] point):
        """カメラ座標系の座標を画像平面上の座標に変換する処理

        画像平面の x, y, z
        """
        cdef DOUBLE_t[3] p
        cdef DOUBLE_t k
        cdef int i

        for i in range(3):
            p[i] = (self.camera_array[i][0] * point[0]
                    + self.camera_array[i][1] * point[1]
                    + self.camera_array[i][2] * point[2]
                    + self.camera_array[i][3])
        k = self.focus / p[2]
        point[0] = k * p[0]
        point[1] = k * p[1]
        point[2] = p[2]

    cdef void _shade_ambient(self, DOUBLE_t *cl) nogil:
        cl[0] += self._ambient_shade[0]
        cl[1] += self._ambient_shade[1]
        cl[2] += self._ambient_shade[2]

    cdef void _shade_diffuse(self, DOUBLE_t *n, DOUBLE_t *cl) nogil:
        cdef DOUBLE_t cos

        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        # TODO: 現状では実行されない
        # if n[0] == 0.0 and n[1] == 0.0 and n[2] == 0.0:
        #     return

        # 反射光を計算
        cos = (self._diffuse_direction[0] * n[0]
               + self._diffuse_direction[1] * n[1]
               + self._diffuse_direction[2] * n[2])

        # ポリゴンが裏を向いているときは, 反射光なし
        if 0.0 < cos:
            return

        cl[0] += -cos * self._diffuse_pre_shade[0]
        cl[1] += -cos * self._diffuse_pre_shade[1]
        cl[2] += -cos * self._diffuse_pre_shade[2]

    cdef void _shade_random(self, DOUBLE_t *cl):
        """ランダムな色をつけるシェーダ"""
        cl[0] += random()
        cl[1] += random()
        cl[2] += random()

    cdef void _shade_specular(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                              DOUBLE_t[:] c, DOUBLE_t *n, DOUBLE_t *cl) nogil:
        cdef DOUBLE_t[3] e, s
        cdef DOUBLE_t sn, norm

        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        # TODO: 現状では実行されない
        # if n[0] == 0.0 and n[1] == 0.0 and n[2] == 0.0:
        #     return

        # ポリゴンの重心
        # g = (polygon[0] + polygon[1] + polygon[2]) / 3

        # ポリゴンから視点への単位方向ベクトル
        e[0] = self.camera_position[0] - a[0]
        e[1] = self.camera_position[1] - a[1]
        e[2] = self.camera_position[2] - a[2]
        norm = sqrt(e[0] ** 2 + e[1] ** 2 + e[2] ** 2)
        e[0] /= norm
        e[1] /= norm
        e[2] /= norm

        s[0] = e[0] - self._specular_direction[0]
        s[1] = e[1] - self._specular_direction[1]
        s[2] = e[2] - self._specular_direction[2]
        norm = sqrt(s[0] ** 2 + s[1] ** 2 + s[2] ** 2)
        s[0] /= norm
        s[1] /= norm
        s[2] /= norm

        sn = s[0] * n[0] + s[1] * n[1] + s[2] * n[2]

        # ポリゴンが裏を向いているときは, 反射光なし
        if sn <= 0.0:
            return

        sn **= self._specular_shininess
        cl[0] += sn * self._specular_pre_shade[0]
        cl[1] += sn * self._specular_pre_shade[1]
        cl[2] += sn * self._specular_pre_shade[2]

    cdef void _shade_vertex(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                            DOUBLE_t[:] c, DOUBLE_t *n, DOUBLE_t *color) nogil:
        """シェーディング処理"""
        color[0] = 0.0
        color[1] = 0.0
        color[2] = 0.0

        if self._is_ambient_shader_enabled == 1:
            self._shade_ambient(color)

        if self._is_diffuse_shader_enabled == 1:
            self._shade_diffuse(n, color)

        if self._is_random_shader_enabled == 1:
            with gil:
                self._shade_random(color)

        if self._is_specular_shader_enabled == 1:
            self._shade_specular(a, b, c, n, color)

    cdef void _draw_pixel(self, int x, int y, DOUBLE_t z, DOUBLE_t *cl) nogil:
        """画素を描画する処理"""
        cdef int xy

        # X: -128 ~ 127 -> (x + 128) -> 0 ~ 255
        # Y: -127 ~ 128 -> (128 - y) -> 0 ~ 255
        # NOTE: サンプル画像がおかしいので X を反転して表示している
        # data_x = self.half_width + x # 反転させないコード
        x = self.half_width - x - 1
        y = self.half_height - y
        xy = y * self.width + x

        # Z バッファでテスト
        if not self.z_buffering or z <= self._z_buffer[xy]:
            # 飽和
            if 1.0 < cl[0]:
                cl[0] = 1.0
            if 1.0 < cl[1]:
                cl[1] = 1.0
            if 1.0 < cl[2]:
                cl[2] = 1.0

            self._data[y][3 * x + 0] = <UINT8_t>(cl[0] * self._depth)
            self._data[y][3 * x + 1] = <UINT8_t>(cl[1] * self._depth)
            self._data[y][3 * x + 2] = <UINT8_t>(cl[2] * self._depth)
            self._z_buffer[xy] = z

    cdef void _draw_polygon_flat(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                 DOUBLE_t[:] c, DOUBLE_t *n):
        """ポリゴンを描画する処理 (フラットシェーディング)"""
        cdef int x, y
        cdef DOUBLE_t px, qx, pz, qz, r, s
        cdef DOUBLE_t[3] d
        cdef DOUBLE_t color[3]

        # ポリゴン全体を1色でシェーディング
        self._shade_vertex(a, b, c, n, color)

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

        # d の座標を求める
        r = (b[1] - a[1]) / (c[1] - a[1])
        d[0] = (1 - r) * a[0] + r * c[0]
        d[1] = (1 - r) * a[1] + r * c[1]
        d[2] = a[2] * c[2] / (r * a[2] + (1 - r) * c[2])

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
                x = <int>px
                if -self.half_width <= x <= self.half_width - 1:
                    # x が同じの時はすぐに終了
                    self._draw_pixel(x, y, pz, color)
                continue
            elif px > qx:
                # x についてソート
                px, qx = qx, px
            for x in range(int_max(<int>ceil(px), -self.half_width),
                           int_min(<int>floor(qx) + 1, self.half_width)):
                r = (x - px) / (qx - px)
                self._draw_pixel(x, y, pz * qz / (r * qz + (1 - r) * pz),
                                 color)

    cdef void _draw_polygon_gouraud(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                    DOUBLE_t[:] c, DOUBLE_t *an, DOUBLE_t *bn,
                                    DOUBLE_t *cn):
        """ポリゴンを描画する処理 (グーローシェーディング)"""
        cdef int x, y
        cdef DOUBLE_t[3] d, dn
        cdef DOUBLE_t px, qx, pz, qz, r, s
        cdef DOUBLE_t[3] _a, _b, _c
        cdef DOUBLE_t ac[3]
        cdef DOUBLE_t bc[3]
        cdef DOUBLE_t cc[3]
        cdef DOUBLE_t dc[3]
        cdef DOUBLE_t pc[3]
        cdef DOUBLE_t qc[3]
        cdef DOUBLE_t rc[3]

        # 座標変換前の座標を保存
        _a[0] = a[0]
        _a[1] = a[1]
        _a[2] = a[2]
        _b[0] = b[0]
        _b[1] = b[1]
        _b[2] = b[2]
        _c[0] = c[0]
        _c[1] = c[1]
        _c[2] = c[2]

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
        d[2] = a[2] * c[2] / (r * a[2] + (1 - r) * c[2])
        dn[0] = (1 - r) * an[0] + r * cn[0]
        dn[1] = (1 - r) * an[1] + r * cn[1]
        dn[2] = (1 - r) * an[2] + r * cn[2]

        # 頂点をそれぞれの法線ベクトルでシェーディング
        self._shade_vertex(_a, _b, _c, an, ac)
        self._shade_vertex(_a, _b, _c, bn, bc)
        self._shade_vertex(_a, _b, _c, cn, cc)
        self._shade_vertex(_a, _b, _c, dn, dc)

        for y in range(int_max(<int>ceil(a[1]), 1 - self.half_height),
                       int_min(<int>floor(c[1]) + 1, self.half_height)):
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
                pz = a[2] * b[2] / (s * a[2] + (1 - s) * b[2])
                qz = a[2] * d[2] / (s * a[2] + (1 - s) * d[2])
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
                pz = c[2] * b[2] / (s * c[2] + (1 - s) * b[2])
                qz = c[2] * d[2] / (s * c[2] + (1 - s) * d[2])
            # x についてループ
            if px == qx:
                x = <int>px
                if -self.half_width <= x <= self.half_width - 1:
                    # x が同じの時はすぐに終了
                    self._draw_pixel(x, y, pz, pc)
                continue
            elif px < qx:
                for x in range(int_max(<int>ceil(px), -self.half_width),
                               int_min(<int>floor(qx) + 1, self.half_width)):
                    r = (x - px) / (qx - px)
                    rc[0] = ((1 - r) * pc[0] + r * qc[0])
                    rc[1] = ((1 - r) * pc[1] + r * qc[1])
                    rc[2] = ((1 - r) * pc[2] + r * qc[2])
                    self._draw_pixel(x, y, pz * qz / (r * qz + (1 - r) * pz), rc)
            else:
                for x in range(int_max(<int>ceil(qx), -self.half_width),
                               int_min(<int>floor(px) + 1, self.half_width)):
                    r = (x - qx) / (px - qx)
                    rc[0] = ((1 - r) * qc[0] + r * pc[0])
                    rc[1] = ((1 - r) * qc[1] + r * pc[1])
                    rc[2] = ((1 - r) * qc[2] + r * pc[2])
                    self._draw_pixel(x, y, pz * qz / (r * qz + (1 - r) * pz), rc)

    cdef void _draw_polygon_phong(self,
                                  DOUBLE_t[:] a, DOUBLE_t[:] b, DOUBLE_t[:] c,
                                  DOUBLE_t *an, DOUBLE_t *bn, DOUBLE_t *cn):
        """ポリゴンを描画する処理 (フォンシェーディング)"""
        cdef int x, y
        cdef DOUBLE_t[3] _a, _b, _c, d, dn, pn, qn, rn
        cdef DOUBLE_t px, qx, pz, qz, r, s, z
        cdef DOUBLE_t color[3]

        # 座標変換前の座標を保存
        _a[0] = a[0]
        _a[1] = a[1]
        _a[2] = a[2]
        _b[0] = b[0]
        _b[1] = b[1]
        _b[2] = b[2]
        _c[0] = c[0]
        _c[1] = c[1]
        _c[2] = c[2]

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
        d[2] = a[2] * c[2] / (r * a[2] + (1 - r) * c[2])
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
                pz = a[2] * b[2] / (s * a[2] + (1 - s) * b[2])
                qz = a[2] * d[2] / (s * a[2] + (1 - s) * d[2])
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
                pz = c[2] * b[2] / (s * c[2] + (1 - s) * b[2])
                qz = c[2] * d[2] / (s * c[2] + (1 - s) * d[2])
            # x についてループ
            if px == qx:
                x = <int>px
                if -self.half_width <= x <= self.half_width - 1:
                    # x が同じの時はすぐに終了
                    self._shade_vertex(_a, _b, _c, pn, color)
                    self._draw_pixel(x, y, pz, color)
                continue
            elif px < qx:
                for x in range(int_max(<int>ceil(px), -self.half_width),
                               int_min(<int>floor(qx) + 1, self.half_width)):
                    r = (x - px) / (qx - px)
                    rn[0] = ((1 - r) * pn[0] + r * qn[0])
                    rn[1] = ((1 - r) * pn[1] + r * qn[1])
                    rn[2] = ((1 - r) * pn[2] + r * qn[2])
                    z = pz * qz / (r * qz + (1 - r) * pz)
                    self._shade_vertex(_a, _b, _c, rn, color)
                    self._draw_pixel(x, y, z, color)
            else:
                for x in range(int_max(<int>ceil(qx), -self.half_width),
                               int_min(<int>floor(px) + 1, self.half_width)):
                    r = (x - qx) / (px - qx)
                    rn[0] = ((1 - r) * qn[0] + r * pn[0])
                    rn[1] = ((1 - r) * qn[1] + r * pn[1])
                    rn[2] = ((1 - r) * qn[2] + r * pn[2])
                    z = pz * qz / (r * qz + (1 - r) * pz)
                    self._shade_vertex(_a, _b, _c, rn, color)
                    self._draw_pixel(x, y, z, color)

    def _prepare_polygons(self, DOUBLE_t[:,:] points, UINT64_t[:,:] indexes):
        cdef DOUBLE_t[:,:,:] polygons
        cdef DOUBLE_t[:,:] polygon
        cdef DOUBLE_t *polygon_normals
        cdef UINT64_t[:] index
        cdef int i

        cp = self.camera_position

        # ポリゴンのリストを作成
        polygons = np.empty((indexes.shape[0], 3, 3), dtype=DOUBLE)
        for i in range(indexes.shape[0]):
            polygon = polygons[i]
            index = indexes[i]
            polygon[0] = points[index[0]]
            polygon[1] = points[index[1]]
            polygon[2] = points[index[2]]

        # すべてのポリゴンの面の法線ベクトルを求める
        polygon_normals = <DOUBLE_t *>malloc(
                sizeof(DOUBLE_t) * indexes.shape[0] * 3)
        calc_polygon_normals(polygons, polygon_normals)

        if self.shading_mode is not ShadingMode.flat:
            # 各頂点の法線ベクトルを計算
            self._vertex_normals = <DOUBLE_t *>malloc(
                sizeof(DOUBLE_t) * points.shape[0] * 3)
            calc_vertex_normals(indexes, polygon_normals, self._vertex_normals,
                                points.shape[0])

        self._points = points
        self._indexes = indexes
        self._polygons = polygons
        self._polygon_normals = polygon_normals

    def prepare_polygons(self, np.ndarray points, np.ndarray indexes):
        self._prepare_polygons(points, indexes)

    def _draw_polygons(self):
        cdef DOUBLE_t[:,:,:] polygons
        cdef DOUBLE_t[:,:] points, polygon
        cdef DOUBLE_t[:] cp, p1, p2, p3
        cdef DOUBLE_t *n
        cdef UINT64_t[:,:] indexes
        cdef UINT64_t[:] index
        cdef int i

        points = self._points
        indexes = self._indexes
        polygons = self._polygons
        cp = self.camera_position

        if self.shading_mode is ShadingMode.flat:
            for i in range(polygons.shape[0]):
                polygon = polygons[i]
                n = self._polygon_normals + 3 * i

                # ポリゴンがカメラを向いていなければ描画しない
                p1 = polygon[0]
                if ((cp[0] - p1[0]) * n[0]
                    + (cp[1] - p1[1]) * n[1]
                    + (cp[2] - p1[2]) * n[2]) < 0:
                    continue
                p2 = polygon[1]
                if ((cp[0] - p2[0]) * n[0]
                    + (cp[1] - p2[1]) * n[1]
                    + (cp[2] - p2[2]) * n[2]) < 0:
                    continue
                p3 = polygon[2]
                if ((cp[0] - p3[0]) * n[0]
                    + (cp[1] - p3[1]) * n[1]
                    + (cp[2] - p3[2]) * n[2]) < 0:
                    continue

                self._draw_polygon_flat(p1, p2, p3, n)
        else:
            # ポリゴンを描画
            if self.shading_mode is ShadingMode.gouraud:
                for i in range(polygons.shape[0]):
                    polygon = polygons[i]
                    index = indexes[i]
                    n = self._polygon_normals + 3 * i

                    # ポリゴンがカメラを向いていなければ描画しない
                    p1 = polygon[0]
                    if ((cp[0] - p1[0]) * n[0]
                        + (cp[1] - p1[1]) * n[1]
                        + (cp[2] - p1[2]) * n[2]) < 0:
                        continue
                    p2 = polygon[1]
                    if ((cp[0] - p2[0]) * n[0]
                        + (cp[1] - p2[1]) * n[1]
                        + (cp[2] - p2[2]) * n[2]) < 0:
                        continue
                    p3 = polygon[2]
                    if ((cp[0] - p3[0]) * n[0]
                        + (cp[1] - p3[1]) * n[1]
                        + (cp[2] - p3[2]) * n[2]) < 0:
                        continue

                    self._draw_polygon_gouraud(
                        p1, p2, p3,
                        self._vertex_normals + index[0] * 3,
                        self._vertex_normals + index[1] * 3,
                        self._vertex_normals + index[2] * 3)
            elif self.shading_mode is ShadingMode.phong:
                for i in range(polygons.shape[0]):
                    polygon = polygons[i]
                    index = indexes[i]
                    n = self._polygon_normals + 3 * i

                    # ポリゴンがカメラを向いていなければ描画しない
                    p1 = polygon[0]
                    if ((cp[0] - p1[0]) * n[0]
                        + (cp[1] - p1[1]) * n[1]
                        + (cp[2] - p1[2]) * n[2]) < 0:
                        continue
                    p2 = polygon[1]
                    if ((cp[0] - p2[0]) * n[0]
                        + (cp[1] - p2[1]) * n[1]
                        + (cp[2] - p2[2]) * n[2]) < 0:
                        continue
                    p3 = polygon[2]
                    if ((cp[0] - p3[0]) * n[0]
                        + (cp[1] - p3[1]) * n[1]
                        + (cp[2] - p3[2]) * n[2]) < 0:
                        continue

                    self._draw_polygon_phong(
                        p1, p2, p3,
                        self._vertex_normals + index[0] * 3,
                        self._vertex_normals + index[1] * 3,
                        self._vertex_normals + index[2] * 3)

    def draw_polygons(self):
        self._draw_polygons()