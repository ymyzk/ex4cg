#cython: language_level=3, boundscheck=False, cdivision=True
# -*- coding: utf-8 -*-

from random import random

from libc.math cimport ceil, floor, sqrt
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
                               DOUBLE_t[:,:] polygon_normals) nogil:
    """ポリゴンの面の法線ベクトルを求める処理

    :param polygons: ポリゴンの配列 (n x 3 x 3)
    :param polygon_normals: ポリゴンの面の法線ベクトルを格納する配列 (n x 3)
    """
    cdef DOUBLE_t[:,:] polygon
    cdef DOUBLE_t[:] a, b, c, polygon_normal
    cdef DOUBLE_t[3] ab, bc, cr
    cdef DOUBLE_t norm
    cdef int i, l

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
        polygon_normal = polygon_normals[i]
        polygon_normal[0] = cr[0] / norm
        polygon_normal[1] = cr[1] / norm
        polygon_normal[2] = cr[2] / norm


cdef void calc_vertex_normals(UINT64_t[:,:] indexes,
                              DOUBLE_t[:,:] polygon_normals,
                              DOUBLE_t[:,:] vertex_normals) nogil:
    """各頂点の法線ベクトルを計算する処理

    各頂点の法線ベクトルは, その頂点を含む全ての面の法線ベクトルの平均である

    :param indexes: ポリゴンのインデックス
    :param polygon_normals: ポリゴンの面の法線ベクトルの配列 (n x 3)
    :param vertex_normals: 頂点の法線ベクトルの配列 (m x 3)
    """
    cdef DOUBLE_t[:,:] vertexes
    cdef DOUBLE_t[:] normal, vertex, vertex_normal
    cdef UINT64_t[:] vertexes_n, index
    cdef UINT64_t vertex_n
    cdef int i, j, l

    l = vertex_normals.shape[0]

    with gil:
        vertexes = np.zeros((l, 3), dtype=DOUBLE)
        vertexes_n = np.zeros(l, dtype=UINT64)

    # 各頂点を含む面の法線ベクトルの和を求める
    l = indexes.shape[0]
    for i in range(l):
        index = indexes[i]
        normal = polygon_normals[i]

        j = index[0]
        vertex = vertexes[j]
        vertex[0] += normal[0]
        vertex[1] += normal[1]
        vertex[2] += normal[2]
        vertexes_n[j] += 1

        j = index[1]
        vertex = vertexes[j]
        vertex[0] += normal[0]
        vertex[1] += normal[1]
        vertex[2] += normal[2]
        vertexes_n[j] += 1

        j = index[2]
        vertex = vertexes[j]
        vertex[0] += normal[0]
        vertex[1] += normal[1]
        vertex[2] += normal[2]
        vertexes_n[j] += 1

    # 各頂点の法線ベクトルの平均値を求める
    l = vertexes.shape[0]
    for i in range(l):
        vertex = vertexes[i]
        vertex_n = vertexes_n[i]
        vertex_normal = vertex_normals[i]
        if 0 < vertex_n:
            vertex_normal[0] = vertex[0] / vertex_n
            vertex_normal[1] = vertex[1] / vertex_n
            vertex_normal[2] = vertex[2] / vertex_n
        else:
            vertex_normal[0] = 0.0
            vertex_normal[1] = 0.0
            vertex_normal[2] = 0.0


cdef class Renderer:
    cdef public object shading_mode
    cdef object camera, shaders
    cdef int depth, width, height, half_width, half_height, z_buffering
    cdef int _depth
    cdef readonly np.ndarray data
    cdef UINT8_t[:,:] _data
    cdef np.ndarray z_buffer
    cdef DOUBLE_t[:,:] _z_buffer
    cdef DOUBLE_t[:] camera_position
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

    def __init__(self, int width, int height, z_buffering=True,
                 int depth=8, shading_mode=ShadingMode.flat):
        """
        :param bool z_buffering: Z バッファを有効にするかどうか
        """
        self.depth = depth
        self.width = width
        self.height = height
        self.z_buffering = z_buffering
        self.shading_mode = shading_mode

        self.data = np.zeros((self.height, self.width * 3), dtype=UINT8)
        self._data = self.data
        self.z_buffer = np.empty((self.height, self.width), dtype=DOUBLE)
        self.z_buffer.fill(float('inf'))
        self._z_buffer = self.z_buffer
        self.half_width = self.width // 2
        self.half_height = self.height // 2
        self._depth = 2 ** depth - 1

    property shaders:
        def __get__(self):
            return self.shaders

        def __set__(self, value):
            self.shaders = value

            # Python で書かれたシェーダーから必要な値を取り出す
            for shader in self.shaders:
                if isinstance(shader, AmbientShader):
                    self._is_ambient_shader_enabled = 1
                    self._ambient_shade[0] = shader.shade[0]
                    self._ambient_shade[1] = shader.shade[1]
                    self._ambient_shade[2] = shader.shade[2]
                elif isinstance(shader, DiffuseShader):
                    self._is_diffuse_shader_enabled = 1
                    self._diffuse_direction[0] = shader.direction[0]
                    self._diffuse_direction[1] = shader.direction[1]
                    self._diffuse_direction[2] = shader.direction[2]
                    self._diffuse_pre_shade[0] = shader.pre_shade[0]
                    self._diffuse_pre_shade[1] = shader.pre_shade[1]
                    self._diffuse_pre_shade[2] = shader.pre_shade[2]
                elif isinstance(shader, RandomColorShader):
                    self._is_random_shader_enabled = 1
                elif isinstance(shader, SpecularShader):
                    self._is_specular_shader_enabled = 1
                    self._specular_direction[0] = shader.direction[0]
                    self._specular_direction[1] = shader.direction[1]
                    self._specular_direction[2] = shader.direction[2]
                    self._specular_shininess = shader.shininess
                    self._specular_pre_shade[0] = shader.pre_shade[0]
                    self._specular_pre_shade[1] = shader.pre_shade[1]
                    self._specular_pre_shade[2] = shader.pre_shade[2]

    property camera:
        def __get__(self):
            return self.camera

        def __set__(self, value):
            self.camera = value

            self.focus = self.camera.focus
            self.camera_position = self.camera.position

    cdef void _convert_point(self, DOUBLE_t[:] point) nogil:
        """カメラ座標系の座標を画像平面上の座標に変換する処理

        画像平面の x, y, z + 元の座標の z
        """
        cdef DOUBLE_t z_ip
        z_ip = self.focus / point[2]
        point[0] = z_ip * point[0]
        point[1] = z_ip * point[1]

    cdef void _shade_ambient(self, DOUBLE_t[:] cl) nogil:
        cl[0] += self._ambient_shade[0]
        cl[1] += self._ambient_shade[1]
        cl[2] += self._ambient_shade[2]

    cdef void _shade_diffuse(self, DOUBLE_t[:] n, DOUBLE_t[:] cl) nogil:
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

    cdef void _shade_random(self, DOUBLE_t[:] cl):
        """ランダムな色をつけるシェーダ"""
        cl[0] += random()
        cl[1] += random()
        cl[2] += random()

    cdef void _shade_specular(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                              DOUBLE_t[:] c, DOUBLE_t[:] n,
                              DOUBLE_t[:] cl) nogil:
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
        if sn < 0.0:
            return

        sn **= self._specular_shininess
        cl[0] += sn * self._specular_pre_shade[0]
        cl[1] += sn * self._specular_pre_shade[1]
        cl[2] += sn * self._specular_pre_shade[2]

    cdef void _shade_vertex(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                            DOUBLE_t[:] c, DOUBLE_t[:] n,
                            DOUBLE_t[:] color) nogil:
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

    cdef void _draw_pixel(self, int x, int y, DOUBLE_t z,
                          DOUBLE_t[:] cl) nogil:
        """画素を描画する処理"""
        cdef int data_x, data_y

        # Z バッファでテスト
        if not self.z_buffering or z <= self._z_buffer[y][x]:
            # 飽和
            if 1.0 < cl[0]:
                cl[0] = 1.0
            if 1.0 < cl[1]:
                cl[1] = 1.0
            if 1.0 < cl[2]:
                cl[2] = 1.0

            # X: -128 ~ 127 -> (x + 128) -> 0 ~ 255
            # Y: -127 ~ 128 -> (128 - y) -> 0 ~ 255
            # NOTE: サンプル画像がおかしいので X を反転して表示している
            # data_x = 3 * (self.half_width + x) # 反転させないコード
            data_x = 3 * (self.half_width - x - 1)
            data_y = self.half_height - y
            self._data[data_y][data_x+0] = <UINT8_t>(cl[0] * self._depth)
            self._data[data_y][data_x+1] = <UINT8_t>(cl[1] * self._depth)
            self._data[data_y][data_x+2] = <UINT8_t>(cl[2] * self._depth)
            self._z_buffer[y][x] = z

    cdef void _draw_polygon_flat(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                 DOUBLE_t[:] c, DOUBLE_t[:] n):
        """ポリゴンを描画する処理 (フラットシェーディング)"""
        cdef int x, y
        cdef DOUBLE_t px, qx, pz, qz, r, s
        cdef DOUBLE_t[3] d, color

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
        self._shade_vertex(a, b, c, n, color)

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
                # x が同じの時はすぐに終了
                self._draw_pixel(<int>px, y, pz, color)
                continue
            elif px > qx:
                # x についてソート
                px, qx = qx, px
            for x in range(int_max(<int>ceil(px), 1 - self.half_width),
                           int_min(<int>floor(qx), self.half_width) + 1):
                r = (x - px) / (qx - px)
                self._draw_pixel(x, y, pz * qz / (r * qz + (1 - r) * pz),
                                 color)

    cdef void _draw_polygon_gouraud(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                    DOUBLE_t[:] c, DOUBLE_t[:] an,
                                    DOUBLE_t[:] bn, DOUBLE_t[:] cn):
        """ポリゴンを描画する処理 (グーローシェーディング)"""
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
        d[2] = a[2] * c[2] / (r * a[2] + (1 - r) * c[2])
        dn[0] = (1 - r) * an[0] + r * cn[0]
        dn[1] = (1 - r) * an[1] + r * cn[1]
        dn[2] = (1 - r) * an[2] + r * cn[2]

        # 頂点をそれぞれの法線ベクトルでシェーディング
        self._shade_vertex(a, b, c, an, ac)
        self._shade_vertex(a, b, c, bn, bc)
        self._shade_vertex(a, b, c, cn, cc)
        self._shade_vertex(a, b, c, dn, dc)

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
                    self._draw_pixel(x, y, pz * qz / (r * qz + (1 - r) * pz), rc)
            else:
                for x in range(int_max(<int>ceil(qx), 1 - self.half_width),
                               int_min(<int>floor(px), self.half_width) + 1):
                    r = (x - qx) / (px - qx)
                    rc[0] = ((1 - r) * qc[0] + r * pc[0])
                    rc[1] = ((1 - r) * qc[1] + r * pc[1])
                    rc[2] = ((1 - r) * qc[2] + r * pc[2])
                    self._draw_pixel(x, y, pz * qz / (r * qz + (1 - r) * pz), rc)

    cdef void _draw_polygon_phong(self, DOUBLE_t[:] a, DOUBLE_t[:] b,
                                  DOUBLE_t[:] c, DOUBLE_t[:] an,
                                  DOUBLE_t[:] bn, DOUBLE_t[:] cn):
        """ポリゴンを描画する処理 (フォンシェーディング)"""
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
                # x が同じの時はすぐに終了
                self._shade_vertex(a, b, c, pn, color)
                self._draw_pixel(<int>px, y, pz, color)
                continue
            elif px < qx:
                for x in range(int_max(<int>ceil(px), 1 - self.half_width),
                               int_min(<int>floor(qx), self.half_width) + 1):
                    r = (x - px) / (qx - px)
                    rn[0] = ((1 - r) * pn[0] + r * qn[0])
                    rn[1] = ((1 - r) * pn[1] + r * qn[1])
                    rn[2] = ((1 - r) * pn[2] + r * qn[2])
                    z = pz * qz / (r * qz + (1 - r) * pz)
                    self._shade_vertex(a, b, c, rn, color)
                    self._draw_pixel(x, y, z, color)
            else:
                for x in range(int_max(<int>ceil(qx), 1 - self.half_width),
                               int_min(<int>floor(px), self.half_width) + 1):
                    r = (x - qx) / (px - qx)
                    rn[0] = ((1 - r) * qn[0] + r * pn[0])
                    rn[1] = ((1 - r) * qn[1] + r * pn[1])
                    rn[2] = ((1 - r) * qn[2] + r * pn[2])
                    z = pz * qz / (r * qz + (1 - r) * pz)
                    self._shade_vertex(a, b, c, rn, color)
                    self._draw_pixel(x, y, z, color)

    def _draw_polygons(self, DOUBLE_t[:,:] points, UINT64_t[:,:] indexes):
        cdef DOUBLE_t[:,:,:] polygons
        cdef DOUBLE_t[:,:] polygon, polygon_normals, vertex_normals
        cdef UINT64_t[:] index
        cdef int i

        # ポリゴンのリストを作成
        polygons = np.empty((indexes.shape[0], 3, 3), dtype=DOUBLE)
        for i in range(indexes.shape[0]):
            polygon = polygons[i]
            index = indexes[i]
            polygon[0] = points[index[0]]
            polygon[1] = points[index[1]]
            polygon[2] = points[index[2]]

        # すべてのポリゴンの面の法線ベクトルを求める
        polygon_normals = np.empty((indexes.shape[0], 3), dtype=DOUBLE)
        calc_polygon_normals(polygons, polygon_normals)

        if self.shading_mode is ShadingMode.flat:
            for i in range(polygons.shape[0]):
                polygon = polygons[i]
                self._draw_polygon_flat(polygon[0], polygon[1],
                                        polygon[2], polygon_normals[i])
        else:
            # 各頂点の法線ベクトルを計算
            vertex_normals = np.empty((points.shape[0], 3), dtype=DOUBLE)
            calc_vertex_normals(indexes, polygon_normals, vertex_normals)

            # ポリゴンを描画
            if self.shading_mode is ShadingMode.gouraud:
                for i in range(polygons.shape[0]):
                    polygon = polygons[i]
                    index = indexes[i]
                    self._draw_polygon_gouraud(polygon[0],
                                               polygon[1],
                                               polygon[2],
                                               vertex_normals[index[0]],
                                               vertex_normals[index[1]],
                                               vertex_normals[index[2]])
            elif self.shading_mode is ShadingMode.phong:
                for i in range(polygons.shape[0]):
                    polygon = polygons[i]
                    index = indexes[i]
                    self._draw_polygon_phong(polygon[0],
                                             polygon[1],
                                             polygon[2],
                                             vertex_normals[index[0]],
                                             vertex_normals[index[1]],
                                             vertex_normals[index[2]])

    def draw_polygons(self, np.ndarray points, np.ndarray indexes):
        self._draw_polygons(points, indexes)