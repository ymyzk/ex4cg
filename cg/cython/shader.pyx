#cython: language_level=3, boundscheck=False, cdivision=True
#cython: profile=True
# -*- coding: utf-8 -*-

from libc.math cimport sqrt
import numpy as np
cimport numpy as np

from cg.cython.utils import random_color


DOUBLE = np.float64
ctypedef np.float64_t DOUBLE_t


cdef inline void _unit_vector(DOUBLE_t[:] v):
    """単位ベクトルに変換する処理"""
    cdef DOUBLE_t norm
    norm = sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    v[0] /= norm
    v[1] /= norm
    v[2] /= norm


cdef inline double _dot_vectors(DOUBLE_t[:] v1, DOUBLE_t[:] v2, int l):
    cdef int i
    cdef double dot = 0.0
    for i in range(l):
        dot += v1[i] * v2[i]
    return dot


cdef class AmbientShader:
    """環境光を計算するシェーダ"""
    cdef DOUBLE_t[:] shade

    def __init__(self, np.ndarray[DOUBLE_t, ndim=1] luminance, float intensity,
                 int depth=8):
        """
        :param numpy.ndarray luminance: 入射光の強さ 0.0-1.0 (r, g, b)
        :param float intensity: 環境光係数 0.0-1.0
        :param int depth: (optional) 階調数 (bit)
        """
        # self.shade = intensity * 2 ** (depth - 1) * luminance
        self.shade = intensity * luminance

    cpdef calc(self, DOUBLE_t[:] a, DOUBLE_t[:] b, DOUBLE_t[:] c,
               DOUBLE_t[:] n, DOUBLE_t[:] cl):
        cl[0] = self.shade[0]
        cl[1] = self.shade[1]
        cl[2] = self.shade[2]


cdef class DiffuseShader:
    """拡散反射を計算するシェーダ"""
    cdef DOUBLE_t[:] direction, _pre_shade

    def __init__(self, np.ndarray[DOUBLE_t, ndim=1] direction,
                 np.ndarray[DOUBLE_t, ndim=1] luminance,
                 np.ndarray[DOUBLE_t, ndim=1] color, int depth=8):
        """
        :param np.ndarray direction: 入射光の方向 (x, y, z)
        :param np.ndarray luminance: 入射光の強さ (r, g, b)
        :param np.ndarray color: 拡散反射係数 (r, g, b)
        :param int depth: (optional) 階調数 (bit)
        """
        # 方向ベクトルを単位ベクトルに変換
        _unit_vector(direction)
        self.direction = direction
        self._pre_shade = (2 ** depth - 1) * color * luminance

    cpdef calc(self, DOUBLE_t[:] a, DOUBLE_t[:] b, DOUBLE_t[:] c,
               DOUBLE_t[:] n, DOUBLE_t[:] cl):
        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        # TODO: 現状では実行されないのでなくてもよい
        # if n[0] == 0.0 and n[1] == 0.0 and n[2] == 0.0:
        #     cl[0] = 0.0
        #     cl[1] = 0.0
        #     cl[2] = 0.0
        #     return
        # 反射光を計算
        cos = _dot_vectors(self.direction, n, 3)
        # ポリゴンが裏を向いているときは, 反射光なし
        if 0.0 < cos:
            cl[0] = 0.0
            cl[1] = 0.0
            cl[2] = 0.0
            return

        cl[0] = -cos * self._pre_shade[0]
        cl[1] = -cos * self._pre_shade[1]
        cl[2] = -cos * self._pre_shade[2]


cdef class RandomColorShader:
    """ランダムな色を返すシェーダ"""
    cdef int depth

    def __init__(self, int depth=8):
        """
        :param int depth: (optional) 階調数 (bit)
        """
        self.depth = depth

    cpdef calc(self, DOUBLE_t[:] a, DOUBLE_t[:] b, DOUBLE_t[:] c,
               DOUBLE_t[:] n, DOUBLE_t[:] cl):
        random_color(cl, self.depth)


cdef class SpecularShader:
    """鏡面反射を計算するシェーダ"""
    cdef DOUBLE_t[:] camera_position, direction, _pre_shade
    cdef DOUBLE_t shininess

    def __init__(self, np.ndarray[DOUBLE_t, ndim=1] camera_position,
                 np.ndarray[DOUBLE_t, ndim=1] direction,
                 np.ndarray[DOUBLE_t, ndim=1] luminance,
                 np.ndarray[DOUBLE_t, ndim=1] color, float shininess,
                 int depth=8):
        """
        :param np.ndarray camera_position: カメラの位置 (x, y, z)
        :param np.ndarray direction: 入射光の方向 (x, y, z)
        :param np.ndarray luminance: 入射光の強さ (r, g, b)
        :param np.ndarray color: 鏡面反射係数 (r, g, b)
        :param np shininess: 鏡面反射強度 s 0.0-1.0
        :param int depth: (optional) 階調数 (bit)
        """
        self.camera_position = camera_position
        # 方向ベクトルを単位ベクトルに変換
        _unit_vector(direction)
        self.direction = direction
        self.shininess = shininess * 128
        self._pre_shade = (2 ** depth - 1) * color * luminance

    cpdef calc(self, DOUBLE_t[:] a, DOUBLE_t[:] b, DOUBLE_t[:] c,
               DOUBLE_t[:] n, DOUBLE_t[:] cl):
        cdef DOUBLE_t e[3]
        cdef DOUBLE_t s[3]
        cdef DOUBLE_t sn

        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        # TODO: 現状では実行されないのでなくてもよい
        # if n[0] == 0.0 and n[1] == 0.0 and n[2] == 0.0:
        #     cl[0] = 0.0
        #     cl[1] = 0.0
        #     cl[2] = 0.0
        #     return
        # ポリゴンの重心
        # g = (polygon[0] + polygon[1] + polygon[2]) / 3
        # ポリゴンから視点への単位方向ベクトル
        e[0] = self.camera_position[0] - a[0]
        e[1] = self.camera_position[1] - a[1]
        e[2] = self.camera_position[2] - a[2]
        _unit_vector(e)
        s[0] = e[0] - self.direction[0]
        s[1] = e[1] - self.direction[1]
        s[2] = e[2] - self.direction[2]
        _unit_vector(s)
        sn = _dot_vectors(s, n, 3)
        # ポリゴンが裏を向いているときは, 反射光なし
        if sn < 0.0:
            cl[0] = 0.0
            cl[1] = 0.0
            cl[2] = 0.0
            return
        sn **= self.shininess
        cl[0] = sn * self._pre_shade[0]
        cl[1] = sn * self._pre_shade[1]
        cl[2] = sn * self._pre_shade[2]