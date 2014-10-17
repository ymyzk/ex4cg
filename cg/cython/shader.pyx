#cython: language_level=3, boundscheck=False, nonecheck=False
# -*- coding: utf-8 -*-

from libc.math cimport sqrt
import numpy as np
cimport numpy as np

from cg.utils import random_color


DOUBLE = np.float64
ctypedef np.float64_t DOUBLE_t

_zeros = np.zeros(3, dtype=DOUBLE)


def _unit_vector(np.ndarray[DOUBLE_t, ndim=1] vector):
    """単位ベクトルを求める処理

    :param np.ndarray vector: 単位ベクトルを求めるベクトル
    :rtype: np.ndarray
    """
    cdef DOUBLE_t norm
    norm = sqrt(vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2)
    return vector / norm


cdef inline double _dot_vectors_internal(double *v1, double *v2, int l):
    cdef int i
    cdef double dot = 0.0
    for i in range(l):
        dot += v1[i] * v2[i]
    return dot


def _dot_vectors(np.ndarray[DOUBLE_t, ndim=1] v1,
                 np.ndarray[DOUBLE_t, ndim=1] v2):
    cdef int l
    l = len(v1)
    return _dot_vectors_internal(<double *>v1.data, <double *>v2.data, l)


cdef class AmbientShader:
    """環境光を計算するシェーダ"""
    cdef np.ndarray shade

    def __init__(self, np.ndarray[DOUBLE_t, ndim=1] luminance, float intensity,
                 int depth=8):
        """
        :param numpy.ndarray luminance: 入射光の強さ 0.0-1.0 (r, g, b)
        :param float intensity: 環境光係数 0.0-1.0
        :param int depth: (optional) 階調数 (bit)
        """
        self.shade = intensity * 2 ** (depth - 1) * luminance

    def calc(self, *_):
        return self.shade


cdef class DiffuseShader:
    """拡散反射を計算するシェーダ"""
    cdef np.ndarray direction, _pre_shade

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
        self.direction = _unit_vector(direction)
        self._pre_shade = (2 ** depth - 1) * color * luminance

    def calc(self, _, np.ndarray[DOUBLE_t, ndim=1] normal):
        """
        :param np.ndarray normal: 法線ベクトル
        """
        cdef DOUBLE_t cos

        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        if np.count_nonzero(normal) == 0:
            return _zeros
        # 反射光を計算
        cos = _dot_vectors(self.direction, normal)
        # ポリゴンが裏を向いているときは, 反射光なし
        if 0.0 < cos:
            return _zeros
        return self._pre_shade * -cos


class RandomColorShader(object):
    """ランダムな色を返すシェーダ"""
    def __init__(self, int depth=8):
        """
        :param int depth: (optional) 階調数 (bit)
        """
        self.depth = depth

    def calc(self, *_):
        return random_color(self.depth)


cdef class SpecularShader:
    """鏡面反射を計算するシェーダ"""
    cdef np.ndarray camera_position, direction, _pre_shade
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
        self.direction = _unit_vector(direction)
        self.shininess = shininess * 128
        self._pre_shade = (2 ** depth - 1) * color * luminance

    def calc(self, np.ndarray[DOUBLE_t, ndim=2] polygon,
             np.ndarray[DOUBLE_t, ndim=1] normal):
        cdef np.ndarray[DOUBLE_t, ndim=1] e, s
        cdef DOUBLE_t sn

        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        # TODO: 現状では実行されないのでなくてもよい
        # if np.count_nonzero(normal) == 0:
        #     return _zeros
        # ポリゴンの重心
        # g = (polygon[0] + polygon[1] + polygon[2]) / 3
        # ポリゴンから視点への単位方向ベクトル
        e = _unit_vector(self.camera_position - polygon[0])
        s = e - self.direction
        s = _unit_vector(s)
        sn = _dot_vectors(s, normal)
        # ポリゴンが裏を向いているときは, 反射光なし
        if sn < 0.0:
            return _zeros
        return sn ** self.shininess * self._pre_shade