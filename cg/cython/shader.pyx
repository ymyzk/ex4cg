#cython: language_level=3, boundscheck=False, nonecheck=False
# -*- coding: utf-8 -*-

import numpy as np
cimport numpy as np

from cg.utils import random_color


DTYPE = np.float64
ctypedef np.float64_t DTYPE_t

_zeros = np.zeros(3, dtype=DTYPE)


def _unit_vector(np.ndarray[DTYPE_t] vector):
    """単位ベクトルを求める処理

    :param np.ndarray vector: 単位ベクトルを求めるベクトル
    :rtype: np.ndarray
    """
    return vector / np.linalg.norm(vector)


class AmbientShader(object):
    """環境光を計算するシェーダ"""
    def __init__(self, np.ndarray[DTYPE_t] luminance, float intensity,
                 int depth=8):
        """
        :param numpy.ndarray luminance: 入射光の強さ 0.0-1.0 (r, g, b)
        :param float intensity: 環境光係数 0.0-1.0
        :param int depth: (optional) 階調数 (bit)
        """
        self.shade = intensity * 2 ** (depth - 1) * luminance

    def calc(self, *_):
        return self.shade


class DiffuseShader(object):
    """拡散反射を計算するシェーダ"""
    def __init__(self, np.ndarray[DTYPE_t] direction,
                 np.ndarray[DTYPE_t] luminance,
                 np.ndarray[DTYPE_t] color, int depth=8):
        """
        :param np.ndarray direction: 入射光の方向 (x, y, z)
        :param np.ndarray luminance: 入射光の強さ (r, g, b)
        :param np.ndarray color: 拡散反射係数 (r, g, b)
        :param int depth: (optional) 階調数 (bit)
        """
        # 方向ベクトルを単位ベクトルに変換
        self.direction = _unit_vector(direction)
        self._pre_shade = (2 ** depth - 1) * color * luminance

    def calc(self, _, np.ndarray[DTYPE_t] normal):
        """
        :param np.ndarray normal: 法線ベクトル
        """
        cdef DTYPE_t cos

        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        if np.count_nonzero(normal) == 0:
            return _zeros
        # 反射光を計算
        cos = np.dot(self.direction, normal)
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


class SpecularShader(object):
    """鏡面反射を計算するシェーダ"""
    def __init__(self, np.ndarray[DTYPE_t] camera_position,
                 np.ndarray[DTYPE_t] direction,
                 np.ndarray[DTYPE_t] luminance,
                 np.ndarray[DTYPE_t] color, float shininess,
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

    def calc(self, np.ndarray[DTYPE_t, ndim=2] polygon,
             np.ndarray[DTYPE_t] normal):
        cdef np.ndarray[DTYPE_t] e, s
        cdef DTYPE_t sn

        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        # TODO: 現状では実行されないのでなくてもよい
        if np.count_nonzero(normal) == 0:
            return _zeros
        # ポリゴンの重心
        # g = (polygon[0] + polygon[1] + polygon[2]) / 3
        # ポリゴンから視点への単位方向ベクトル
        e = _unit_vector(self.camera_position - polygon[0])
        s = e - self.direction
        s = _unit_vector(s)
        sn = np.dot(s, normal)
        # ポリゴンが裏を向いているときは, 反射光なし
        if sn < 0.0:
            return _zeros
        return sn ** self.shininess * self._pre_shade