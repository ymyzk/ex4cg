#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

import numpy as np

from cg.utils import random_color


DOUBLE = np.float64

_zeros = np.zeros(3, dtype=DOUBLE)


def _unit_vector(vector):
    """単位ベクトルを求める処理

    :param numpy.ndarray vector: 単位ベクトルを求めるベクトル
    :rtype: numpy.ndarray
    """
    return vector / np.linalg.norm(vector)


class ShadingMode(Enum):
    flat = 0
    gouraud = 1
    phong = 2


class AmbientShader(object):
    """環境光を計算するシェーダ"""
    def __init__(self, luminance, intensity):
        """
        :param numpy.ndarray luminance: 入射光の強さ 0.0-1.0 (r, g, b)
        :param float intensity: 環境光係数 0.0-1.0)
        """
        self.shade = intensity * luminance

    def calc(self, *_):
        return self.shade


class DiffuseShader(object):
    """拡散反射を計算するシェーダ"""
    def __init__(self, direction, luminance, color):
        """
        :param numpy.ndarray direction: 入射光の方向 (x, y, z)
        :param numpy.ndarray luminance: 入射光の強さ (r, g, b)
        :param numpy.ndarray color: 拡散反射係数 (r, g, b)
        """
        # 方向ベクトルを単位ベクトルに変換
        self.direction = _unit_vector(direction)
        self._pre_shade = color * luminance

    def calc(self, _, normal):
        """
        :param numpy.ndarray normal: 法線ベクトル
        """
        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        # TODO: 現状では実行されないのでなくてもよい
        if np.count_nonzero(normal) == 0:
            return _zeros
        # 反射光を計算
        cos = -np.dot(self.direction, normal)
        # ポリゴンが裏を向いているときは, 反射光なし
        if cos < 0.0:
            return _zeros
        return self._pre_shade * cos


class RandomColorShader(object):
    """ランダムな色を返すシェーダ"""
    def calc(self, *_):
        return random_color()


class SpecularShader(object):
    """鏡面反射を計算するシェーダ"""
    def __init__(self,
                 camera_position, direction, luminance, color, shininess):
        """
        :param numpy.ndarray camera_position: カメラの位置 (x, y, z)
        :param numpy.ndarray direction: 入射光の方向 (x, y, z)
        :param numpy.ndarray luminance: 入射光の強さ (r, g, b)
        :param numpy.ndarray color: 鏡面反射係数 (r, g, b)
        :param float shininess: 鏡面反射強度 s 0.0-1.0
        """
        self.camera_position = camera_position
        # 方向ベクトルを単位ベクトルに変換
        self.direction = _unit_vector(direction)
        self.shininess = shininess * 128
        self._pre_shade = color * luminance

    def calc(self, polygon, normal):
        # 法線ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        # TODO: 現状では実行されないのでなくてもよい
        if np.count_nonzero(normal) == 0:
            return _zeros
        # ポリゴンの重心
        # g = (polygon[0] + polygon[1] + polygon[2]) / 3
        # ポリゴンから視点への単位方向ベクトル
        e = _unit_vector(self.camera_position - polygon[0])
        s = e - self.direction
        s /= np.linalg.norm(s)
        sn = np.dot(s, normal)
        # ポリゴンが裏を向いているときは, 反射光なし
        if sn < 0.0:
            return _zeros
        return sn ** self.shininess * self._pre_shade