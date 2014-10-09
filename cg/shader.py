#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class DiffuseShader(object):
    def __init__(self, direction, color, depth=8):
        """
        :param direction: 入射光の方向 (x, y, z)
        :param color: 拡散反射係数 (r, g, b)
        :param depth:
        :return:
        """
        # 方向ベクトルを単位ベクトルに変換
        self.direction = direction / np.linalg.norm(direction)
        self.color = color
        self.depth = 8

    def calc(self, polygon):
        # 直交ベクトル
        # 反時計回りを表
        # cross = np.cross(polygon[0] - polygon[1], polygon[1] - polygon[2])
        # 時計回りを表
        cross = np.cross(polygon[2] - polygon[1], polygon[1] - polygon[0])
        # 直交ベクトルがゼロベクトルであれば, 計算不能
        # Ex: 面積0のポリゴン
        if np.count_nonzero(cross) == 0:
            return np.zeros(3)
        # 法線ベクトル (単位ベクトル化)
        normal = cross / np.linalg.norm(cross)
        # 反射光を計算
        cos = -np.dot(self.direction, normal)
        # ポリゴンが裏を向いているときは, 反射光なし
        if cos < 0:
            return np.zeros(3)
        diffuse = (2 ** self.depth - 1) * np.dot(cos, self.color)
        return diffuse.astype(np.uint8)