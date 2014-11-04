#!/usr/bin/env python
# -*- coding: utf-8 -*-

from math import cos, pi, sin, sqrt

import numpy as np


class Camera(object):
    def __init__(self, position, angle, focus):
        """
        :param numpy.ndarray position: カメラの位置 (x, y, z)
        :param numpy.ndarray angle: カメラの角度 (x, y, z) [rad]
        x, y, z 軸の回転量
        (0, 0, 0) から (0, 0, 1) を見る向きを (0, 0, 0) とする
        :param float focus: カメラの焦点距離
        """
        self.position = position
        self.angle = angle
        self.focus = focus
        # 座標変換行列
        self.array = np.array((
            (1.0, 0.0,              0.0, 0.0),
            (0.0, 1.0,              0.0, 0.0),
            (0.0, 0.0,              1.0, 0.0),
            (0.0, 0.0, 1.0 / self.focus, 0.0)
        ), dtype=np.float64)

        # 平行移動
        move = np.array((
            (1.0, 0.0, 0.0, -self.position[0]),
            (0.0, 1.0, 0.0, -self.position[1]),
            (0.0, 0.0, 1.0, -self.position[2]),
            (0.0, 0.0, 0.0,               1.0)
        ), dtype=np.float64)
        self.array = np.dot(self.array, move)

        # (0, 0, 0)　から (0, 0, 1) を見る向きからの回転
        # x 軸の周りの回転
        axr = self.angle[0] / 180.0 * pi  # radian
        axc = cos(axr)  # cos
        axs = sin(axr)  # sin
        rotate_x = np.array((
            (1.0, 0.0,  0.0, 0.0),
            (0.0, axc, -axs, 0.0),
            (0.0, axs,  axc, 0.0),
            (0.0, 0.0,  0.0, 1.0)
        ), dtype=np.float64)
        self.array = np.dot(self.array, rotate_x)

        # y 軸の周りの回転
        ayr = self.angle[1] / 180.0 * pi  # radian
        ayc = cos(ayr)  # cos
        ays = sin(ayr)  # sin
        rotate_y = np.array((
            (+ayc, 0.0, ays, 0.0),
            (+0.0, 1.0, 0.0, 0.0),
            (-ays, 0.0, ayc, 0.0),
            (+0.0, 0.0, 0.0, 1.0)
        ), dtype=np.float64)
        self.array = np.dot(self.array, rotate_y)

        # z 軸の周りの回転
        azr = self.angle[2] / 180.0 * pi  # radian
        azc = cos(azr)  # cos
        azs = sin(azr)  # sin
        rotate_z = np.array((
            (azc, -azs, 0.0, 0.0),
            (azs,  azc, 0.0, 0.0),
            (0.0,  0.0, 1.0, 0.0),
            (0.0,  0.0, 0.0, 1.0)
        ), dtype=np.float64)
        self.array = np.dot(self.array, rotate_z)

        # 4x4 -> 4x3 に変換
        self.array = np.delete(self.array, 3, 0)