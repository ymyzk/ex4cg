#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class Camera(object):
    def __init__(self, position, angle, focus):
        self.position = position
        self.angle = angle
        self.focus = focus
        # カメラ座標系 -> 画像平面の変換行列
        self.array = np.array((
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0 / self.focus, 0.0)
        ))