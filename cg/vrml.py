#!/usr/bin/env python
# -*- coding: utf-8 -*-


from enum import Enum

import numpy as np


DOUBLE = np.float64
UINT64 = np.uint64


class Status(Enum):
    material = 1
    point = 2
    index = 3


class Vrml(object):
    def __init__(self):
        self.diffuse_color = None
        self.specular_color = None
        self.ambient_intensity = None
        self.shininess = None
        self.points = np.zeros((0, 3), dtype=DOUBLE)
        self.indexes = np.zeros((0, 3), dtype=UINT64)

    def load(self, fp):
        """VRML ファイルを読み込む処理"""
        status = Status.material
        points = []
        indexes = []
        for l in fp:
            # コメント行の読み飛ばし
            l = l.strip()
            if l.startswith('#'):
                continue
            items = tuple(filter(lambda i: i != '',
                                 l.replace(',', '').split(' ')))
            if len(items) == 0:
                continue

            if status is Status.index and len(items) == 4:
                indexes.append((
                    int(items[0]),
                    int(items[1]),
                    int(items[2])))
            elif status is Status.point and len(items) == 3:
                points.append((
                    float(items[0]),
                    float(items[1]),
                    float(items[2])))
            elif status is Status.material:
                if items[0] == 'diffuseColor':
                    self.diffuse_color = np.array(
                        tuple(map(float, items[1:4])), dtype=DOUBLE)
                elif items[0] == 'specularColor':
                    self.specular_color = np.array(
                        tuple(map(float, items[1:4])), dtype=DOUBLE)
                elif items[0] == 'shininess':
                    self.shininess = float(items[1])
                elif items[0] == 'ambientIntensity':
                    self.ambient_intensity = float(items[1])
            if items[0] == 'point':
                status = Status.point
            if items[0] == 'coordIndex':
                status = Status.index
        self.points = np.array(points, dtype=DOUBLE)
        self.indexes = np.array(indexes, dtype=UINT64)