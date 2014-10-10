#!/usr/bin/env python
# -*- coding: utf-8 -*-


from enum import Enum

import numpy as np


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
        self.points = []
        self.indexes = []

    def load(self, fp):
        """VRML ファイルを読み込む処理"""
        status = Status.material
        self.points = []
        self.indexes = []
        for l in fp:
            # コメント行の読み飛ばし
            l = l.strip()
            if l.startswith('#'):
                continue
            items = tuple(filter(lambda i: i != '', l.split(' ')))
            if len(items) == 0:
                continue

            if items[0] in ('diffuseColor', 'specularColor',
                            'ambientIntensity', 'shininess',):
                if items[0] == 'diffuseColor':
                    self.diffuse_color = np.array(
                        tuple(map(float, items[1:4])), dtype=np.float64)
                elif items[0] == 'specularColor':
                    self.specular_color = np.array(
                        tuple(map(float, items[1:4])), dtype=np.float64)
                elif items[0] == 'shininess':
                    self.shininess = float(items[1])
                elif items[0] == 'ambientIntensity':
                    self.ambient_intensity = float(items[1])
            elif items[0] == 'point':
                status = Status.point
                continue
            elif items[0] == 'coordIndex':
                status = Status.index
                continue

            if (status is Status.point and
                    len(items) == 3 and
                    items[2].endswith(',')):
                l = [float(i.replace(',', '')) for i in items]
                self.points.append(np.array(l, dtype=np.float64))

            if status is Status.index and len(items) == 4:
                self.indexes.append(
                    [int(i.replace(',', '')) for i in items[:3]])