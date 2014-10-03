#!/usr/bin/env python
# -*- coding: utf-8 -*-


import enum

import numpy as np


Status = enum.Enum("Status", "material point index")

class Vrml(object):
    def __init__(self):
        self.diffuse_color = None
        self.specular_color = None
        self.ambient_intensity = None
        self.shiness = None
        self.polygons = []

    def load(self, fp):
        """VRML ファイルを読み込む処理"""
        status = Status.material
        points = []
        indexes = []
        self.polygons = []
        for l in fp:
            # コメント行の読み飛ばし
            l = l.strip()
            if l.startswith('#'):
                continue
            items = l.split(' ')
            if len(items) == 0:
                continue

            if items[0] in ('diffuseColor', 'specularColor',
                            'ambientIntensity', 'shininess',):
                if items[0] == 'diffuseColor':
                    self.diffuse_color = np.array(items[1:4])
            elif items[0] == 'point':
                status = Status.point
                continue
            elif items[0] == 'coordIndex':
                status = Status.index
                continue

            if (status is Status.point and
                    len(items) == 3 and
                    items[2].endswith(',')):
                points.append(np.array(
                    tuple(map(lambda i: float(i.replace(',', '')), items))))

            if (status is Status.index and len(items) == 4):
                indexes.append(
                    tuple(map(lambda i: int(i.replace(',', '')), items[:3])))

        for index in indexes:
            self.polygons.append(
                tuple((map(lambda i: points[i], index)))
            )

        # print(points)
        # print(indexes)
        # print(self.polygons)
