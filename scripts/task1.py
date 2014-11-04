#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals
import sys

import numpy as np

from cg.camera import Camera
from cg.ppm import PpmImage
from cg.renderer import Renderer
from cg.shader import RandomColorShader
from cg.utils import random_polygons


points, indexes = random_polygons(3)

# ポリゴンごとに3点の座標を出力する
for index in indexes:
    polygon = tuple((map(lambda i: points[i], index)))
    print(', '.join(map(str, polygon)), file=sys.stderr)

if __name__ == '__main__':
    width = height = 256
    depth = 8
    camera = Camera(position=np.array((0.0, 0.0, 0.0)),
                    angle=np.array((0.0, 0.0, 0.0)),
                    focus=256.0)
    shader = RandomColorShader()
    renderer = Renderer(width=width, height=height, z_buffering=False)
    renderer.camera = camera
    renderer.shaders = [shader]

    renderer.prepare_polygons(points, indexes)
    renderer.draw_polygons()

    name = "task1.ppm"
    image = PpmImage(name, width, height, renderer.data, depth=depth)

    # ファイルに保存
    with open(name, 'w') as f:
        image.dump(f)