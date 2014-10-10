#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

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
    camera = Camera(position=(0.0, 0.0, 0.0),
                    angle=(0.0, 0.0, 1.0),
                    focus=256.0)
    shader = RandomColorShader(depth=depth)
    renderer = Renderer(camera=camera, shaders=[shader], depth=depth,
                        width=width, height=height, zbuffering=False)

    renderer.draw_polygons(points, indexes)

    name = "task1.ppm"
    image = PpmImage(name, width, height, renderer.data, depth=depth)

    # ファイルに保存
    with open(name, 'w') as f:
        image.dump(f)