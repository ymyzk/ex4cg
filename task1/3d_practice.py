#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from cg.camera import Camera
from cg.ppm import PpmImage
from cg.renderer import Renderer
from cg.utils import random_polygons


points, polygons = random_polygons(3)

# ポリゴンごとに3点の座標を出力する
for polygon in polygons:
    print(', '.join(map(lambda i: str(points[i]), polygon[:3])),
          file=sys.stderr)

if __name__ == '__main__':
    width = height = 256
    depth = 8
    camera = Camera(positon=(0.0, 0.0, 0.0),
                    angle=(0.0, 0.0, 1.0),
                    focus=256.0)
    renderer = Renderer(camera=camera, depth=depth, width=width, height=height)

    for polygon in polygons:
        renderer.draw_polygon((points[polygon[0]],
                               points[polygon[1]],
                               points[polygon[2]]))

    name = "3d.ppm"
    image = PpmImage(name, width, height, renderer.data, depth=depth)

    # ファイルに保存
    with open(name, 'w') as f:
        image.dump(f)