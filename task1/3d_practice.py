#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np

from cg.camera import Camera
from cg.ppm import PpmImage
from cg.utils import random_color, random_polygons


points, polygons = random_polygons(3)

# ポリゴンごとに3点の座標を出力する
for polygon in polygons:
    print(', '.join(map(lambda i: str(points[i]), polygon[:3])),
          file=sys.stderr)

if __name__ == '__main__':
    width = height = 256
    camera = Camera(positon=(0.0, 0.0, 0.0),
                    angle=(0.0, 0.0, 1.0),
                    focus=256.0,
                    width=width,
                    height=height)
    depth = 8
    data = np.zeros((height, width * 3), dtype=np.int8)

    for polygon in polygons:
        color = random_color(depth)
        for point in camera.rasterize([points[polygon[0]],
                                       points[polygon[1]],
                                       points[polygon[2]]]):
            # X: -128 ~ 127 -> (x + 128) -> 0 ~ 255
            # Y: -127 ~ 128 -> (128 - y) -> 0 ~ 255
            data_x = 3 * ((width // 2) + point[0])
            data_y = (height // 2) - point[1]
            data[data_y][data_x:data_x+3] = color

    name = "3d.ppm"
    image = PpmImage(name, width, height, data, depth=depth)

    # ファイルに保存
    with open(name, 'w') as f:
        image.dump(f)