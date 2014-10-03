#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np

from cg.camera import Camera
from cg.ppm import PpmImage
from cg.renderer import Renderer
from cg.shader import DiffuseShader
from cg.vrml import Vrml
from cg.utils import random_polygons


# polygons = random_polygons(3)
# polygons = (
#     # 時計周り
#     np.array(([15, 1, 30], [-15, 1, 30], [1, 12, 25])),
#     # 反時計回り
#     np.array(([15, -1, 30], [-13, -1, 45], [-7, -20, 25])),
#     np.array(([0, 0, 40], [-20, 0, 50], [0, 15, 50])),
#     np.array(([0, 150, 500], [-200, 0, 500], [0, 0, 400])),
# )

# ポリゴンごとに3点の座標を出力する
# for polygon in polygons:
#     print(', '.join(map(str, polygon[:3])), file=sys.stderr)

if __name__ == '__main__':
    vrml = Vrml()
    with open('av2.txt', 'r') as f:
        vrml.load(f)
    width = height = 256
    camera = Camera(positon=(0.0, 0.0, 0.0),
                    angle=(0.0, 0.0, 1.0),
                    focus=256.0)
    shader = DiffuseShader(direction=np.array((-1.0, -1.0, 2.0)),
                           color=(1.0, 0.0, 1.0))
    renderer = Renderer(camera=camera, shader=shader,
                        width=width, height=height)

    # for polygon in polygons:
    for polygon in vrml.polygons:
        print(', '.join(map(str, polygon[:3])), file=sys.stderr)
        renderer.draw_polygon(polygon)

    name = "3d.ppm"
    image = PpmImage(name, width, height, renderer.data)

    # ファイルに保存
    with open(name, 'w') as f:
        image.dump(f)