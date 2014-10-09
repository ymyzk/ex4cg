#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

import numpy as np

from cg.camera import Camera
from cg.ppm import PpmImage
from cg.renderer import Renderer
from cg.shader import DiffuseShader
from cg.vrml import Vrml


if __name__ == '__main__':
    vrml = Vrml()
    with open('av3_r.txt', 'r') as f:
        vrml.load(f)
    width = height = 256
    camera = Camera(positon=(0.0, 0.0, 0.0),
                    angle=(0.0, 0.0, 1.0),
                    focus=256.0)
    shader = DiffuseShader(direction=np.array((-1.0, -1.0, 2.0)),
                           luminance=np.array((1.0, 1.0, 1.0)),
                           color=vrml.diffuse_color)
    renderer = Renderer(camera=camera, shader=shader,
                        width=width, height=height)

    for i, polygon in enumerate(vrml.polygons):
        print(', '.join(map(str, polygon[:3])), file=sys.stderr)
        renderer.draw_polygon(polygon)

    name = "3d.ppm"
    image = PpmImage(name, width, height, renderer.data)

    # ファイルに保存
    with open(name, 'w') as f:
        image.dump(f)