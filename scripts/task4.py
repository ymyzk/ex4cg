#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np

from cg.camera import Camera
from cg.ppm import PpmImage
from cg.renderer import Renderer
from cg.shader import (AmbientShader, DiffuseShader, RandomColorShader,
                       ShadingMode, SpecularShader)
from cg.vrml import Vrml


def main(args):
    # VRML ファイルの読み込み
    vrml = Vrml()
    try:
        vrml.load(args.input)
    finally:
        args.input.close()

    width = height = 256
    camera = Camera(position=(0.0, 0.0, 0.0),
                    angle=(0.0, 0.0, 1.0),
                    focus=256.0)
    shaders = []
    if vrml.diffuse_color is not None:
        shaders.append(DiffuseShader(direction=np.array((-1.0, -1.0, 2.0)),
                                     luminance=np.array((1.0, 1.0, 1.0)),
                                     color=vrml.diffuse_color))
    if vrml.specular_color is not None:
        shaders.append(SpecularShader(camera_position=camera.position,
                                      direction=np.array((-1.0, -1.0, 2.0)),
                                      luminance=np.array((1.0, 1.0, 1.0)),
                                      color=vrml.specular_color,
                                      shininess=vrml.specular_color))
    if vrml.ambient_intensity is not None:
        shaders.append(AmbientShader(luminance=np.array((1.0, 1.0, 1.0)),
                                     intensity=vrml.ambient_intensity))
    if len(shaders) == 0:
        shaders.append(RandomColorShader())

    renderer = Renderer(camera=camera, shaders=shaders,
                        width=width, height=height,
                        # shading_mode=ShadingMode.phong)
                        shading_mode=ShadingMode.gouraud)

    for index in vrml.indexes:
        polygon = tuple((map(lambda i: vrml.points[i], index)))
        print(', '.join(map(str, polygon)), file=sys.stderr)

    renderer.draw_polygons(vrml.points, vrml.indexes)

    name = os.path.splitext(args.input.name)[0] + '.ppm'
    image = PpmImage(name, width, height, renderer.data)

    # ファイルに保存
    with open(name, 'w') as f:
        image.dump(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 4')
    parser.add_argument('-o', type=argparse.FileType('w'), metavar='file',
                        default=None, help='Write ppm image to <file>')
    parser.add_argument('input', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='VRML file')
    main(parser.parse_args())