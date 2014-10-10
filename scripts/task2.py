#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys

import numpy as np

from cg.camera import Camera
from cg.ppm import PpmImage
from cg.renderer import Renderer
from cg.shader import DiffuseShader, RandomColorShader
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
    if vrml.diffuse_color is None:
        shader = RandomColorShader()
    else:
        shader = DiffuseShader(direction=np.array((-1.0, -1.0, 2.0)),
                               luminance=np.array((1.0, 1.0, 1.0)),
                               color=vrml.diffuse_color)

    renderer = Renderer(camera=camera, shaders=[shader],
                        width=width, height=height)

    for polygon in vrml.polygons:
        print(', '.join(map(str, polygon[:3])), file=sys.stderr)

    renderer.draw_polygons(vrml.polygons)

    name = os.path.splitext(args.input.name)[0] + '.ppm'
    image = PpmImage(name, width, height, renderer.data)

    # ファイルに保存
    with open(name, 'w') as f:
        image.dump(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 2')
    parser.add_argument('-o', type=argparse.FileType('w'), metavar='file',
                        default=None, help='Write ppm image to <file>')
    parser.add_argument('input', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='VRML file')
    main(parser.parse_args())