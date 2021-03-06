#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import time

import numpy as np

from cg.camera import Camera
from cg.ppm import PpmImage
from cg.shader import ShadingMode
from cg.vrml import Vrml

from cg.renderer import Renderer
from cg.shader import (AmbientShader, DiffuseShader, RandomColorShader,
                       SpecularShader)


# For line profiler
# @profile
def main(args):
    # プロファイリング
    performance = {'start': time.clock() * 1000}

    # VRML ファイルの読み込み
    vrml = Vrml()
    try:
        vrml.load(args.input)
    finally:
        args.input.close()
    performance['vrml'] = time.clock() * 1000

    # シェーディング方式
    shading_mode = ShadingMode.flat
    if args.s == 'gouraud':
        shading_mode = ShadingMode.gouraud
    elif args.s == 'phong':
        shading_mode = ShadingMode.phong

    width = height = 256
    camera = Camera(position=np.array((0.0, 0.0, 0.0), dtype=np.float64),
                    angle=np.array((0.0, 0.0, 0.0), dtype=np.float64),
                    focus=256.0)
    shaders = []
    if vrml.diffuse_color is not None:
        shaders.append(DiffuseShader(direction=np.array((-1.0, -1.0, 2.0),
                                                        dtype=np.float64),
                                     luminance=np.array((1.0, 1.0, 1.0),
                                                        dtype=np.float64),
                                     color=vrml.diffuse_color))
    if vrml.specular_color is not None:
        shaders.append(SpecularShader(camera_position=camera.position,
                                      direction=np.array((-1.0, -1.0, 2.0),
                                                         dtype=np.float64),
                                      luminance=np.array((1.0, 1.0, 1.0),
                                                         dtype=np.float64),
                                      color=vrml.specular_color,
                                      shininess=vrml.shininess))
    if vrml.ambient_intensity is not None:
        shaders.append(AmbientShader(
            luminance=np.array((1.0, 1.0, 1.0), dtype=np.float64),
            intensity=vrml.ambient_intensity))
    if len(shaders) == 0:
        shaders.append(RandomColorShader())

    renderer = Renderer(width=width, height=height, shading_mode=shading_mode)
    renderer.camera = camera
    renderer.shaders = shaders

    renderer.prepare_polygons(vrml.points, vrml.indexes)
    performance['prepare'] = time.clock() * 1000

    renderer.draw_polygons()
    performance['draw'] = time.clock() * 1000

    image = PpmImage(width, height, renderer.data)

    # ファイルに保存
    if args.o is not None:
        try:
            image.dump(args.o)
        finally:
            args.o.close()
    else:
        with open(os.path.splitext(args.input.name)[0] + '.ppm', 'w') as f:
            image.dump(f)
    performance['ppm'] = time.clock() * 1000

    # パフォーマンスの記録を JSON ファイルに保存
    performance['total'] = performance['ppm'] - performance['start']
    performance['ppm'] -= performance['draw']
    performance['draw'] -= performance['prepare']
    performance['prepare'] -= performance['vrml']
    performance['vrml'] -= performance['start']
    del performance['start']
    if args.p is not None:
        try:
            json.dump(performance, args.p)
        finally:
            args.p.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Task 4 (Python)')
    parser.add_argument('-o', type=argparse.FileType('w'), metavar='file',
                        default=None, help='Write ppm image to <file>')
    parser.add_argument('-p', type=argparse.FileType('w'), metavar='file',
                        default=None,
                        help='Write performance profile to JSON <file>')
    parser.add_argument('-s', choices=['flat', 'gouraud', 'phong'], type=str,
                        default='flat', help='Shading mode')
    parser.add_argument('input', type=argparse.FileType('r'),
                        default=sys.stdin,
                        help='VRML file')
    main(parser.parse_args())