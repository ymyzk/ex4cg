#!/usr/bin/env python
# -*- coding: utf-8 -*-


from cg.ppm import PpmImage


if __name__ == '__main__':
    # 適当な画像を作成
    name = "test.ppm"
    depth = 8
    width = height = 64
    data = [[(i + j) % 2 ** depth for i in range(3 * width)]
            for j in range(height)]
    image = PpmImage(name, width, height, data, depth=depth)

    # ファイルに保存
    with open("test.ppm", 'w') as f:
        image.dump(f)