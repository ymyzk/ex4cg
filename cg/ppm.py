#!/usr/bin/env python
# -*- coding: utf-8 -*-


class PpmImage(object):
    """PPM 画像を表すクラス"""

    def __init__(self, name, width, height, image, depth=8):
        """
        :param name:
        :param width:
        :param height:
        :param image:
        :param depth depth: 各色の階調数 (bit)
        :return:
        """
        self.name = name
        self.width = width
        self.height = height
        self.image = image
        self.depth = depth

    def dump(self, fp):
        """ファイルに画像データを書き込む処理"""
        fp.write('P3\n')
        fp.write('# ' + self.name + '\n')
        fp.write('{0:d} {1:d}\n'.format(self.width, self.height))
        fp.write('{0:d}\n'.format(2 ** self.depth - 1))

        # 画像の高さが不十分であれば例外を送出
        if len(self.image) != self.height:
            raise IndexError()
        for row in self.image:
            # 画像の幅が不十分であれば例外を送出
            if len(row) != 3 * self.width:
                raise IndexError()
            for x in range(0, self.width * 3, 3):
                fp.write('{0:3d} {1:3d} {2:3d}\n'.format(*row[x:x+3]))