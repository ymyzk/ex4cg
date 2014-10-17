#!/usr/bin/env python
# -*- coding: utf-8 -*-


class PpmImage(object):
    """PPM 画像を表すクラス"""

    def __init__(self, name, width, height, image, depth=8):
        """
        :param name:
        :type name: str or unicode
        :param int width:
        :param int height:
        :param numpy.ndarray image:
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

        for row in self.image:
            buffer = ['{0:3d} {1:3d} {2:3d}\n'.format(*row[x*3:x*3+3])
                      for x in range(self.width)]
            fp.write(''.join(buffer))