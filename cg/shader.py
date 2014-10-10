#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from cg.utils import random_color


class Shader(object):
    """シェーダ"""
    @staticmethod
    def _orthogonal_vector(polygon):
        """ポリゴンの直行ベクトルを求める処理"""
        # 直交ベクトル
        # 反時計回りを表
        # cross = np.cross(polygon[0] - polygon[1], polygon[1] - polygon[2])
        # 時計回りを表
        cross = np.cross(polygon[2] - polygon[1], polygon[1] - polygon[0])
        return cross

    @staticmethod
    def _unit_vector(vector):
        """単位ベクトルを求める処理"""
        return vector / np.linalg.norm(vector)


class AmbientShader(Shader):
    """環境光を計算するシェーダ"""
    def __init__(self, luminance, intensity, depth=8):
        """
        :param luminance: 入射光の強さ 0.0-1.0 (r, g, b)
        :param intensity: 環境光係数 0.0-1.0
        :param depth:
        """
        self.luminance = luminance
        self.intensity = intensity * 2 ** (depth - 1)

    def calc(self, polygon):
        return self.intensity * self.luminance


class DiffuseShader(Shader):
    """拡散反射を計算するシェーダ"""
    def __init__(self, direction, luminance, color, depth=8):
        """
        :param direction: 入射光の方向 (x, y, z)
        :param luminance: 入射光の強さ (r, g, b)
        :param color: 拡散反射係数 (r, g, b)
        :param depth:
        """
        # 方向ベクトルを単位ベクトルに変換
        self.direction = self._unit_vector(direction)
        self.luminance = luminance
        self.color = color
        self.depth = depth

    def calc(self, polygon):
        # 直交ベクトル
        cross = self._orthogonal_vector(polygon)
        # 直交ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        if np.count_nonzero(cross) == 0:
            return np.zeros(3)
        # 法線ベクトル (単位ベクトル化)
        normal = self._unit_vector(cross)
        # 反射光を計算
        cos = -np.dot(self.direction, normal)
        # ポリゴンが裏を向いているときは, 反射光なし
        if cos < 0:
            return np.zeros(3)
        diffuse = (2 ** self.depth - 1) * cos * self.color * self.luminance
        return diffuse


class RandomColorShader(Shader):
    """ランダムな色を返すシェーダ"""
    def __init__(self, depth=8):
        self.depth = depth

    def calc(self, polygon):
        return random_color(self.depth)


class SpecularShader(Shader):
    """鏡面反射を計算するシェーダ"""
    def __init__(self, camera_position, direction, luminance, color, shininess,
                 depth=8):
        """
        :param camera_position: カメラの位置 (x, y, z)
        :param direction: 入射光の方向 (x, y, z)
        :param luminance: 入射光の強さ (r, g, b)
        :param color: 鏡面反射係数 (r, g, b)
        :param shininess: 鏡面反射強度 s
        :param depth:
        """
        self.camera_position = camera_position
        # 方向ベクトルを単位ベクトルに変換
        self.direction = self._unit_vector(direction)
        self.luminance = luminance
        self.color = color
        self.shininess = shininess * 128
        self.depth = depth

    def calc(self, polygon):
        # ポリゴンの直交ベクトル
        cross = self._orthogonal_vector(polygon)
        # 直交ベクトルがゼロベクトルであれば, 計算不能 (ex. 面積0のポリゴン)
        if np.count_nonzero(cross) == 0:
            return np.zeros(3)
        # ポリゴンの法線ベクトル (単位ベクトル化)
        normal = self._unit_vector(cross)
        # ポリゴンの重心
        g = (polygon[0] + polygon[1] + polygon[2]) / 3
        # ポリゴンの重心から視点への単位方向ベクトル
        e = self._unit_vector(self.camera_position - g)
        s = e - self.direction
        s /= np.linalg.norm(s)
        sn = np.dot(s, normal)
        # ポリゴンが裏を向いているときは, 反射光なし
        if sn < 0:
            return np.zeros(3)
        specular = sn ** self.shininess * self.color * self.luminance
        return (2 ** self.depth - 1) * specular