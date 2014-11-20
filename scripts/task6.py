#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals
import sys

import numpy as np
from PyQt4 import QtCore, QtGui

from cg import shader
from cg.camera import Camera
from cg.cython import Renderer as CyRenderer
from cg.ppm import PpmImage
from cg.renderer import Renderer as PyRenderer
from cg.shader import ShadingMode
from cg.vrml import Vrml


depth = 8


class ImageWidget(QtGui.QLabel):
    def __init__(self):
        super().__init__()

    def set_image(self, image, width, height):
        self.setFixedSize(width, height)
        self.setPixmap(QtGui.QPixmap(
            QtGui.QImage(image, width, height, QtGui.QImage.Format_RGB888)))


class SeekBarWidget(QtGui.QWidget):
    current_frame_changed = QtCore.pyqtSignal(int)
    key_frame_changed = QtCore.pyqtSignal(int, bool)

    def __init__(self):
        super().__init__()

        layout = QtGui.QHBoxLayout()
        self.setLayout(layout)

        self._seek_slider = QtGui.QSlider(1, self)
        self._seek_slider.setMinimum(0)
        self._seek_slider.setMaximum(29)
        self._seek_slider.setValue(0)
        self._seek_slider.valueChanged.connect(self._slider_changed)
        layout.addWidget(self._seek_slider)

        self._key_frame_checkbox = QtGui.QCheckBox('Key Frame')
        self._key_frame_checkbox.setCheckState(0)
        self._key_frame_checkbox.stateChanged.connect(self._key_frame_changed)
        layout.addWidget(self._key_frame_checkbox)

        self._current_frame = QtGui.QSpinBox()
        self._current_frame.setMinimum(0)
        self._current_frame.setMaximum(29)
        self._current_frame.setValue(0)
        self._current_frame.valueChanged.connect(self._current_frame_changed)
        layout.addWidget(self._current_frame)

        self._num_frames = QtGui.QSpinBox()
        self._num_frames.setMinimum(0)
        self._num_frames.setMaximum(1000)
        self._num_frames.setValue(30)
        self._num_frames.valueChanged.connect(self._num_frames_changed)
        layout.addWidget(self._num_frames)

    @property
    def current_frame(self):
        return self._current_frame.value()

    @current_frame.setter
    def current_frame(self, frame):
        self._current_frame.setValue(frame)
        self._seek_slider.setValue(frame)

    @property
    def num_frames(self):
        return self._num_frames.value()

    @num_frames.setter
    def num_frames(self, frames):
        self._seek_slider.setValue(min(self.current_frame, frames - 1))
        self._seek_slider.setMaximum(frames - 1)
        self._current_frame.setMaximum(frames - 1)
        self._num_frames.setValue(frames)

    @property
    def is_key_frame(self):
        return True if self._key_frame_checkbox.checkState() == 2 else False

    @is_key_frame.setter
    def is_key_frame(self, b):
        status = self._key_frame_checkbox.blockSignals(True)
        self._key_frame_checkbox.setCheckState(2 if b else 0)
        self._key_frame_checkbox.blockSignals(status)

    def _key_frame_changed(self, i):
        self.key_frame_changed.emit(self.current_frame,
                                    True if i == 2 else False)

    def _slider_changed(self, i):
        self._current_frame.setValue(i)
        self.current_frame_changed.emit(i)

    def _current_frame_changed(self, i):
        self._seek_slider.setValue(i)
        self.current_frame_changed.emit(i)

    def _num_frames_changed(self, i):
        self.num_frames = i


class Application(object):
    def __init__(self):
        self.vrml = Vrml()
        self.renderer = None
        self.data = None
        self.width = self.height = 256
        # キーフレームの辞書
        self.key_frames = {}
        # アニメーション用のタイマー
        self.timer = None
        # アニメーションに用いるフレームの配列
        self.frames = []
        # アニメーション中のフレーム番号
        self.frame_i = 0

        # UI
        self.app = QtGui.QApplication(sys.argv)
        self.main_window = QtGui.QMainWindow()

        # Main
        main_panel = QtGui.QWidget()
        main_panel_layout = QtGui.QVBoxLayout()
        main_panel.setLayout(main_panel_layout)

        # Menu Bar
        menu_bar = self.main_window.menuBar()

        menu_file = menu_bar.addMenu('&File')

        open_action = QtGui.QAction('&Open', self.main_window)
        open_action.setShortcut('Ctrl+O')
        open_action.triggered.connect(self.open_vrml)
        menu_file.addAction(open_action)

        save_action = QtGui.QAction('&Save', self.main_window)
        save_action.setShortcut('Ctrl+S')
        save_action.triggered.connect(self.save_ppm)
        menu_file.addAction(save_action)

        close_action = QtGui.QAction('&Close', self.main_window)
        close_action.setShortcut('Ctrl+W')
        close_action.triggered.connect(QtGui.qApp.quit)
        menu_file.addAction(close_action)

        menu_render = menu_bar.addMenu('&Render')

        render_action = QtGui.QAction('&Render', self.main_window)
        render_action.setShortcut('Ctrl+R')
        render_action.triggered.connect(self.render)
        menu_render.addAction(render_action)

        menu_animate = menu_bar.addMenu('&Animate')

        animate_action = QtGui.QAction('&Animate', self.main_window)
        animate_action.setShortcut('Ctrl+T')
        animate_action.triggered.connect(self.animate)
        menu_animate.addAction(animate_action)

        # Status Bar
        self.status_bar = QtGui.QStatusBar()
        self.main_window.setStatusBar(self.status_bar)
        self.status_bar.showMessage('Ready.')

        # Image
        image_panel = QtGui.QWidget()
        image_panel_layout = QtGui.QHBoxLayout()
        image_panel.setLayout(image_panel_layout)
        main_panel_layout.addWidget(image_panel)

        self.image_label = ImageWidget()
        image_panel_layout.addStretch()
        image_panel_layout.addWidget(self.image_label)
        image_panel_layout.addStretch()

        # Seek Bar
        self.seek_bar = SeekBarWidget()
        self.seek_bar.key_frame_changed.connect(self.key_frame_changed)
        self.seek_bar.current_frame_changed.connect(self.frame_changed)
        main_panel_layout.addWidget(self.seek_bar)

        # Tab
        control_tab = QtGui.QTabWidget()
        main_panel_layout.addWidget(control_tab)

        # File Tab
        file_panel = QtGui.QWidget()
        file_panel_layout = QtGui.QGridLayout()
        file_panel.setLayout(file_panel_layout)
        control_tab.addTab(file_panel, 'File')

        # Open file button
        open_file_button = QtGui.QPushButton('Open VRML file...')
        open_file_button.clicked.connect(self.open_vrml)
        file_panel_layout.addWidget(open_file_button, 0, 0)

        self.file_label = QtGui.QLabel()
        self.file_label.setText("Please select VRML file")
        file_panel_layout.addWidget(self.file_label, 1, 0)

        save_file_button = QtGui.QPushButton('Save PPM file...')
        save_file_button.clicked.connect(self.save_ppm)
        file_panel_layout.addWidget(save_file_button, 2, 0)

        # Camera Tab
        camera_panel = QtGui.QWidget()
        camera_panel_layout = QtGui.QGridLayout()
        camera_panel.setLayout(camera_panel_layout)
        control_tab.addTab(camera_panel, 'Camera')

        camera_panel_layout.addWidget(
            QtGui.QLabel('Position (x, y, z): '), 0, 0)
        self.camera_position_x = QtGui.QDoubleSpinBox()
        self.camera_position_x.setMinimum(-32768.0)
        self.camera_position_x.setMaximum(32767.0)
        self.camera_position_x.setValue(0.0)
        self.camera_position_x.valueChanged.connect(self.value_changed)
        camera_panel_layout.addWidget(self.camera_position_x, 0, 1)
        self.camera_position_y = QtGui.QDoubleSpinBox()
        self.camera_position_y.setMinimum(-32768.0)
        self.camera_position_y.setMaximum(32767.0)
        self.camera_position_y.setValue(0.0)
        self.camera_position_y.valueChanged.connect(self.value_changed)
        camera_panel_layout.addWidget(self.camera_position_y, 0, 2)
        self.camera_position_z = QtGui.QDoubleSpinBox()
        self.camera_position_z.setMinimum(-32768.0)
        self.camera_position_z.setMaximum(32767.0)
        self.camera_position_z.setValue(0.0)
        self.camera_position_z.valueChanged.connect(self.value_changed)
        camera_panel_layout.addWidget(self.camera_position_z, 0, 3)

        camera_panel_layout.addWidget(
            QtGui.QLabel('Angle (x, y, z): '), 1, 0)
        self.camera_angle_x = QtGui.QDoubleSpinBox()
        self.camera_angle_x.setMinimum(-360.0)
        self.camera_angle_x.setMaximum(360.0)
        self.camera_angle_x.setValue(0.0)
        self.camera_angle_x.valueChanged.connect(self.value_changed)
        camera_panel_layout.addWidget(self.camera_angle_x, 1, 1)
        self.camera_angle_y = QtGui.QDoubleSpinBox()
        self.camera_angle_y.setMinimum(-360.0)
        self.camera_angle_y.setMaximum(360.0)
        self.camera_angle_y.setValue(0.0)
        self.camera_angle_y.valueChanged.connect(self.value_changed)
        camera_panel_layout.addWidget(self.camera_angle_y, 1, 2)
        self.camera_angle_z = QtGui.QDoubleSpinBox()
        self.camera_angle_z.setMinimum(-360.0)
        self.camera_angle_z.setMaximum(360.0)
        self.camera_angle_z.setValue(0.0)
        self.camera_angle_z.valueChanged.connect(self.value_changed)
        camera_panel_layout.addWidget(self.camera_angle_z, 1, 3)

        camera_panel_layout.addWidget(
            QtGui.QLabel('Focus (f): '), 2, 0)
        self.camera_focus = QtGui.QDoubleSpinBox()
        self.camera_focus.setMaximum(1024.0)
        self.camera_focus.setValue(256.0)
        self.camera_focus.valueChanged.connect(self.value_changed)
        camera_panel_layout.addWidget(self.camera_focus, 2, 1)

        # Diffuse Tab
        diffuse_panel = QtGui.QWidget()
        diffuse_panel_layout = QtGui.QGridLayout()
        diffuse_panel.setLayout(diffuse_panel_layout)
        control_tab.addTab(diffuse_panel, 'Diffuse Shader')

        self.diffuse_checkbox = QtGui.QCheckBox('Enable')
        self.diffuse_checkbox.setCheckState(2)
        self.diffuse_checkbox.stateChanged.connect(self.value_changed)
        diffuse_panel_layout.addWidget(self.diffuse_checkbox, 0, 0)

        diffuse_panel_layout.addWidget(
            QtGui.QLabel('Direction (x, y, z): '), 1, 0)
        self.diffuse_direction_x = QtGui.QDoubleSpinBox()
        self.diffuse_direction_x.setMinimum(-128.0)
        self.diffuse_direction_x.setMaximum(127.0)
        self.diffuse_direction_x.setValue(-1.0)
        self.diffuse_direction_x.valueChanged.connect(self.value_changed)
        diffuse_panel_layout.addWidget(self.diffuse_direction_x, 1, 1)
        self.diffuse_direction_y = QtGui.QDoubleSpinBox()
        self.diffuse_direction_y.setMinimum(-128.0)
        self.diffuse_direction_y.setMaximum(127.0)
        self.diffuse_direction_y.setValue(-1.0)
        self.diffuse_direction_y.valueChanged.connect(self.value_changed)
        diffuse_panel_layout.addWidget(self.diffuse_direction_y, 1, 2)
        self.diffuse_direction_z = QtGui.QDoubleSpinBox()
        self.diffuse_direction_z.setMinimum(-128.0)
        self.diffuse_direction_z.setMaximum(127.0)
        self.diffuse_direction_z.setValue(2.0)
        self.diffuse_direction_z.valueChanged.connect(self.value_changed)
        diffuse_panel_layout.addWidget(self.diffuse_direction_z, 1, 3)

        diffuse_panel_layout.addWidget(
            QtGui.QLabel('Luminance (r, g, b): '), 2, 0)
        self.diffuse_luminance_r = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_r.setMinimum(0.0)
        self.diffuse_luminance_r.setMaximum(1.0)
        self.diffuse_luminance_r.setSingleStep(0.1)
        self.diffuse_luminance_r.setValue(1.0)
        self.diffuse_luminance_r.valueChanged.connect(self.value_changed)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_r, 2, 1)
        self.diffuse_luminance_g = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_g.setMinimum(0.0)
        self.diffuse_luminance_g.setMaximum(1.0)
        self.diffuse_luminance_g.setSingleStep(0.1)
        self.diffuse_luminance_g.setValue(1.0)
        self.diffuse_luminance_g.valueChanged.connect(self.value_changed)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_g, 2, 2)
        self.diffuse_luminance_b = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_b.setMinimum(0.0)
        self.diffuse_luminance_b.setMaximum(1.0)
        self.diffuse_luminance_b.setSingleStep(0.1)
        self.diffuse_luminance_b.setValue(1.0)
        self.diffuse_luminance_b.valueChanged.connect(self.value_changed)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_b, 2, 3)

        # Specular Tab
        specular_panel = QtGui.QWidget()
        specular_panel_layout = QtGui.QGridLayout()
        specular_panel.setLayout(specular_panel_layout)
        control_tab.addTab(specular_panel, 'Sqecular Shader')

        self.specular_checkbox = QtGui.QCheckBox('Enable')
        self.specular_checkbox.setCheckState(2)
        self.specular_checkbox.stateChanged.connect(self.value_changed)
        specular_panel_layout.addWidget(self.specular_checkbox, 0, 0)

        specular_panel_layout.addWidget(
            QtGui.QLabel('Direction (x, y, z): '), 1, 0)
        self.specular_direction_x = QtGui.QDoubleSpinBox()
        self.specular_direction_x.setMinimum(-128.0)
        self.specular_direction_x.setMaximum(127.0)
        self.specular_direction_x.setValue(-1.0)
        self.specular_direction_x.valueChanged.connect(self.value_changed)
        specular_panel_layout.addWidget(self.specular_direction_x, 1, 1)
        self.specular_direction_y = QtGui.QDoubleSpinBox()
        self.specular_direction_y.setMinimum(-128.0)
        self.specular_direction_y.setMaximum(127.0)
        self.specular_direction_y.setValue(-1.0)
        self.specular_direction_y.valueChanged.connect(self.value_changed)
        specular_panel_layout.addWidget(self.specular_direction_y, 1, 2)
        self.specular_direction_z = QtGui.QDoubleSpinBox()
        self.specular_direction_z.setMinimum(-128.0)
        self.specular_direction_z.setMaximum(127.0)
        self.specular_direction_z.setValue(2.0)
        self.specular_direction_z.valueChanged.connect(self.value_changed)
        specular_panel_layout.addWidget(self.specular_direction_z, 1, 3)

        specular_panel_layout.addWidget(QtGui.QLabel(
            'Luminance (r, g, b): '), 2, 0)
        self.specular_luminance_r = QtGui.QDoubleSpinBox()
        self.specular_luminance_r.setMinimum(0.0)
        self.specular_luminance_r.setMaximum(1.0)
        self.specular_luminance_r.setSingleStep(0.1)
        self.specular_luminance_r.setValue(1.0)
        self.specular_luminance_r.valueChanged.connect(self.value_changed)
        specular_panel_layout.addWidget(self.specular_luminance_r, 2, 1)
        self.specular_luminance_g = QtGui.QDoubleSpinBox()
        self.specular_luminance_g.setMinimum(0.0)
        self.specular_luminance_g.setMaximum(1.0)
        self.specular_luminance_g.setSingleStep(0.1)
        self.specular_luminance_g.setValue(1.0)
        self.specular_luminance_g.valueChanged.connect(self.value_changed)
        specular_panel_layout.addWidget(self.specular_luminance_g, 2, 2)
        self.specular_luminance_b = QtGui.QDoubleSpinBox()
        self.specular_luminance_b.setMinimum(0.0)
        self.specular_luminance_b.setMaximum(1.0)
        self.specular_luminance_b.setSingleStep(0.1)
        self.specular_luminance_b.setValue(1.0)
        self.specular_luminance_b.valueChanged.connect(self.value_changed)
        specular_panel_layout.addWidget(self.specular_luminance_b, 2, 3)

        # Ambient Tab
        ambient_panel = QtGui.QWidget()
        ambient_panel_layout = QtGui.QGridLayout()
        ambient_panel.setLayout(ambient_panel_layout)
        control_tab.addTab(ambient_panel, 'Ambient Shader')

        self.ambient_checkbox = QtGui.QCheckBox('Enable')
        self.ambient_checkbox.setCheckState(2)
        self.ambient_checkbox.stateChanged.connect(self.value_changed)
        ambient_panel_layout.addWidget(self.ambient_checkbox, 0, 0)

        ambient_panel_layout.addWidget(
            QtGui.QLabel('Luminance (r, g, b): '), 1, 0)
        self.ambient_luminance_r = QtGui.QDoubleSpinBox()
        self.ambient_luminance_r.setMinimum(0.0)
        self.ambient_luminance_r.setMaximum(1.0)
        self.ambient_luminance_r.setSingleStep(0.1)
        self.ambient_luminance_r.setValue(1.0)
        self.ambient_luminance_r.valueChanged.connect(self.value_changed)
        ambient_panel_layout.addWidget(self.ambient_luminance_r, 1, 1)
        self.ambient_luminance_g = QtGui.QDoubleSpinBox()
        self.ambient_luminance_g.setMinimum(0.0)
        self.ambient_luminance_g.setMaximum(1.0)
        self.ambient_luminance_g.setSingleStep(0.1)
        self.ambient_luminance_g.setValue(1.0)
        self.ambient_luminance_g.valueChanged.connect(self.value_changed)
        ambient_panel_layout.addWidget(self.ambient_luminance_g, 1, 2)
        self.ambient_luminance_b = QtGui.QDoubleSpinBox()
        self.ambient_luminance_b.setMinimum(0.0)
        self.ambient_luminance_b.setMaximum(1.0)
        self.ambient_luminance_b.setSingleStep(0.1)
        self.ambient_luminance_b.setValue(1.0)
        self.ambient_luminance_b.valueChanged.connect(self.value_changed)
        ambient_panel_layout.addWidget(self.ambient_luminance_b, 1, 3)

        # Render Tab
        render_panel = QtGui.QWidget()
        render_panel_layout = QtGui.QGridLayout()
        render_panel.setLayout(render_panel_layout)
        control_tab.addTab(render_panel, 'Render')

        shading_mode = QtGui.QGroupBox()
        shading_mode_layout = QtGui.QHBoxLayout()
        shading_mode.setLayout(shading_mode_layout)
        shading_mode.setTitle('Shading Mode')
        render_panel_layout.addWidget(shading_mode, 0, 0, 1, 3)

        self.shading_mode_flat = QtGui.QRadioButton('Flat (Constant)')
        self.shading_mode_flat.setChecked(1)
        self.shading_mode_flat.toggled.connect(self.value_changed)
        shading_mode_layout.addWidget(self.shading_mode_flat)
        self.shading_mode_gouraud = QtGui.QRadioButton('Gouraud')
        self.shading_mode_gouraud.toggled.connect(self.value_changed)
        shading_mode_layout.addWidget(self.shading_mode_gouraud)
        self.shading_mode_phong = QtGui.QRadioButton('Phong')
        self.shading_mode_phong.toggled.connect(self.value_changed)
        shading_mode_layout.addWidget(self.shading_mode_phong)

        backend = QtGui.QGroupBox()
        backend_layout = QtGui.QHBoxLayout()
        backend.setLayout(backend_layout)
        backend.setTitle('Backend')
        render_panel_layout.addWidget(backend, 1, 0, 1, 3)

        self.backend_python = QtGui.QRadioButton('Python')
        self.backend_python.setChecked(1)
        backend_layout.addWidget(self.backend_python)
        self.backend_cython = QtGui.QRadioButton('Python + Cython')
        backend_layout.addWidget(self.backend_cython)

        render_panel_layout.addWidget(
            QtGui.QLabel('Size (w, h): '), 2, 0)
        self.render_size_w = QtGui.QSpinBox()
        self.render_size_w.setMinimum(0)
        self.render_size_w.setMaximum(512)
        self.render_size_w.setValue(self.width)
        self.render_size_w.valueChanged.connect(self.value_changed)
        render_panel_layout.addWidget(self.render_size_w, 2, 1)
        self.render_size_h = QtGui.QSpinBox()
        self.render_size_h.setMinimum(0)
        self.render_size_h.setMaximum(512)
        self.render_size_h.setValue(self.height)
        self.render_size_h.valueChanged.connect(self.value_changed)
        render_panel_layout.addWidget(self.render_size_h, 2, 2)

        render_button = QtGui.QPushButton('Render')
        render_button.clicked.connect(self.render)
        render_panel_layout.addWidget(render_button, 3, 0, 1, 3)

        # Animate Tab
        animate_panel = QtGui.QWidget()
        animate_panel_layout = QtGui.QGridLayout()
        animate_panel.setLayout(animate_panel_layout)
        control_tab.addTab(animate_panel, 'Animate')

        animate_panel_layout.addWidget(
            QtGui.QLabel('Animation is executed by Python + Cython.'), 0, 0)
        animate_panel_layout.addWidget(
            QtGui.QLabel('Frames per second: '), 1, 0)
        self.animate_fps = QtGui.QSpinBox()
        self.animate_fps.setMinimum(0)
        self.animate_fps.setMaximum(300)
        self.animate_fps.setValue(30)
        animate_panel_layout.addWidget(self.animate_fps, 1, 1, 1, 1)

        animate_button = QtGui.QPushButton('Animate')
        animate_button.clicked.connect(self.animate)
        animate_panel_layout.addWidget(animate_button, 2, 0, 1, 2)

        self.main_window.setCentralWidget(main_panel)

    def run(self):
        self.main_window.show()
        self.app.exec_()

    def open_vrml(self):
        file_path = QtGui.QFileDialog.getOpenFileName()

        if file_path == '':
            return

        with open(file_path) as f:
            self.vrml.load(f)
        self.file_label.setText(file_path)
        self.render()

    def save_ppm(self):
        file_path = QtGui.QFileDialog.getSaveFileName()

        if file_path == '':
            return

        image = PpmImage(file_path, self.width, self.height, self.data)

        with open(file_path, 'w') as f:
            image.dump(f)

    def value_changed(self):
        self._render(self.get_frame(), is_cython=True)

    def frame_changed(self, f):
        self.seek_bar.is_key_frame = f in self.key_frames

        # アニメーション中
        if self.timer is not None or len(self.frames) == 0:
            return

        frame = self.frames[f]
        self.restore_frame(frame)
        self._render(frame, is_cython=True)

    def key_frame_changed(self, i, b):
        if b:
            self.add_key_frame(i)
            self.interpoate()
        else:
            self.del_key_frame(i)
            self.interpoate()

    def render(self):
        self.status_bar.showMessage('Rendering..')
        self._render(self.get_frame())
        self.status_bar.showMessage('Rendered.')

    def animate(self):
        """アニメーション処理を開始する処理"""
        # アニメーションスタート
        self.frame_i = 0
        self.timer = QtCore.QTimer()
        self.main_window.connect(self.timer, QtCore.SIGNAL('timeout()'),
                                 self._animate)
        self.timer.start(1000 / self.animate_fps.value())

    def _render(self, frame, is_cython=False):
        if is_cython or self.backend_cython.isChecked():
            Renderer = CyRenderer
        else:
            Renderer = PyRenderer

        mode = ShadingMode.flat
        if self.shading_mode_flat.isChecked():
            mode = ShadingMode.flat
        elif self.shading_mode_gouraud.isChecked():
            mode = ShadingMode.gouraud
        elif self.shading_mode_phong.isChecked():
            mode = ShadingMode.phong

        self.width = self.render_size_w.value()
        self.height = self.render_size_h.value()

        self.renderer = Renderer(width=self.width, height=self.height,
                                 shading_mode=mode)
        self.renderer.prepare_polygons(self.vrml.points, self.vrml.indexes)

        camera = Camera(
            position=np.array((
                frame['camera_position_x'],
                frame['camera_position_y'],
                frame['camera_position_z'])),
            angle=np.array((
                frame['camera_angle_x'],
                frame['camera_angle_y'],
                frame['camera_angle_z'])),
            focus=frame['camera_focus'])
        self.renderer.camera = camera

        shaders = []
        if frame['diffuse']:
            shaders.append(shader.DiffuseShader(
                direction=np.array((
                    frame['diffuse_direction_x'],
                    frame['diffuse_direction_y'],
                    frame['diffuse_direction_z'])),
                luminance=np.array((
                    frame['diffuse_luminance_r'],
                    frame['diffuse_luminance_g'],
                    frame['diffuse_luminance_b'])),
                color=self.vrml.diffuse_color))
        if (frame['specular'] and
                self.vrml.specular_color is not None and
                self.vrml.shininess is not None):
            shaders.append(shader.SpecularShader(
                camera_position=camera.position,
                direction=np.array((
                    frame['specular_direction_x'],
                    frame['specular_direction_y'],
                    frame['specular_direction_z'])),
                luminance=np.array((
                    frame['specular_luminance_r'],
                    frame['specular_luminance_g'],
                    frame['specular_luminance_b'])),
                color=self.vrml.specular_color,
                shininess=self.vrml.shininess))
        if (frame['ambient'] and
                self.vrml.ambient_intensity is not None):
            shaders.append(shader.AmbientShader(
                luminance=np.array((
                    frame['ambient_luminance_r'],
                    frame['ambient_luminance_g'],
                    frame['ambient_luminance_b'])),
                intensity=self.vrml.ambient_intensity))
        if len(shaders) == 0:
            shaders.append(shader.RandomColorShader())
        self.renderer.shaders = shaders

        self.renderer.draw_polygons()
        self.data = self.renderer.data
        self.image_label.set_image(self.data, self.width, self.height)
        self.renderer.clear()

    def get_frame(self):
        """フレームの情報を UI から取得する処理"""
        info = {
            'camera_position_x': self.camera_position_x.value(),
            'camera_position_y': self.camera_position_y.value(),
            'camera_position_z': self.camera_position_z.value(),
            'camera_angle_x': self.camera_angle_x.value(),
            'camera_angle_y': self.camera_angle_y.value(),
            'camera_angle_z': self.camera_angle_z.value(),
            'camera_focus': self.camera_focus.value(),
            'diffuse': self.diffuse_checkbox.checkState() == 2,
            'diffuse_direction_x': self.diffuse_direction_x.value(),
            'diffuse_direction_y': self.diffuse_direction_y.value(),
            'diffuse_direction_z': self.diffuse_direction_z.value(),
            'diffuse_luminance_r': self.diffuse_luminance_r.value(),
            'diffuse_luminance_g': self.diffuse_luminance_g.value(),
            'diffuse_luminance_b': self.diffuse_luminance_b.value(),
            'specular': self.specular_checkbox.checkState() == 2,
            'specular_direction_x': self.specular_direction_x.value(),
            'specular_direction_y': self.specular_direction_y.value(),
            'specular_direction_z': self.specular_direction_z.value(),
            'specular_luminance_r': self.specular_luminance_r.value(),
            'specular_luminance_g': self.specular_luminance_g.value(),
            'specular_luminance_b': self.specular_luminance_b.value(),
            'ambient': self.ambient_checkbox.checkState() == 2,
            'ambient_luminance_r': self.ambient_luminance_r.value(),
            'ambient_luminance_g': self.ambient_luminance_g.value(),
            'ambient_luminance_b': self.ambient_luminance_b.value()
        }
        return info

    def restore_frame(self, frame):
        """フレームの情報を UI に反映させる処理"""
        self.camera_position_x.setValue(frame['camera_position_x'])
        self.camera_position_y.setValue(frame['camera_position_y'])
        self.camera_position_z.setValue(frame['camera_position_z'])
        self.camera_angle_x.setValue(frame['camera_angle_x'])
        self.camera_angle_y.setValue(frame['camera_angle_y'])
        self.camera_angle_z.setValue(frame['camera_angle_z'])
        self.camera_focus.setValue(frame['camera_focus'])
        self.diffuse_checkbox.setCheckState(2 if frame['diffuse'] else 0)
        self.diffuse_direction_x.setValue(frame['diffuse_direction_x'])
        self.diffuse_direction_y.setValue(frame['diffuse_direction_y'])
        self.diffuse_direction_z.setValue(frame['diffuse_direction_z'])
        self.diffuse_luminance_r.setValue(frame['diffuse_luminance_r'])
        self.diffuse_luminance_g.setValue(frame['diffuse_luminance_g'])
        self.diffuse_luminance_b.setValue(frame['diffuse_luminance_b'])
        self.specular_checkbox.setCheckState(2 if frame['specular'] else 0)
        self.specular_direction_x.setValue(frame['specular_direction_x'])
        self.specular_direction_y.setValue(frame['specular_direction_y'])
        self.specular_direction_z.setValue(frame['specular_direction_z'])
        self.specular_luminance_r.setValue(frame['specular_luminance_r'])
        self.specular_luminance_g.setValue(frame['specular_luminance_g'])
        self.specular_luminance_b.setValue(frame['specular_luminance_b'])
        self.ambient_checkbox.setCheckState(2 if frame['ambient'] else 0)
        self.ambient_luminance_r.setValue(frame['ambient_luminance_r'])
        self.ambient_luminance_g.setValue(frame['ambient_luminance_g'])
        self.ambient_luminance_b.setValue(frame['ambient_luminance_b'])

    def add_key_frame(self, i):
        """キーフレームを追加する処理"""
        self.key_frames[i] = self.get_frame()

    def del_key_frame(self, i):
        """キーフレームを削除する処理"""
        if i in self.key_frames:
            del self.key_frames[i]

    def _interpolate(self, a, b, r):
        """キーフレーム間のフレームを描画するための情報を補完する処理"""
        result = {}
        for k in a:
            if isinstance(a[k], (int, float)):
                result[k] = (1 - r) * a[k] + r * b[k]
            else:
                result[k] = a[k]
        return result

    def interpoate(self):
        """キーフレーム間のフレームを保管する処理"""
        # キーフレームをもとに補完する
        keys = sorted(self.key_frames.keys())
        key_frames = self.key_frames.copy()

        # 先頭フレームがキーフレームでなければ
        # 先頭フレームのキーフレームを, 最初のキーフレームと同じとして扱う
        if keys[0] != 0:
            key_frames[0] = key_frames[keys[0]]
            keys = [0] + keys

        # 同様に末尾フレームを補完
        if keys[-1] < self.seek_bar.num_frames - 1:
            key_frames[self.seek_bar.num_frames - 1] = key_frames[keys[-1]]
            keys += [self.seek_bar.num_frames - 1]
        self.frames = []
        for i in range(len(keys) - 1):
            ka = keys[i]
            kb = keys[i + 1]
            fa = key_frames[ka]
            fb = key_frames[kb]
            for kc in range(ka, kb):
                # a -- (r) -- c -- (1-r) -- b
                r = (kc - ka) / (kb - ka)
                self.frames.append(self._interpolate(fa, fb, r))
        self.frames.append(key_frames[keys[-1]])

    def _animate(self):
        """アニメーションの1フレームを描画する処理"""
        if len(self.frames) <= self.frame_i:
            self.timer.stop()
            self.timer = None
            return

        self._render(self.frames[self.frame_i], is_cython=True)

        self.status_bar.showMessage('Rendered {0}'.format(self.frame_i + 1))

        self.seek_bar.current_frame = self.frame_i
        self.frame_i += 1


def main():
    app = Application()
    app.run()


if __name__ == '__main__':
    main()