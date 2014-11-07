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


class Application(object):
    def __init__(self):
        self.vrml = Vrml()
        self.data = None
        self.width = self.height = 256
        self.key_frames = {}

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
        camera_panel_layout.addWidget(self.camera_position_x, 0, 1)
        self.camera_position_y = QtGui.QDoubleSpinBox()
        self.camera_position_y.setMinimum(-32768.0)
        self.camera_position_y.setMaximum(32767.0)
        self.camera_position_y.setValue(0.0)
        camera_panel_layout.addWidget(self.camera_position_y, 0, 2)
        self.camera_position_z = QtGui.QDoubleSpinBox()
        self.camera_position_z.setMinimum(-32768.0)
        self.camera_position_z.setMaximum(32767.0)
        self.camera_position_z.setValue(0.0)
        camera_panel_layout.addWidget(self.camera_position_z, 0, 3)

        camera_panel_layout.addWidget(
            QtGui.QLabel('Angle (x, y, z): '), 1, 0)
        self.camera_angle_x = QtGui.QDoubleSpinBox()
        self.camera_angle_x.setMinimum(-360.0)
        self.camera_angle_x.setMaximum(360.0)
        self.camera_angle_x.setValue(0.0)
        camera_panel_layout.addWidget(self.camera_angle_x, 1, 1)
        self.camera_angle_y = QtGui.QDoubleSpinBox()
        self.camera_angle_y.setMinimum(-360.0)
        self.camera_angle_y.setMaximum(360.0)
        self.camera_angle_y.setValue(0.0)
        camera_panel_layout.addWidget(self.camera_angle_y, 1, 2)
        self.camera_angle_z = QtGui.QDoubleSpinBox()
        self.camera_angle_z.setMinimum(-360.0)
        self.camera_angle_z.setMaximum(360.0)
        self.camera_angle_z.setValue(0.0)
        camera_panel_layout.addWidget(self.camera_angle_z, 1, 3)

        camera_panel_layout.addWidget(
            QtGui.QLabel('Focus (f): '), 2, 0)
        self.camera_focus = QtGui.QDoubleSpinBox()
        self.camera_focus.setMaximum(1024.0)
        self.camera_focus.setValue(256.0)
        camera_panel_layout.addWidget(self.camera_focus, 2, 1)

        # Diffuse Tab
        diffuse_panel = QtGui.QWidget()
        diffuse_panel_layout = QtGui.QGridLayout()
        diffuse_panel.setLayout(diffuse_panel_layout)
        control_tab.addTab(diffuse_panel, 'Diffuse Shader')

        self.diffuse_checkbox = QtGui.QCheckBox('Enable')
        self.diffuse_checkbox.setCheckState(2)
        diffuse_panel_layout.addWidget(self.diffuse_checkbox, 0, 0)

        diffuse_panel_layout.addWidget(
            QtGui.QLabel('Direction (x, y, z): '), 1, 0)
        self.diffuse_direction_x = QtGui.QDoubleSpinBox()
        self.diffuse_direction_x.setMinimum(-128.0)
        self.diffuse_direction_x.setMaximum(127.0)
        self.diffuse_direction_x.setValue(-1.0)
        diffuse_panel_layout.addWidget(self.diffuse_direction_x, 1, 1)
        self.diffuse_direction_y = QtGui.QDoubleSpinBox()
        self.diffuse_direction_y.setMinimum(-128.0)
        self.diffuse_direction_y.setMaximum(127.0)
        self.diffuse_direction_y.setValue(-1.0)
        diffuse_panel_layout.addWidget(self.diffuse_direction_y, 1, 2)
        self.diffuse_direction_z = QtGui.QDoubleSpinBox()
        self.diffuse_direction_z.setMinimum(-128.0)
        self.diffuse_direction_z.setMaximum(127.0)
        self.diffuse_direction_z.setValue(2.0)
        diffuse_panel_layout.addWidget(self.diffuse_direction_z, 1, 3)

        diffuse_panel_layout.addWidget(
            QtGui.QLabel('Luminance (r, g, b): '), 2, 0)
        self.diffuse_luminance_r = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_r.setMinimum(0.0)
        self.diffuse_luminance_r.setMaximum(1.0)
        self.diffuse_luminance_r.setSingleStep(0.1)
        self.diffuse_luminance_r.setValue(1.0)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_r, 2, 1)
        self.diffuse_luminance_g = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_g.setMinimum(0.0)
        self.diffuse_luminance_g.setMaximum(1.0)
        self.diffuse_luminance_g.setSingleStep(0.1)
        self.diffuse_luminance_g.setValue(1.0)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_g, 2, 2)
        self.diffuse_luminance_b = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_b.setMinimum(0.0)
        self.diffuse_luminance_b.setMaximum(1.0)
        self.diffuse_luminance_b.setSingleStep(0.1)
        self.diffuse_luminance_b.setValue(1.0)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_b, 2, 3)

        # Specular Tab
        specular_panel = QtGui.QWidget()
        specular_panel_layout = QtGui.QGridLayout()
        specular_panel.setLayout(specular_panel_layout)
        control_tab.addTab(specular_panel, 'Sqecular Shader')

        self.specular_checkbox = QtGui.QCheckBox('Enable')
        self.specular_checkbox.setCheckState(2)
        specular_panel_layout.addWidget(self.specular_checkbox, 0, 0)

        specular_panel_layout.addWidget(
            QtGui.QLabel('Direction (x, y, z): '), 1, 0)
        self.specular_direction_x = QtGui.QDoubleSpinBox()
        self.specular_direction_x.setMinimum(-128.0)
        self.specular_direction_x.setMaximum(127.0)
        self.specular_direction_x.setValue(-1.0)
        specular_panel_layout.addWidget(self.specular_direction_x, 1, 1)
        self.specular_direction_y = QtGui.QDoubleSpinBox()
        self.specular_direction_y.setMinimum(-128.0)
        self.specular_direction_y.setMaximum(127.0)
        self.specular_direction_y.setValue(-1.0)
        specular_panel_layout.addWidget(self.specular_direction_y, 1, 2)
        self.specular_direction_z = QtGui.QDoubleSpinBox()
        self.specular_direction_z.setMinimum(-128.0)
        self.specular_direction_z.setMaximum(127.0)
        self.specular_direction_z.setValue(2.0)
        specular_panel_layout.addWidget(self.specular_direction_z, 1, 3)

        specular_panel_layout.addWidget(QtGui.QLabel(
            'Luminance (r, g, b): '), 2, 0)
        self.specular_luminance_r = QtGui.QDoubleSpinBox()
        self.specular_luminance_r.setMinimum(0.0)
        self.specular_luminance_r.setMaximum(1.0)
        self.specular_luminance_r.setSingleStep(0.1)
        self.specular_luminance_r.setValue(1.0)
        specular_panel_layout.addWidget(self.specular_luminance_r, 2, 1)
        self.specular_luminance_g = QtGui.QDoubleSpinBox()
        self.specular_luminance_g.setMinimum(0.0)
        self.specular_luminance_g.setMaximum(1.0)
        self.specular_luminance_g.setSingleStep(0.1)
        self.specular_luminance_g.setValue(1.0)
        specular_panel_layout.addWidget(self.specular_luminance_g, 2, 2)
        self.specular_luminance_b = QtGui.QDoubleSpinBox()
        self.specular_luminance_b.setMinimum(0.0)
        self.specular_luminance_b.setMaximum(1.0)
        self.specular_luminance_b.setSingleStep(0.1)
        self.specular_luminance_b.setValue(1.0)
        specular_panel_layout.addWidget(self.specular_luminance_b, 2, 3)

        # Ambient Tab
        ambient_panel = QtGui.QWidget()
        ambient_panel_layout = QtGui.QGridLayout()
        ambient_panel.setLayout(ambient_panel_layout)
        control_tab.addTab(ambient_panel, 'Ambient Shader')

        self.ambient_checkbox = QtGui.QCheckBox('Enable')
        self.ambient_checkbox.setCheckState(2)
        ambient_panel_layout.addWidget(self.ambient_checkbox, 0, 0)

        ambient_panel_layout.addWidget(
            QtGui.QLabel('Luminance (r, g, b): '), 1, 0)
        self.ambient_luminance_r = QtGui.QDoubleSpinBox()
        self.ambient_luminance_r.setMinimum(0.0)
        self.ambient_luminance_r.setMaximum(1.0)
        self.ambient_luminance_r.setSingleStep(0.1)
        self.ambient_luminance_r.setValue(1.0)
        ambient_panel_layout.addWidget(self.ambient_luminance_r, 1, 1)
        self.ambient_luminance_g = QtGui.QDoubleSpinBox()
        self.ambient_luminance_g.setMinimum(0.0)
        self.ambient_luminance_g.setMaximum(1.0)
        self.ambient_luminance_g.setSingleStep(0.1)
        self.ambient_luminance_g.setValue(1.0)
        ambient_panel_layout.addWidget(self.ambient_luminance_g, 1, 2)
        self.ambient_luminance_b = QtGui.QDoubleSpinBox()
        self.ambient_luminance_b.setMinimum(0.0)
        self.ambient_luminance_b.setMaximum(1.0)
        self.ambient_luminance_b.setSingleStep(0.1)
        self.ambient_luminance_b.setValue(1.0)
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
        shading_mode_layout.addWidget(self.shading_mode_flat)
        self.shading_mode_gouraud = QtGui.QRadioButton('Gouraud')
        shading_mode_layout.addWidget(self.shading_mode_gouraud)
        self.shading_mode_phong = QtGui.QRadioButton('Phong')
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
        render_panel_layout.addWidget(self.render_size_w, 2, 1)
        self.render_size_h = QtGui.QSpinBox()
        self.render_size_h.setMinimum(0)
        self.render_size_h.setMaximum(512)
        self.render_size_h.setValue(self.height)
        render_panel_layout.addWidget(self.render_size_h, 2, 2)

        render_button = QtGui.QPushButton('Render')
        render_button.clicked.connect(self.render)
        render_panel_layout.addWidget(render_button, 3, 0, 1, 3)

        # Animate Tab
        animate_panel = QtGui.QWidget()
        animate_panel_layout = QtGui.QGridLayout()
        animate_panel.setLayout(animate_panel_layout)
        control_tab.addTab(animate_panel, 'Animate')

        animate_panel_layout.addWidget(QtGui.QLabel('# Frame: '), 0, 0)
        self.animate_key_frame = QtGui.QSpinBox()
        self.animate_key_frame.setMinimum(0)
        self.animate_key_frame.setMaximum(1024)
        self.animate_key_frame.setValue(0)
        animate_panel_layout.addWidget(self.animate_key_frame, 0, 1)

        key_frame_button = QtGui.QPushButton('Key Frame')
        key_frame_button.clicked.connect(self.key_frame)
        animate_panel_layout.addWidget(key_frame_button, 0, 2)

        animate_panel_layout.addWidget(QtGui.QLabel('FPS: '), 1, 0)
        self.animate_key_frame = QtGui.QSpinBox()
        self.animate_key_frame.setMinimum(0)
        self.animate_key_frame.setMaximum(300)
        self.animate_key_frame.setValue(30)
        animate_panel_layout.addWidget(self.animate_key_frame, 1, 1, 1, 2)

        animate_button = QtGui.QPushButton('Animate')
        animate_button.clicked.connect(self.animate)
        animate_panel_layout.addWidget(animate_button, 2, 0, 1, 3)

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

    def key_frame(self):
        """キーフレームとして情報を保存する処理"""
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
            'ambient_luminance_b': self.ambient_luminance_b.value(),
            'random': (self.diffuse_checkbox.checkState() != 2
                       and self.specular_checkbox.checkState() == 2
                       and self.ambient_checkbox.checkState() == 2)
        }

        key_frame = self.animate_key_frame.value()
        self.key_frames[key_frame] = info

    def render(self):
        self.status_bar.showMessage('Rendering..')

        camera = Camera(
            position=np.array((
                self.camera_position_x.value(),
                self.camera_position_y.value(),
                self.camera_position_z.value())),
            angle=np.array((
                self.camera_angle_x.value(),
                self.camera_angle_y.value(),
                self.camera_angle_z.value())),
            focus=self.camera_focus.value())

        if self.backend_cython.isChecked():
            Renderer = CyRenderer
        else:
            Renderer = PyRenderer

        shaders = []
        if (self.diffuse_checkbox.checkState() == 2 and
                self.vrml.diffuse_color is not None):
            shaders.append(shader.DiffuseShader(
                direction=np.array((
                    self.diffuse_direction_x.value(),
                    self.diffuse_direction_y.value(),
                    self.diffuse_direction_z.value())),
                luminance=np.array((
                    self.diffuse_luminance_r.value(),
                    self.diffuse_luminance_g.value(),
                    self.diffuse_luminance_b.value())),
                color=self.vrml.diffuse_color))
        if (self.specular_checkbox.checkState() == 2 and
                self.vrml.specular_color is not None and
                self.vrml.shininess is not None):
            shaders.append(shader.SpecularShader(
                camera_position=camera.position,
                direction=np.array((
                    self.specular_direction_x.value(),
                    self.specular_direction_y.value(),
                    self.specular_direction_z.value())),
                luminance=np.array((
                    self.specular_luminance_r.value(),
                    self.specular_luminance_g.value(),
                    self.specular_luminance_b.value())),
                color=self.vrml.specular_color,
                shininess=self.vrml.shininess))
        if (self.ambient_checkbox.checkState() == 2 and
                self.vrml.ambient_intensity is not None):
            shaders.append(shader.AmbientShader(
                luminance=np.array((
                    self.ambient_luminance_r.value(),
                    self.ambient_luminance_g.value(),
                    self.ambient_luminance_b.value())),
                intensity=self.vrml.ambient_intensity))
        if len(shaders) == 0:
            shaders.append(shader.RandomColorShader())

        self.width = self.render_size_w.value()
        self.height = self.render_size_h.value()

        mode = ShadingMode.flat
        if self.shading_mode_flat.isChecked():
            mode = ShadingMode.flat
        elif self.shading_mode_gouraud.isChecked():
            mode = ShadingMode.gouraud
        elif self.shading_mode_phong.isChecked():
            mode = ShadingMode.phong

        renderer = Renderer(width=self.width, height=self.height,
                            shading_mode=mode)
        renderer.camera = camera
        renderer.shaders = shaders
        renderer.prepare_polygons(self.vrml.points, self.vrml.indexes)
        renderer.draw_polygons()

        self.data = renderer.data
        self.image_label.set_image(self.data, self.width, self.height)

        self.status_bar.showMessage('Rendered.')

    def animate(self):
        pass


def main():
    app = Application()
    app.run()


if __name__ == '__main__':
    main()