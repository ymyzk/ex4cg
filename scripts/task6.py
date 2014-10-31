import sys

import numpy as np
from PyQt4 import QtCore, QtGui

from cg import shader
from cg.camera import Camera
from cg.cython.renderer import Renderer as CyRenderer
from cg.renderer import Renderer as PyRenderer
from cg.shader import ShadingMode
from cg.vrml import Vrml


width = height = 256
depth = 8


class ImageWidget(QtGui.QLabel):
    def __init__(self):
        super().__init__()

    def set_image(self, image):
        self.setPixmap(QtGui.QPixmap(
            QtGui.QImage(image, width, height, QtGui.QImage.Format_RGB888)))


class Application(object):
    def __init__(self):
        self.vrml = Vrml()

        # UI
        self.app = QtGui.QApplication(sys.argv)
        self.main_window = QtGui.QMainWindow()

        # Main
        main_panel = QtGui.QWidget()
        main_panel_layout = QtGui.QVBoxLayout()
        main_panel.setLayout(main_panel_layout)

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
        self.image_label.setFixedSize(width, height)
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

        # Camera Tab
        camera_panel = QtGui.QWidget()
        camera_panel_layout = QtGui.QGridLayout()
        camera_panel.setLayout(camera_panel_layout)
        control_tab.addTab(camera_panel, 'Camera')

        # camera_panel_layout.addWidget(
        #     QtGui.QLabel('Position (x, y, z): '), 0, 0)
        # camera_position_x = QtGui.QDoubleSpinBox()
        # camera_position_x.setValue(0.0)
        # camera_panel_layout.addWidget(camera_position_x, 0, 1)
        # camera_position_y = QtGui.QDoubleSpinBox()
        # camera_position_y.setValue(0.0)
        # camera_panel_layout.addWidget(camera_position_y, 0, 2)
        # camera_position_z = QtGui.QDoubleSpinBox()
        # camera_position_z.setValue(0.0)
        # camera_panel_layout.addWidget(camera_position_z, 0, 3)

        # camera_panel_layout.addWidget(
        #     QtGui.QLabel('Angle (x, y, z): '), 1, 0)
        # camera_angle_x = QtGui.QDoubleSpinBox()
        # camera_angle_x.setValue(0.0)
        # camera_panel_layout.addWidget(camera_angle_x, 1, 1)
        # camera_angle_y = QtGui.QDoubleSpinBox()
        # camera_angle_y.setValue(0.0)
        # camera_panel_layout.addWidget(camera_angle_y, 1, 2)
        # camera_angle_z = QtGui.QDoubleSpinBox()
        # camera_angle_z.setValue(1.0)
        # camera_panel_layout.addWidget(camera_angle_z, 1, 3)

        camera_panel_layout.addWidget(
            QtGui.QLabel('Focus (f): '), 0, 0)
        self.camera_focus = QtGui.QDoubleSpinBox()
        self.camera_focus.setMaximum(1024.0)
        self.camera_focus.setValue(256.0)
        camera_panel_layout.addWidget(self.camera_focus, 0, 1)

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
        self.diffuse_luminance_x = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_x.setMinimum(0.0)
        self.diffuse_luminance_x.setMaximum(1.0)
        self.diffuse_luminance_x.setSingleStep(0.1)
        self.diffuse_luminance_x.setValue(1.0)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_x, 2, 1)
        self.diffuse_luminance_y = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_y.setMinimum(0.0)
        self.diffuse_luminance_y.setMaximum(1.0)
        self.diffuse_luminance_y.setSingleStep(0.1)
        self.diffuse_luminance_y.setValue(1.0)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_y, 2, 2)
        self.diffuse_luminance_z = QtGui.QDoubleSpinBox()
        self.diffuse_luminance_z.setMinimum(0.0)
        self.diffuse_luminance_z.setMaximum(1.0)
        self.diffuse_luminance_z.setSingleStep(0.1)
        self.diffuse_luminance_z.setValue(1.0)
        diffuse_panel_layout.addWidget(self.diffuse_luminance_z, 2, 3)

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
        self.specular_luminance_x = QtGui.QDoubleSpinBox()
        self.specular_luminance_x.setMinimum(0.0)
        self.specular_luminance_x.setMaximum(1.0)
        self.specular_luminance_x.setSingleStep(0.1)
        self.specular_luminance_x.setValue(1.0)
        specular_panel_layout.addWidget(self.specular_luminance_x, 2, 1)
        self.specular_luminance_y = QtGui.QDoubleSpinBox()
        self.specular_luminance_y.setMinimum(0.0)
        self.specular_luminance_y.setMaximum(1.0)
        self.specular_luminance_y.setSingleStep(0.1)
        self.specular_luminance_y.setValue(1.0)
        specular_panel_layout.addWidget(self.specular_luminance_y, 2, 2)
        self.specular_luminance_z = QtGui.QDoubleSpinBox()
        self.specular_luminance_z.setMinimum(0.0)
        self.specular_luminance_z.setMaximum(1.0)
        self.specular_luminance_z.setSingleStep(0.1)
        self.specular_luminance_z.setValue(1.0)
        specular_panel_layout.addWidget(self.specular_luminance_z, 2, 3)

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
        self.ambient_luminance_x = QtGui.QDoubleSpinBox()
        self.ambient_luminance_x.setMinimum(0.0)
        self.ambient_luminance_x.setMaximum(1.0)
        self.ambient_luminance_x.setSingleStep(0.1)
        self.ambient_luminance_x.setValue(1.0)
        ambient_panel_layout.addWidget(self.ambient_luminance_x, 1, 1)
        self.ambient_luminance_y = QtGui.QDoubleSpinBox()
        self.ambient_luminance_y.setMinimum(0.0)
        self.ambient_luminance_y.setMaximum(1.0)
        self.ambient_luminance_y.setSingleStep(0.1)
        self.ambient_luminance_y.setValue(1.0)
        ambient_panel_layout.addWidget(self.ambient_luminance_y, 1, 2)
        self.ambient_luminance_z = QtGui.QDoubleSpinBox()
        self.ambient_luminance_z.setMinimum(0.0)
        self.ambient_luminance_z.setMaximum(1.0)
        self.ambient_luminance_z.setSingleStep(0.1)
        self.ambient_luminance_z.setValue(1.0)
        ambient_panel_layout.addWidget(self.ambient_luminance_z, 1, 3)

        # Render Tab
        render_panel = QtGui.QWidget()
        render_panel_layout = QtGui.QGridLayout()
        render_panel.setLayout(render_panel_layout)
        control_tab.addTab(render_panel, 'Render')

        shading_mode = QtGui.QGroupBox()
        shading_mode_layout = QtGui.QHBoxLayout()
        shading_mode.setLayout(shading_mode_layout)
        shading_mode.setTitle('Shading Mode')
        render_panel_layout.addWidget(shading_mode, 0, 0)

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
        render_panel_layout.addWidget(backend, 1, 0)

        self.backend_python = QtGui.QRadioButton('Python')
        self.backend_python.setChecked(1)
        backend_layout.addWidget(self.backend_python)
        self.backend_cython = QtGui.QRadioButton('Python + Cython')
        backend_layout.addWidget(self.backend_cython)

        render_button = QtGui.QPushButton('Render')
        render_button.clicked.connect(self.render)
        render_panel_layout.addWidget(render_button, 2, 0)

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

    def render(self):
        self.status_bar.showMessage('Rendering..')

        camera = Camera(position=np.array((0.0, 0.0, 0.0)),
                        angle=np.array((0.0, 0.0, 1.0)),
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
                    self.diffuse_luminance_x.value(),
                    self.diffuse_luminance_y.value(),
                    self.diffuse_luminance_z.value())),
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
                    self.specular_luminance_x.value(),
                    self.specular_luminance_y.value(),
                    self.specular_luminance_z.value())),
                color=self.vrml.specular_color,
                shininess=self.vrml.shininess))
        if (self.ambient_checkbox.checkState() == 2 and
                self.vrml.ambient_intensity is not None):
            shaders.append(shader.AmbientShader(
                luminance=np.array((
                    self.ambient_luminance_x.value(),
                    self.ambient_luminance_y.value(),
                    self.ambient_luminance_z.value())),
                intensity=self.vrml.ambient_intensity))
        if len(shaders) == 0:
            shaders.append(shader.RandomColorShader())

        mode = ShadingMode.flat
        if self.shading_mode_flat.isChecked():
            mode = ShadingMode.flat
        elif self.shading_mode_gouraud.isChecked():
            mode = ShadingMode.gouraud
        elif self.shading_mode_phong.isChecked():
            mode = ShadingMode.phong

        renderer = Renderer(width=width, height=height,
                            shading_mode=mode)
        renderer.camera = camera
        renderer.shaders = shaders
        renderer.draw_polygons(self.vrml.points, self.vrml.indexes)
        self.image_label.set_image(renderer.data.data)

        self.status_bar.showMessage('Rendered.')


def main():
    app = Application()
    app.run()


if __name__ == '__main__':
    main()