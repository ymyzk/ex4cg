import sys

import numpy as np
from PyQt4 import QtCore, QtGui

from cg import shader as py_shader
from cg.camera import Camera
from cg.cython import shader as cy_shader
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


def main():
    vrml = Vrml()

    def open_vrml():
        file_path = QtGui.QFileDialog.getOpenFileName()
        with open(file_path) as f:
            vrml.load(f)
        file_label.setText(file_path)
        render()

    def render():
        camera = Camera(position=np.array((0.0, 0.0, 0.0)),
                        angle=np.array((0.0, 0.0, 1.0)),
                        focus=camera_focus.value())

        if backend_cython.isChecked():
            Renderer = CyRenderer
            shader = cy_shader
        else:
            Renderer = PyRenderer
            shader = py_shader

        shaders = []
        if (diffuse_checkbox.checkState() == 2 and
                vrml.diffuse_color is not None):
            shaders.append(shader.DiffuseShader(
                direction=np.array((
                    diffuse_direction_x.value(),
                    diffuse_direction_y.value(),
                    diffuse_direction_z.value())),
                luminance=np.array((
                    diffuse_luminance_x.value(),
                    diffuse_luminance_y.value(),
                    diffuse_luminance_z.value())),
                color=vrml.diffuse_color))
        if (specular_checkbox.checkState() == 2 and
                vrml.specular_color is not None and
                vrml.shininess is not None):
            shaders.append(shader.SpecularShader(
                camera_position=camera.position,
                direction=np.array((
                    specular_direction_x.value(),
                    specular_direction_y.value(),
                    specular_direction_z.value())),
                luminance=np.array((
                    specular_luminance_x.value(),
                    specular_luminance_y.value(),
                    specular_luminance_z.value())),
                color=vrml.specular_color,
                shininess=vrml.shininess))
        if (ambient_checkbox.checkState() == 2 and
                vrml.ambient_intensity is not None):
            shaders.append(shader.AmbientShader(
                luminance=np.array((
                    ambient_luminance_x.value(),
                    ambient_luminance_y.value(),
                    ambient_luminance_z.value())),
                intensity=vrml.ambient_intensity))
        if len(shaders) == 0:
            shaders.append(shader.RandomColorShader())

        mode = ShadingMode.flat
        if shading_mode_flat.isChecked():
            mode = ShadingMode.flat
        elif shading_mode_gouraud.isChecked():
            mode = ShadingMode.gouraud
        elif shading_mode_phong.isChecked():
            mode = ShadingMode.phong

        renderer = Renderer(camera=camera, shaders=shaders,
                            width=width, height=height,
                            shading_mode=mode)
        renderer.draw_polygons(vrml.points, vrml.indexes)
        image_label.set_image(renderer.data.data)


    # VRML ファイルの読み込み

    app = QtGui.QApplication(sys.argv)
    main_window = QtGui.QMainWindow()

    # Main
    main_panel = QtGui.QWidget()
    main_panel_layout = QtGui.QVBoxLayout()
    main_panel.setLayout(main_panel_layout)

    # Image
    image_panel = QtGui.QWidget()
    image_panel_layout = QtGui.QHBoxLayout()
    image_panel.setLayout(image_panel_layout)
    main_panel_layout.addWidget(image_panel)

    image_label = ImageWidget()
    image_label.setFixedSize(width, height)
    image_panel_layout.addStretch()
    image_panel_layout.addWidget(image_label)
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
    open_file_button.clicked.connect(open_vrml)
    file_panel_layout.addWidget(open_file_button, 0, 0)

    file_label = QtGui.QLabel()
    file_label.setText("Please select VRML file")
    file_panel_layout.addWidget(file_label, 1, 0)

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
    camera_focus = QtGui.QDoubleSpinBox()
    camera_focus.setMaximum(1024.0)
    camera_focus.setValue(256.0)
    camera_panel_layout.addWidget(camera_focus, 0, 1)

    # Diffuse Tab
    diffuse_panel = QtGui.QWidget()
    diffuse_panel_layout = QtGui.QGridLayout()
    diffuse_panel.setLayout(diffuse_panel_layout)
    control_tab.addTab(diffuse_panel, 'Diffuse Shader')

    diffuse_checkbox = QtGui.QCheckBox('Enable')
    diffuse_checkbox.setCheckState(2)
    diffuse_panel_layout.addWidget(diffuse_checkbox, 0, 0)

    diffuse_panel_layout.addWidget(QtGui.QLabel('Direction (x, y, z): '), 1, 0)
    diffuse_direction_x = QtGui.QDoubleSpinBox()
    diffuse_direction_x.setMinimum(-128.0)
    diffuse_direction_x.setMaximum(127.0)
    diffuse_direction_x.setValue(-1.0)
    diffuse_panel_layout.addWidget(diffuse_direction_x, 1, 1)
    diffuse_direction_y = QtGui.QDoubleSpinBox()
    diffuse_direction_y.setMinimum(-128.0)
    diffuse_direction_y.setMaximum(127.0)
    diffuse_direction_y.setValue(-1.0)
    diffuse_panel_layout.addWidget(diffuse_direction_y, 1, 2)
    diffuse_direction_z = QtGui.QDoubleSpinBox()
    diffuse_direction_z.setMinimum(-128.0)
    diffuse_direction_z.setMaximum(127.0)
    diffuse_direction_z.setValue(2.0)
    diffuse_panel_layout.addWidget(diffuse_direction_z, 1, 3)

    diffuse_panel_layout.addWidget(QtGui.QLabel('Luminance (r, g, b): '), 2, 0)
    diffuse_luminance_x = QtGui.QDoubleSpinBox()
    diffuse_luminance_x.setMinimum(0.0)
    diffuse_luminance_x.setMaximum(1.0)
    diffuse_luminance_x.setSingleStep(0.1)
    diffuse_luminance_x.setValue(1.0)
    diffuse_panel_layout.addWidget(diffuse_luminance_x, 2, 1)
    diffuse_luminance_y = QtGui.QDoubleSpinBox()
    diffuse_luminance_y.setMinimum(0.0)
    diffuse_luminance_y.setMaximum(1.0)
    diffuse_luminance_y.setSingleStep(0.1)
    diffuse_luminance_y.setValue(1.0)
    diffuse_panel_layout.addWidget(diffuse_luminance_y, 2, 2)
    diffuse_luminance_z = QtGui.QDoubleSpinBox()
    diffuse_luminance_z.setMinimum(0.0)
    diffuse_luminance_z.setMaximum(1.0)
    diffuse_luminance_z.setSingleStep(0.1)
    diffuse_luminance_z.setValue(1.0)
    diffuse_panel_layout.addWidget(diffuse_luminance_z, 2, 3)

    # Specular Tab
    specular_panel = QtGui.QWidget()
    specular_panel_layout = QtGui.QGridLayout()
    specular_panel.setLayout(specular_panel_layout)
    control_tab.addTab(specular_panel, 'Sqecular Shader')

    specular_checkbox = QtGui.QCheckBox('Enable')
    specular_checkbox.setCheckState(2)
    specular_panel_layout.addWidget(specular_checkbox, 0, 0)

    specular_panel_layout.addWidget(
        QtGui.QLabel('Direction (x, y, z): '), 1, 0)
    specular_direction_x = QtGui.QDoubleSpinBox()
    specular_direction_x.setMinimum(-128.0)
    specular_direction_x.setMaximum(127.0)
    specular_direction_x.setValue(-1.0)
    specular_panel_layout.addWidget(specular_direction_x, 1, 1)
    specular_direction_y = QtGui.QDoubleSpinBox()
    specular_direction_y.setMinimum(-128.0)
    specular_direction_y.setMaximum(127.0)
    specular_direction_y.setValue(-1.0)
    specular_panel_layout.addWidget(specular_direction_y, 1, 2)
    specular_direction_z = QtGui.QDoubleSpinBox()
    specular_direction_z.setMinimum(-128.0)
    specular_direction_z.setMaximum(127.0)
    specular_direction_z.setValue(2.0)
    specular_panel_layout.addWidget(specular_direction_z, 1, 3)

    specular_panel_layout.addWidget(QtGui.QLabel(
        'Luminance (r, g, b): '), 2, 0)
    specular_luminance_x = QtGui.QDoubleSpinBox()
    specular_luminance_x.setMinimum(0.0)
    specular_luminance_x.setMaximum(1.0)
    specular_luminance_x.setSingleStep(0.1)
    specular_luminance_x.setValue(1.0)
    specular_panel_layout.addWidget(specular_luminance_x, 2, 1)
    specular_luminance_y = QtGui.QDoubleSpinBox()
    specular_luminance_y.setMinimum(0.0)
    specular_luminance_y.setMaximum(1.0)
    specular_luminance_y.setSingleStep(0.1)
    specular_luminance_y.setValue(1.0)
    specular_panel_layout.addWidget(specular_luminance_y, 2, 2)
    specular_luminance_z = QtGui.QDoubleSpinBox()
    specular_luminance_z.setMinimum(0.0)
    specular_luminance_z.setMaximum(1.0)
    specular_luminance_z.setSingleStep(0.1)
    specular_luminance_z.setValue(1.0)
    specular_panel_layout.addWidget(specular_luminance_z, 2, 3)

    # Ambient Tab
    ambient_panel = QtGui.QWidget()
    ambient_panel_layout = QtGui.QGridLayout()
    ambient_panel.setLayout(ambient_panel_layout)
    control_tab.addTab(ambient_panel, 'Ambient Shader')

    ambient_checkbox = QtGui.QCheckBox('Enable')
    ambient_checkbox.setCheckState(2)
    ambient_panel_layout.addWidget(ambient_checkbox, 0, 0)

    ambient_panel_layout.addWidget(QtGui.QLabel('Luminance (r, g, b): '), 1, 0)
    ambient_luminance_x = QtGui.QDoubleSpinBox()
    ambient_luminance_x.setMinimum(0.0)
    ambient_luminance_x.setMaximum(1.0)
    ambient_luminance_x.setSingleStep(0.1)
    ambient_luminance_x.setValue(1.0)
    ambient_panel_layout.addWidget(ambient_luminance_x, 1, 1)
    ambient_luminance_y = QtGui.QDoubleSpinBox()
    ambient_luminance_y.setMinimum(0.0)
    ambient_luminance_y.setMaximum(1.0)
    ambient_luminance_y.setSingleStep(0.1)
    ambient_luminance_y.setValue(1.0)
    ambient_panel_layout.addWidget(ambient_luminance_y, 1, 2)
    ambient_luminance_z = QtGui.QDoubleSpinBox()
    ambient_luminance_z.setMinimum(0.0)
    ambient_luminance_z.setMaximum(1.0)
    ambient_luminance_z.setSingleStep(0.1)
    ambient_luminance_z.setValue(1.0)
    ambient_panel_layout.addWidget(ambient_luminance_z, 1, 3)

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

    shading_mode_flat = QtGui.QRadioButton('Flat (Constant)')
    shading_mode_flat.setChecked(1)
    shading_mode_layout.addWidget(shading_mode_flat)
    shading_mode_gouraud = QtGui.QRadioButton('Gouraud')
    shading_mode_layout.addWidget(shading_mode_gouraud)
    shading_mode_phong = QtGui.QRadioButton('Phong')
    shading_mode_layout.addWidget(shading_mode_phong)

    backend = QtGui.QGroupBox()
    backend_layout = QtGui.QHBoxLayout()
    backend.setLayout(backend_layout)
    backend.setTitle('Backend')
    render_panel_layout.addWidget(backend, 1, 0)

    backend_python = QtGui.QRadioButton('Python')
    backend_python.setChecked(1)
    backend_layout.addWidget(backend_python)
    backend_cython = QtGui.QRadioButton('Python + Cython')
    backend_layout.addWidget(backend_cython)

    render_button = QtGui.QPushButton('Render')
    render_button.clicked.connect(render)
    render_panel_layout.addWidget(render_button, 2, 0)

    main_window.setCentralWidget(main_panel)
    main_window.show()

    app.exec_()


if __name__ == '__main__':
    main()