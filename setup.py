#!/usr/bin/env python

from setuptools import setup, Extension

try:
    from Cython.Distutils import build_ext
    USE_CYTHON = True
except ImportError:
    USE_CYTHON = False

__author__ = 'Yusuke Miyazaki <miyazaki.dev@gmail.com>'
__version__ = '0.1'

requires = [
    'numpy>=1.9.0'
]

if USE_CYTHON:
    ext_modules = [Extension('cg.cython', sources=['cython/shader.pyx'])]
    cmdclass = {'build_ext': build_ext}
else:
    ext_modules = [Extension('cg.cython', sources=['cython/shader.c'])]
    cmdclass = {}

setup(
    name='ex4cg',
    version=__version__,
    author=__author__,
    author_email='miyazaki.dev@gmail.com',
    description='',
    packages=['cg'],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4'
    ]
)