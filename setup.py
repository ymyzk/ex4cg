#!/usr/bin/env python

from setuptools import setup

__author__ = 'Yusuke Miyazaki <miyazaki.dev@gmail.com>'
__version__ = '0.1'

requires = [
    'numpy>=1.9.0'
]

setup(
    name='ex4cg',
    version=__version__,
    author=__author__,
    author_email='miyazaki.dev@gmail.com',
    description='',
    packages=['cg'],
    install_requires=requires,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4'
    ]
)