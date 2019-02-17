#!/usr/bin/env python
import os
from setuptools import setup

version = os.environ.get('MILA_VERSION', '0.0.0')

setup(
    name='gradmask',
    description='Simple research project example.',
    version=version,
    author='MILA',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'gradmask = gradmask.main:main'
        ]
    }

)
