#!/usr/bin/env python
import os
from setuptools import setup

version = os.environ.get('MILA_VERSION', '0.1.0')

setup(
    name='activmask',
    description='A way to get your CNN to ignore predictive features.',
    version=version,
    author='Joseph D Viviano',
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'activmask = activmask.main:main'
        ]
    }

)
