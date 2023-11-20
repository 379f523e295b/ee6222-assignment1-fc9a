#!/usr/bin/env python3

import setuptools


setuptools.setup(
    name='tensorflow-datasets-ee6222har',
    version='0.0.0a',
    description='EE6222 HAR Tensorflow Dataset Addon',
    python_requires='>=3.6',
    packages=setuptools.find_namespace_packages('.'),
    package_data={
        'tensorflow_datasets_ee6222har'
        '.ee6222har.data': ['*.zip']
    },
    # TODO
    #packages=setuptools.find_packages('.'),
    # TODO
    install_requires=[
        'tensorflow-datasets',
        'pandas',
        'bidict'
    ],
    extras_require={
        'dev': []
    },
)
