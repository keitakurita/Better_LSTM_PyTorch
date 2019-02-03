#!/usr/bin/env python
import os
from setuptools import setup, find_packages

cd = os.path.dirname(__file__)

# read version info
long_description = open(os.path.join(cd, 'README.rst'), "rt", encoding="utf-8").read()

setup(
    name="better_lstm",
    version=0.1,
    author="Keita Kurita",
    author_email="keita.kurita@gmail.com",
    description="LSTM with best practices incorporated",
    long_description=long_description,
    license="MIT",
    url="https://github.com/keitakurita/torchtable",
    python_requires = ">=3.6",
    keywords = "PyTorch, deep learning, machine learning",
    setup_requires=["pytest", ],
    install_requires=[
        "torch>=1.0.0",
    ],
    packages=find_packages(),
)
