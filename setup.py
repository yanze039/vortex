from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = f.read().split("\n")

setup(
    name="vortex",
    version="0.0.1",
    description="Inference and utilities for inference and training of deep signal processing models",
    long_description=readme,
    long_description_content_type="text/markdown",
    author="Michael Poli",
    url="http://github.com/zymrael/vortex",
    license="Apache-2.0",
    packages=find_packages(where="vortex"),
    install_requires=requirements,
)
