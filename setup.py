from setuptools import setup, find_packages

setup(
    name="cil",
    version="0.1.0",
    url="https://github.com/khanhvu207/ethz-cil-project",
    description="ETHZ CIL project",
    install_requires=[],
    packages=find_packages(where='src'),
    package_dir={"": "src"},
)
