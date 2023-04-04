from setuptools import setup, find_packages
import sys

version = sys.version_info
required_py_version = 3.6
if version[0] < int(required_py_version) or (
    version[0] == int(required_py_version)
    and version[1] < required_py_version - int(required_py_version)
):
    raise SystemError("Minimum supported python version is %.2f" % required_py_version)


# adapted from pip's definition, https://github.com/pypa/pip/blob/master/setup.py
def get_version(rel_path):
    with open(rel_path) as file:
        for line in file:
            if line.startswith("__version__"):
                delim = '"' if '"' in line else "'"
                version = line.split(delim)[1]
                return version
    raise RuntimeError("Unable to find version string.")


setup(
    name="clmm",
    version=get_version("clmm/__init__.py"),
    author="The LSST DESC CLMM Contributors",
    license="BSD 3-Clause License",
    url="https://github.com/LSSTDESC/CLMM",
    packages=find_packages(),
    description="A comprehensive package for galaxy cluster weak lensing",
    long_description=open("README.md").read(),
    package_data={"": ["README.md", "LICENSE"]},
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD 3-Clause",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=["astropy>=4.0", "numpy", "scipy", "healpy"],
    python_requires=">" + str(required_py_version),
)
