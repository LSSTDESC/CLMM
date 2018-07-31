from setuptools import setup

setup(
      name='clmm',
      version='0.1',
      author='LSST-DESC CL WG',
      author_email='avestruz@uchicago.edu',
      url='https://github.com/LSSTDESC/CLMM',
      packages=["clmm", "clmm.models", "clmm.summarizer", "clmm.core"],
      description='A comprehensive package for galaxy cluster weak lensing',
      long_description=open("README.md").read(),
      package_data={"": ["README.md", "LICENSE"]},
      include_package_data=True,
      classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD 3-Clause",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python"
        ],
      install_requires=["astropy", "matplotlib", "numpy", "scipy"]
)
