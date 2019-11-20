from setuptools import setup, find_packages

setup(
      name='clmm',
      version='0.1',
      author='The LSST DESC CLMM Contributors',
      author_email='avestruz@uchicago.edu',
      license='BSD 3-Clause License',
      url='https://github.com/LSSTDESC/CLMM',
      packages=find_packages(),
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
      install_requires=["astropy", "numpy", "scipy"]
)
