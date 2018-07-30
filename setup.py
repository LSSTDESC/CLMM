from setuptools import setup

setup(name='clmm',
      version='0.1',
      author='DESC-CL-WG',
      author_email='avestruz@uchicago.edu, aimalz@nyu.edu',
      url='http://github.com/LSSTDESC/CLMM',
      packages=["clmm"],
      description='A comprehensive package for galaxy cluster weak lensing',
      long_description=open("README.md").read(),
      package_data={"": ["README.md", "LICENSE"]},
      include_package_data=True,
      license='MIT',
      install_requires=["astropy", "matplotlib", "numpy", "scipy"],
      zip_safe=False
)
