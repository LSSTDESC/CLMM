# Installation instructions

We strongly recommend you make a new `conda` environment for the installation of CLMM so you can avoid conflicts with versions of
libraries used for other programs.
Here we present several ways to do it.

## Table of contents
1. [Basic installation](#basic_install)
2. [Developer installation](#dev_install)
    * [Setting up the proper environment on cori.nersc.gov](#access_to_the_proper_environment_on_cori)
3. [An alternative installation at NERSC or at CC-IN2P3 for DESC members](#from_desc_conda_env)
4. [Making a local copy of CLMM](#making_a_local_copy_of_clmm)


## Basic installation <a name="basic_install"></a>

To create a `conda` environment, run:

```bash
    conda env create -n clmm
```

CLMM can be found both on PyPI and conda-forge repositories and can be easily installed with

```bash
    pip install clmm
```

or

```bash
    conda install -C conda-forge clmm
```

This should automatically pull all necessary dependencies, with the exception of the theory backend libraries.

### Theory backend installation
First, choose and install a theory backend for CLMM.
This can be CCL (versions 3.1.2 or later),
NumCosmo (versions between v0.19 and v0.22),
or cluster_toolkit and they are installable as follows.

To install CCL as the theory/cosmology backend, run

```bash
    conda install -c conda-forge pyccl
```
or
```bash
    pip install pyccl
```

To install NumCosmo, run

```bash
    conda install -c conda-forge numcosmo
```

Now, to install cluster-toolkit, cluster-toolkit has a gsl dependency, you'll also need gsl.

```bash
    conda install gsl
    git clone https://github.com/tmcclintock/cluster_toolkit.git
    cd cluster_toolkit
    python setup.py install
    cd ..
```
**Note**: While cluster-toolkit mentions the potential need to install CAMB/CLASS for all cluster-toolkit functionality, you do not need to install these to run CLMM.

Note, you may choose to install some or all of the ccl, numcosmo, and/or cluster_toolkit packages.  You need at least one.  If you install cluster_toolkit and others, then you need to install cluster_toolkit *last*.   If you have already installed cluster_toolkit before the other packages, simply run, `pip uninstall cluster_toolkit` then re-install cluster_toolkit.

## Developer installation <a name="dev_install"></a>

Here we provide a quick guide for a developer instalation, using `environment.yml` to create a suitable conda environment and install all necessary packages.

If you're creating this environment on NERSC, it is necessary to run this before creating the environment:

```bash
    module load python
```

To create and activate your conda environment, run:

```bash
    conda env create -f environment.yml
    conda activate clmm
```

This procedure installs all available theory backends as well as the tools necessary for development.


### Setting up the proper environment on cori.nersc.gov <a name="access_to_the_proper_environment_on_cori"></a>

If you have access to NERSC, this will likely be the easiest to make sure you have the appropriate environment.  After logging into cori.nersc.gov, you will need to execute the following.  We recommend executing line-by-line to avoid errors:

```bash
    module load python  # Also loads anaconda
    conda create --name clmmenv  # Create an anaconda environment for clmm
    source activate clmmenv  # switch to your newly created environment
    conda install pip  # need pip to install everything else necessary for clmm
    conda install ipython # need to have the ipython tied to this environment
    conda install -c conda-forge firefox  # Need a browser to view jupyter notebooks
```

Note, for regular contributions and use, we recommend adding `module load python` to your `~/.bashrc` so you have anaconda installed every time you log in.  You will subseqeuntly also want to be in the correct environment whenever working with `clmm`, which means running `source activate clmmenv` at the start of each session.

Once in your CLMM conda env, you may follow the [basic procedure](#basic_install) to install CLMM and its dependencies.

The above allows you to develop at NERSC and run pytest.  Your workflow as a developer would be to make your changes, do a `python setup.py install` then `pytest` to make sure your changes did not break any tests.

If you are a DESC member you may also add to your CLMM environment the GCR and GCRCatalog packages to access the DC2 datasets at NERSC. To run the DC2 example notebooks provided in CLMM, the following need to be installed in your CLMM environment at NERSC. Once in your CLMM environment (`source activate clmmenv`), run

```bash
    pip install pandas
    pip install pyarrow
    pip install healpy
    pip install h5py
    pip install GCR
    pip install https://github.com/LSSTDESC/gcr-catalogs/archive/master.zip
    pip install FoFCatalogMatching
```

To open up a notebook from NERSC in your browser, you will need to go to the [nersc jupyter portal](https://jupyter.nersc.gov) and sign in. You will need to make this conda environment available in the kernel list:

```bash
    python -m ipykernel install --user --name=conda-clmmenv
```

Clicking on the upper right corner of the notebook will provide options for your kernel.  Choose the kernel `conda-clmmenv` that you just created.  You will need to do a temporary install of both cluster_toolkit and clmm in the first cell of your jupyter notebook:

```python

def install_clmm_pipeline(upgrade=False):
    import sys
    try:
        import clmm
        import cluster_toolkit
        installed = True
    except ImportError:
        installed = False
    if not upgrade:
        print('clmm is already installed and upgrade is False')
    else:
        !{sys.executable} -m pip install --user --upgrade git+https://github.com/tmcclintock/cluster_toolkit.git
        !{sys.executable} -m pip install --user --upgrade git+https://github.com/LSSTDESC/CLMM
install_clmm_pipeline(upgrade=True)  # Comment this if you do not need to adjust your environment, but this is useful in cori

```

## An alternative installation at NERSC or at CC-IN2P3 for DESC members <a name="from_desc_conda_env"></a>

The LSST-DESC collabration has setup a specific conda python environment at both NERSC and CC-IN2P3. See instructions [there](https://github.com/LSSTDESC/desc-python). This conda environment comes with ready access to DESC specific ressources and software such as DC2 catalogs, GCRCatalogs or CCL. Cloning that conda environment and proceeding from there makes the installation procedure lighter as some packages won't need to be installed.

## Making a local copy of CLMM <a name="making_a_local_copy_of_clmm"></a>

As a newcomer, you likely will not have edit access to the main CLMM repository.
Without edit privileges, you won't be able to create or push changes to branches in the base repository. You can get around this by creating a [fork](https://help.github.com/articles/fork-a-repo/), a linked copy of the CLMM repository under your Github username. You can then push code changes to your fork which can later be merged with the base repository.

To create a fork, navigate to the [CLMM home page](https://github.com/LSSTDESC/CLMM) and click 'Fork' in the upper right hand corner. The fork has been created under your username on Github's remote server and can now be cloned to your local repository with

```bash
    git clone git@github.com:YOUR-USERNAME/CLMM.git
    cd CLMM
    git remote add base git@github.com:LSSTDESC/CLMM.git
```
If you do have edit privileges to CLMM, it may be easier to simply clone the base CLMM repository.
``` bash
    git clone git@github.com:LSSTDESC/CLMM.git
```
