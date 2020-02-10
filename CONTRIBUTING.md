# Contributing to CLMM

This is a brief guide to contributing to CLMM, including information about identifiying code issues and submitting code changes or documentation.

## Identifying Issues

Action items for CLMM code improvements are listed as [GitHub Issues](https://github.com/LSSTDESC/CLMM/issues).
Issues marked with the label `good first issue` are well-suited for new contributors.

## Access to the proper environment on cori.nersc.gov

If you have access to nersc, this will likely be the easiest to make sure you have the appropriate environment.  After logging into cori.nersc.gov, you will need to execute the following:

```bash
	module load python  # Also loads anaconda
	conda create --name clmmenv  # Create an anaconda environment for clmm
	source activate clmmenv  # switch to your newly created environment
	conda install pip  # need pip to install everything else necessary for clmm	
	conda install ipython # need to have the ipython tied to this environment
	conda install -c conda-forge firefox  # Need a browser to view jupyter notebooks  
```

You can now go through the steps in the Requirements section of README.md.  Note, you'll need to separately install cluster-toolkit in the current version of CLMM.  Since cluster-toolkit has a gsl dependency, you'll also need gsl.

```bash
	conda install gsl
	git clone https://github.com/tmcclintock/cluster_toolkit.git
	cd cluster_toolkit
	python setup.py install
	cd ..
```

Now, you should have cluster_toolkit installed, and are ready to install CLMM

```bash
	pip install numpy scipy astropy matplotlib
	pip install pytest sphinx sphinx_rtd_theme
	pip install jupyter  # need to have jupyter notebook tied to this environment, you can then see the environment in jupyter.nersc.gov
	git clone https://github.com/LSSTDESC/CLMM.git  # For those with edit access to CLMM, see below for otherwise
  	cd CLMM   
  	python setup.py install --user     # build from source
```

The above allows you to develop in NERSC and run pytest.  Your workflow as a developer would be to make your changes, do a `python setup.py install --user` then `pytest` to make sure your changes did not break any tests.

To open up a notebook from nersc in your browser, you will need to go to the [nersc jupyter portal](https://jupyter.nersc.gov) and sign in.  Clicking on the upper right corner of the notebook will provide options for your kernel.  Choose your `conda env:conda-clmmenv` that you just created.  You will need to do a temporary install of both cluster_toolkit and clmm in the first cell of your jupyter notebook:

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

## Making a local copy of CLMM

As a newcomer, you likely will not have edit access to the main CLMM repository.
Without edit privileges, you won't be able to create or push changes to branches in the base repository.
You can get around this by creating a [fork](https://help.github.com/articles/fork-a-repo/), a linked copy of the CLMM repository under your Github username.
You can then push code changes to your fork which can later be merged with the base repository.
To create a fork, navigate to the [CLMM home page](https://github.com/LSSTDESC/CLMM) and click 'Fork' in the upper right hand corner.
The fork has been created under your username on Github's remote server and can now be cloned to your local repository with

```bash
    git clone git@github.com:YOUR-USERNAME/CLMM.git
    git remote add base git@github.com:LSSTDESC/CLMM.git
```
If you do have edit privileges to CLMM, it may be easier to simply clone the base CLMM repository.
``` bash
    git clone git@github.com:LSSTDESC/CLMM.git
```

## Making and submitting changes
Once you've created a local copy of CLMM on your machine, you can begin making changes to the code and submitting them for review.
To do this, follow the following steps from within your local copy of CLMM (forked or base).

1. Checkout a new branch to contain your code changes independently from the master repository.
    [Branches](https://help.github.com/articles/about-branches/) allow you to isolate temporary development work without permanently affecting code in the repository.
    ```bash
    git checkout -b branchname
    ```
    Your `branchname` should be descriptive of your code changes.
    If you are addressing a particular issue #`xx`, then `branchname` should be formatted as `issue/xx/summary` where `summary` is a description of your changes.
2. Make your changes in the files stored in your local directory.
3. Commit and push your changes to the `branchname` branch of the remote repository.
    ```bash
    git add NAMES-OF-CHANGED-FILES
    git commit -m "Insert a descriptive commit message here"
    git pull origin master
    git push origin branchname
    ```
4. You can continue to edit your code and push changes to the `branchname` remote branch.
    Once you are satisfied with your changes, you can submit a [pull request](https://help.github.com/articles/about-pull-requests/) to merge your changes from `branchname` into the master branch.
    Navigate to the [CLMM Pull Requests](https://github.com/LSSTDESC/CLMM/pulls) and click 'New pull request.'
    Select `branchname`, fill out a title and description for the pull request, and, optionally, request review by a CLMM team member.
    Once the pull request is approved, it will be merged into the CLMM master branch.

NOTE: Code is not complete without unit tests and documentation. Please ensure that unit tests (both new and old) all pass and that docs compile successfully.

To test this, first install the code by running `python setup.py install --user` (required after any change whatsoever to the `.py` files in `clmm/` directory). To run all of the unit tests, run `pytest` in the root package directory. To test the docs, in the root package directory after installing, run `./update_docs`. This script both deletes the old compiled documentation files and rebuilds them. You can view the compiled docs by running `open docs/_build/html/index.html`.

## Adding documentation

If you are adding documentation either in the form of example jupyter notebooks or new python modules, your documentation will need to compile for our online documentation hosted by the LSST-DESC website: http://lsstdesc.org/CLMM/

We have done most of the hard work for you. Simply edit the configuration file, `docs/doc-config.ini`. If you are looking at add a module, put the module name under the `APIDOC` heading. If you are adding a demo notebook to demonstrate how to use the code, place the path from the `docs/` directory to the notebook under the `DEMO` heading. If you are adding an example notebook that shows off how to use `CLMM` to do science, place the path from the `docs/` directory to the notebook under the `EXAMPLE` heading. 

Once it has been added to the config file, simply run `./update_docs` from the top level directory of the repository and your documentation should compile and be linked in the correct places!


## Reviewing an open pull request

To review an open pull request submitted by another developer, there are several steps that you should take.

1. For each new or changed file, ensure that the changes are correct, well-documented, and easy to follow. If you notice anything that can be improved, leave an inline comment (click the line of code in the review interface on Github).
2. For any new or changed code, ensure that there are new tests in the appropriate directory. Try to come up with additional tests that either do or can break the code. If you can think of any such tests, suggest adding them in your review.
3. Double check any relevant documentation. For any new or changed code, ensure that the documentation is accurate (i.e. references, equations, parameter descriptions).
4. Next, checkout the branch to a location that you can run `CLMM`. From the top level package directory (the directory that has `setup.py`) install the code via `python setup.py install --user`. Then, run `pytest` to run the full testing suite.
5. Now that tests are passing, the code likely works (assuming we have sufficient tests!) so we want to finalize the new code. We can do this by running a linter on any new or changed files `pylint {filename}`. This will take a look at the file and identify any style problems. If there are only a couple, feel free to resolve them yourself, otherwise leave a comment in your review that the author should perform this step.
6. We can now check that the documentation looks as it should. We provide a convenient bash script to compile the documentation. To completely rebuild the documentation, AFTER INSTALLING (if you made any changes, even to docstrings), run `./update_docs`. This will delete any compiled documentation that may already exist and rebuild it. If this runs without error, you can then take a look by `open docs/_build/html/index.html`. Make sure any docstrings that were changed compile correctly.
7. Finally, install (`python setup.py install --user`), run tests (`pytest`), and compile documentation (`./update_docs`) one file time and if everything passes, accept the review!

NOTE: We have had several branches that have exploded in commit number. If you are merging a branch and it has more than ~20 commits, strongly recommend using the "Squash and Merge" option for merging a branch.

## Steps to merging a pull request

To ensure consistency between our code and documentation, we need to take care of a couple of more things after accepting a review on a PR into master.

1. Change the version number of the code located in `clmm/__init__.py`, commit the change to the branch, and push. If you are unsure of how you should change the version number, don't hesitate to ask!

We use [semantic versioning](https://semver.org/), X.Y.Z.. If the PR makes a small change, such as a bug fix, documentation updates, style changes, etc., increment Z. If the PR adds a new feature, such as adding support for a new profile, increment Y (and reset Z to 0). If a PR adds a feature or makes a change that breaks the old API, increment X (and reset Y and Z to 0). After the first tagged release of CLMM, anything that is a candidate to increment X should be extensively discussed beforehand. 

2. "Squash and Merge" the pull request into master. It asks for a squashed commit message. This should be descriptive of the feature or bug resolved in the PR and should be pre-prended by a [conventional commit scope](https://www.conventionalcommits.org/).

Please choose from `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`. If this commit breaks the previous API, add an explanation mark (for example, `fix!:`). Definitions of each scope can be found at the above link.

Note: `fix:` should correspond to version changes to Y. The rest of the scopes above should be version changes to Z.

3. Update the public documentation.

This is easy! On your local computer just `git checkout publish-docs` to access the branch that hosts the compiled documentation.   You will then need to merge all of the latest changes from master `git merge master`.  Next, from the main CLMM directory (the one that contains `setup.py`) run `./publish_docs` and it does all of the work for you (including automatically pushing changes to Github)!  Note, you will want to execute all cells of demo notebooks before running`./publish_docs` in order for the output to show in the public documentation.

## Additional resources

Here's a list of additional resources which you may find helpful in navigating git for the first time.
* The DESC Confluence page on [Getting Started with Git and Github](https://confluence.slac.stanford.edu/display/LSSTDESC/Getting+Started+with+Git+and+GitHub)
* [Phil Marshall's Getting Started repository and FAQ](https://github.com/drphilmarshall/GettingStarted#forks)
* [Phil Marshall's Git tutorial video lesson](https://www.youtube.com/watch?v=2g9lsbJBPEs)
* [The Github Help Pages](https://help.github.com/)

## Contact (alphabetical order)
* [Michel Aguena](https://github.com/m-aguena) (LIneA)
* [Doug Applegate](https://github.com/deapplegate) (Novartis)
* [Camille Avestruz](https://github.com/cavestruz) (University of Michigan)
* [Lucie Baumont](https://github.com/lbaumo) (SBU)
* [Miyoung Choi](https://github.com/mchoi8739) (UTD)
* [Celine Combet](https://github.com/combet) (LPSC)
* [Matthew Fong](https://github.com/matthewwf2001) (UTD)
* [Shenming Fu](https://github.com/shenmingfu)(Brown)
* [Matthew Ho](https://github.com/maho3) (CMU)
* [Matthew Kirby](https://github.com/matthewkirby) (Arizona)
* [Brandyn Lee](https://github.com/brandynlee) (UTD)
* [Anja von der Linden](https://github.com/anjavdl) (SBU)
* [Binyang Liu](https://github.com/rbliu) (Brown)
* [Alex Malz](https://github.com/aimalz) (NYU --> RUB)
* [Tom McClintock](https://github.com/tmcclintock) (BNL)
* [Hironao Miyatake](https://github.com/HironaoMiyatake) (Nagoya)
* [Mariana Penna-Lima](https://github.com/pennalima) (UBrasilia)
* [Marina Ricci](https://github.com/marina-ricci) (LAPP)
* [Cristobal Sifon](https://github.com/cristobal-sifon) (Princeton)
* [Melanie Simet](https://github.com/msimet) (JPL)
* [Martin Sommer](https://github.com/sipplund) (Bonn)
* [Heidi Wu](https://github.com/hywu) (Ohio)
* [Mijin Yoon](https://github.com/mijinyoon) (RUB)

The current administrators of the repository are Michel Aguena, Camille Avestruz, Matthew Kirby, and Alex Malz.
