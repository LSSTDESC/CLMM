# Contributing to CLMM

This is a brief guide to contributing to CLMM, including information about identifiying code issues and submitting code changes.

## Identifying Issues

Action items for CLMM code improvements are listed as [GitHub Issues](https://github.com/LSSTDESC/CLMM/issues).
Issues marked with the label `good first issue` are well-suited for new contributors.

## Access to the proper environment on cori.nersc.gov

If you have access to nersc, this will likely be the easiest to make sure you have the appropriate environment.  After logging into cori.nersc.gov, you will need to execute the following:

```bash
	bash source 
```

The above allows you to access a set of environments in your jupyter notebook.  To open up a notebook from nersc in your browser, you will need to go to the [nersc jupyter portal](https://jupyter.nersc.gov) and sign in.  Clicking on the upper right corner of the notebook will provide options for your kernel.

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
    git add .
    git commit -m "Insert a descriptive commit message here"
    git pull origin master
    git push origin branchname
    ```
4. You can continue to edit your code and push changes to the `branchname` remote branch.
    Once you are satisfied with your changes, you can submit a [pull request](https://help.github.com/articles/about-pull-requests/) to merge your changes from `branchname` into the master branch.
    Navigate to the [CLMM Pull Requests](https://github.com/LSSTDESC/CLMM/pulls) and click 'New pull request.'
    Select `branchname`, fill out a title and description for the pull request, and, optionally, request review by a CLMM team member.
    Once the pull request is approved, it will be merged into the CLMM master branch.

NOTE: Code is not complete without unit tests and documentation.
Please ensure that unit tests (both new and old) all pass and that docs compile successfully.
To test this, first install the code by running `python setup.py install --user`. To run all of the unit tests, run `pytest` in the root package directory.
To test the docs, in the root package directory, run `./compile-docs` to delete any existing documents and to rebuild the documentation. You can view the compiled docs by running `open docs/_build/html/index.html`.


## Reviewing an open pull request

To review an open pull request submitted by another developer, there are several steps that you should take.

1. For each new or changed file, ensure that the changes are correct, well-documented, and easy to follow. If you notice anything that can be improved, leave an inline comment (click the line of code in the review interface on Github).
2. For any new or changed code, ensure that there are new tests in the appropriate directory. Try to come up with additional tests that either do or can break the code. If you can think of any such tests, suggest adding them in your review.
3. Double check any relevant documentation. For any new or changed code, ensure that the documentation is accurate (i.e. references, equations, parameter descriptions).
4. Next, checkout the branch to a location that you can run `CLMM`. From the top level package directory (the directory that has `setup.py`) install the code via `python setup.py install --user`. Then, run `pytest` to run the full testing suite.
5. Now that tests are passing, the code likely works (assuming we have sufficient tests!) so we want to finalize the new code. We can do this by running a linter on any new or changed files `pylint {filename}`. This will take a look at the file and identify any style problems. If there are only a couple, feel free to resolve them yourself, otherwise leave a comment in your review that the author should perform this step.
6. We can now check that the documentation looks as it should. We provide a convenient bash script to compile the documentation. To completely rebuild the documentation, AFTER INSTALLING (if you made any changes, even to docstrings), run `./compile-docs`. This will delete any compiled documentation that may already exist and rebuild it. If this runs without error, you can then take a look by `open docs/_build/html/index.html`. Make sure any docstrings that were changed compile correctly.
7. Finally, install (`python setup.py install --user`), run tests (`pytest`), and compile documentation (`./compile-docs`) one file time and if everything passes, accept the review!

NOTE: We have had several branches that have exploded in commit number. If you are merging a branch and it has more than ~20 commits, strongly recommend using the "Squash and Merge" option for merging a branch.

## Additional resources

Here's a list of additional resources which you may find helpful in navigating git for the first time.
* The DESC Confluence page on [Getting Started with Git and Github](https://confluence.slac.stanford.edu/display/LSSTDESC/Getting+Started+with+Git+and+GitHub)
* [Phil Marshall's Getting Started repository and FAQ](https://github.com/drphilmarshall/GettingStarted#forks)
* [Phil Marshall's Git tutorial video lesson](https://www.youtube.com/watch?v=2g9lsbJBPEs)
* [The Github Help Pages](https://help.github.com/)

## Contact (alphabetical order)
* [Michel Aguena](https://github.com/m-aguena) (LIneA)
* [Doug Applegate](https://github.com/deapplegate) (Novartis)
* [Camille Avestruz](https://github.com/cavestruz) (UChicago)
* [Lucie Baumont](https://github.com/lbaumo) (SBU)
* [Miyoung Choi](https://github.com/mchoi8739) (UTD)
* [Celine Combet](https://github.com/combet) (LSPC)
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
