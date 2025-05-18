# Contributing to CLMM

This is a brief guide to contributing to CLMM, including information about identifiying code issues and submitting code changes or documentation.

* [Main readme](README.md)

## Table of contents
1. [Identifying Issues](#identifying_issues)
2. [Making and submitting changes](#making_and_submitting_changes)
3. [Adding documentation](#adding_documentation)
4. [Reviewing an open pull request](#reviewing_an_open_pull_request)
5. [Steps to merging a pull request](#steps_to_merging_a_pull_request)
6. [Updating Public Documentation on lsstdesc.org](#updating_public_docs)
7. [Additional resources](#additional_resources)
8. [Contact](#contact)

## Identifying Issues <a name="identifying_issues"></a>

Action items for CLMM code improvements are listed as [GitHub Issues](https://github.com/LSSTDESC/CLMM/issues).
Issues marked with the label `good first issue` are well-suited for new contributors.


## Making and submitting changes <a name="making_and_submitting_changes"></a>
Once you've [created a local copy of CLMM](INSTALL.md) on your machine, you can begin making changes to the code and submitting them for review.
To do this, follow the following steps from within your local copy of CLMM (forked or base).

1. Checkout a new branch to contain your code changes independently from the `main` repository.
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
    git pull origin main
    git push origin branchname
    ```
4. You can continue to edit your code and push changes to the `branchname` remote branch.
    Once you are satisfied with your changes, you can submit a [pull request](https://help.github.com/articles/about-pull-requests/) to merge your changes from `branchname` into the `main` branch.
    Navigate to the [CLMM Pull Requests](https://github.com/LSSTDESC/CLMM/pulls) and click 'New pull request.'
    Select `branchname`, fill out a title and description for the pull request, and, optionally, request review by a CLMM team member.
    Once the pull request is approved, it will be merged into the CLMM `main` branch.


### Requirements for every change <a name="adding_codes"></a>

We want to make sure the code formatting in `CLMM` is standardized accordind to our guidelines, which facilitates the reading of the code.
The code is also not complete without unit tests and documentation. Please ensure that unit tests (both new and old) all pass and that docs compile successfully.
So remember these steps:

- **Formatting the code:** there are tools that will format the code automatically for you, so you don't have to worry about the correct syntax when developing it.
Once you add your changes, `black clmm/` and `isort clmm/` before you commit your changes. These tools will correctly format the spacing and importing order respectilely.
**Tip:** `black` can also be run on notebooks, just install the notebook extention: `pip install "black[jupyter]"`.

- **Structuring the code:** Run `pylint clmm/`, this will produce a diagnostic about the implementation and tell you what needs to be improved.

- **Reinstall CLMM:** After your changes, resinstall `CLMM` by running `python setup.py install --user` (required after any change whatsoever to the `.py` files in `clmm/` directory). This will ensure your implementation is being used in the tests and documentation.
**Developer tip:** You can install `CLMM` in a editable mode, where the latest files on the repo will always be used, with the command `pip install . -e`.
In this case you will not have to reinstall it at every change.

- **Unit tests:** To run all of the unit tests, run `pytest` in the root package directory.

- **Coverage:** Check the coverage of the code, i. e. how many lines of the code are being tested in our unittests. To do it locally, you need the `pytest-cov` library (it can be insatalled with `pip`). Then run the command `pytest tests/ --cov=clmm --cov-report term-missing`, it will tell you what fraction of the code is being tested and which lines are not being tested.

- **Notebooks:** Make sure all notebooks can run correctly. We provide a convenient bash script to compile the documentation, run `./run_notebooks` in the main `CLMM` directory. It will create an executed version of all notebooks in the `_executed_nbs/` directory. By default it uses a `python3` kernel, if you want to choose another one, just pass it to the command: `./run_notebooks -k MY_KERNEL`. (Note that depending on your theory backend, some cells in some notebooks will give an error. This expected behaviour is explicitely mentioned before the cells and should not be a cause of worry.)

- **Build the documentation:** To test the docs, in the root package directory after installing, run `./update_docs`. This script both deletes the old compiled documentation files and rebuilds them. You can view the compiled docs by running `open docs/_build/html/index.html`.

> **NOTE:** If the changes you are making affect which CCL versions are compatible with the code,
> please update `clmm/theory/_ccl_supported_versions.py`, `README.md` and `INSTALL.md` accordingly.

All these steps (except running the notebooks) are run automatically on each pull request on GitHub, so you will know if any of them require further attention before considering youre changes are ready to be reviewed. Check `details` on `Build and Check / build-gcc-ubuntu (pull_request)` section at the end of the PR and `coverage` under the `coveralls` comment in the middle of the PR.

## Adding documentation <a name="adding_documentation"></a>

If you are adding documentation either in the form of example jupyter notebooks or new python modules, your documentation will need to compile for our online documentation hosted by the LSST-DESC website: http://lsstdesc.org/CLMM/

We have done most of the hard work for you. Simply edit the configuration file, `docs/doc-config.ini`. If you are looking at add a module, put the module name under the `APIDOC` heading. If you are adding a demo notebook to demonstrate how to use the code, place the path from the `docs/` directory to the notebook under the `DEMO` heading. If you are adding an example notebook that shows off how to use `CLMM` to do science, place the path from the `docs/` directory to the notebook under the `EXAMPLE` heading. 

Once it has been added to the config file, simply run `./update_docs` from the top level directory of the repository and your documentation should compile and be linked in the correct places!


## Reviewing an open pull request <a name="reviewing_an_open_pull_request"></a>

To review an open pull request submitted by another developer, there are several steps that you should take.

1. **Code changes:** For each new or changed file, ensure that the changes are correct, well-documented, and easy to follow. If you notice anything that can be improved, leave an inline comment (click the line of code in the review interface on Github).
1. **Documentation:** Double check any relevant documentation. For any new or changed code, ensure that the documentation is accurate (i.e. references, equations, parameter descriptions).
1. **Check unittests:** For any new or changed code, ensure that there are new tests in the appropriate directory. Try to come up with additional tests that either do or can break the code. If you can think of any such tests, suggest adding them in your review.

1. **Tests for formatting, documentation and unit tests:** The CI runs automatically tests for code formatting, documentation and unit tests, so the checks will fail if there are any problems. To see how to perform these steps manually, check the [Requirements for every change](#adding_codes) section.

1. **Notebooks:** Make sure all notebooks can run correctly as this is not run by the CI, check [Requirements for every change](#adding_codes) for details.

1. **Documentation:** Check that the documentation looks as it should, see [Requirements for every change](#adding_codes) for details.

## Steps to merging a pull request <a name="steps_to_merging_a_pull_request"></a>

To ensure consistency between our code and documentation, we need to take care of a couple of more things after accepting a review on a PR into `main`.

1. In the branch of the pull request, change the version number of the code located in `clmm/__init__.py`, commit and push. If you are unsure of how you should change the version number, don't hesitate to ask!

We use [semantic versioning](https://semver.org/), X.Y.Z. If the PR makes a small change, such as a bug fix, documentation updates, style changes, etc., increment Z. If the PR adds a new feature, such as adding support for a new profile, increment Y (and reset Z to 0). If a PR adds a feature or makes a change that breaks the old API, increment X (and reset Y and Z to 0). After the first tagged release of CLMM, anything that is a candidate to increment X should be extensively discussed beforehand. 

2. "Squash and Merge" the pull request into `main`. It asks for a squashed commit message. This should be descriptive of the feature or bug resolved in the PR and should be pre-prended by a [conventional commit scope](https://www.conventionalcommits.org/).

Please choose from `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`. If this commit breaks the previous API, add an explanation mark (for example, `fix!:`). Definitions of each scope can be found at the above link.

Note: `fix:` should correspond to version changes to Y. The rest of the scopes above should be version changes to Z.

3. Tag and push this new version of the code. In the `main` branch use the following commands:

    ```bash
    git tag X.Y.Z
    git push --tag
    ```

of course replacing `X.Y.Z` by the new version.

## Updating Public Documentation on lsstdesc.org <a name="updating_public_docs"></a>

This is easy! Once you have merged all approved changes into `main`, you will want to update the public documentation.
All these steps should be done on the `publish-docs` branch (just `git checkout publish-docs` on your local computer):
1. Merge all of the latest changes from main `git merge main`.
2. If you have figures in notebooks that you would like rendered on the website, you will want to execute all cells of demo notebooks.
3. From the `main` CLMM directory (the one that contains `setup.py`) run `./publish_docs` (note, this is different from `./update_docs` that you did in your development branch) and it does all of the work for you (including automatically pushing changes to Github)!

## Additional resources <a name="additional_resources"></a>

Here's a list of additional resources which you may find helpful in navigating git for the first time.
* The DESC Confluence page on [Getting Started with Git and Github](https://confluence.slac.stanford.edu/display/LSSTDESC/Getting+Started+with+Git+and+GitHub)
* [Phil Marshall's Getting Started repository and FAQ](https://github.com/drphilmarshall/GettingStarted#forks)
* [Phil Marshall's Git tutorial video lesson](https://www.youtube.com/watch?v=2g9lsbJBPEs)
* [The Github Help Pages](https://help.github.com/)

## Contact (alphabetical order) <a name="contact"></a>
* [Michel Aguena](https://github.com/m-aguena) (INAF / LIneA)
* [Doug Applegate](https://github.com/deapplegate) (Novartis)
* [Camille Avestruz](https://github.com/cavestruz) (University of Michigan)
* [Lucie Baumont](https://github.com/lbaumo) (INAF)
* [Miyoung Choi](https://github.com/mchoi8739) (UTD)
* [Celine Combet](https://github.com/combet) (LPSC)
* [Matthew Fong](https://github.com/matthewwf2001) (UTD)
* [Shenming Fu](https://github.com/shenmingfu)(Brown)
* [Matthew Ho](https://github.com/maho3) (CMU)
* [Matthew Kirby](https://github.com/matthewkirby) (Arizona)
* [Brandyn Lee](https://github.com/brandynlee) (UTD)
* [Anja von der Linden](https://github.com/anjavdl) (SBU)
* [Binyang Liu](https://github.com/rbliu) (Brown)
* [Alex Malz](https://github.com/aimalz) (CMU)
* [Tom McClintock](https://github.com/tmcclintock) (BNL)
* [Hironao Miyatake](https://github.com/HironaoMiyatake) (Nagoya)
* [Constantin Payerne](https://github.com/payerne) (LPSC)
* [Mariana Penna-Lima](https://github.com/pennalima) (UnB - Brasilia / LIneA)
* [Marina Ricci](https://github.com/marina-ricci) (LMU)
* [Cristobal Sifon](https://github.com/cristobal-sifon) (Princeton)
* [Melanie Simet](https://github.com/msimet) (JPL)
* [Martin Sommer](https://github.com/sipplund) (Bonn)
* [Sandro Vitenti](https://github.com/vitenti) (LIneA / UEL - Londrina)
* [Heidi Wu](https://github.com/hywu) (Ohio)
* [Mijin Yoon](https://github.com/mijinyoon) (RUB)

The current administrators of the repository are Michel Aguena, Camille Avestruz, Céline Combet, Matthew Kirby, and Alex Malz.
