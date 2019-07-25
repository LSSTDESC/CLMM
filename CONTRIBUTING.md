# Contributing to CLMM

This is a brief guide to contributing to CLMM, including information about identifiying code issues and submitting code changes.

## Identifying Issues

Action items for CLMM code improvements are listed as [GitHub Issues](https://github.com/LSSTDESC/CLMM/issues).
Issues marked with the label `good first issue` are well-suited for new contributors.

## Making a local copy of CLMM

As a newcomer, you likely will not have edit access to the main CLMM repository.
Without edit privledges, you won't be able to create or push changes to branches in the base repository.
You can get around this by creating a [fork](https://help.github.com/articles/fork-a-repo/), a linked copy of the CLMM repository under your Github username.
You can then push code changes to your fork which can later be merged with the base repository.
To create a fork, navigate to the [CLMM home page](https://github.com/LSSTDESC/CLMM) and click 'Fork' in the upper right hand corner.
The fork has been created under your username on Github's remote server and can now be cloned to your local repository with

```bash
    git clone git@github.com:YOUR-USERNAME/CLMM.git
    git remote add base git@github.com:LSSTDESC/CLMM.git
```

If you do have edit privledges to CLMM, it may be easier to simply clone the base CLMM repository.

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
To run all of the unit tests, run `pytest` in the root package directory.
To test the docs, in the root package directory, run `make -C docs/ clean` to delete any existing documents and then `make -C docs/ html` to rebuild the documentation.
If you do not first run `clean`, you may compile locally but fail in continuous integration.

## Additional resources

Here's a list of additional resources which you may find helpful in navigating git for the first time.
* The DESC Confluence page on [Getting Started with Git and Github](https://confluence.slac.stanford.edu/display/LSSTDESC/Getting+Started+with+Git+and+GitHub)
* [Phil Marshall's Getting Started repository and FAQ](https://github.com/drphilmarshall/GettingStarted#forks)
* [Phil Marshall's Git tutorial video lesson](https://www.youtube.com/watch?v=2g9lsbJBPEs)
* [The Github Help Pages](https://help.github.com/)
