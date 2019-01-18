# Contributing to CLMM

This is a brief guide to contributing to CLMM, including information about identifiying code issues and submitting code changes.

### Identifying Issues
Action items for CLMM code improvements are listed as [issues in the github repository](https://github.com/LSSTDESC/CLMM/issues). Issues marked with the label `good first issue` are well-suited for new contributors.

### Making and submitting changes
1. Clone the CLMM repository to your local machine.
``` bash
    git clone git@github.com:LSSTDESC/CLMM.git
```
2. Checkout a new branch to contain your code changes independently from the master code. [Branches](https://help.github.com/articles/about-branches/) allow you to isolate development work without affecting code in the repository. The `branchname` should be descriptive of your code changes. If you are addressing a particular Issue #`xx`, then `branchname` should be formatted as `issue/xx/summary` where `summary` is a description of your changes.
```bash
    git checkout -b branchname
```
3. Make your changes in the files stored in your local directory.
4. Commit and push your changes to the `branchname` branch of the remote repository. After this step, your changes will appear as an active branch of CLMM, [here](https://github.com/LSSTDESC/CLMM/branches).
```bash
    git add .
    git commit -m "Insert commit message here"
    git pull origin master
    git push origin branchname
```
5. You can continue to edit your code and push changes to the `branchname` remote branch. Once you are satisfied with your changes, you can submit a [pull request](https://help.github.com/articles/about-pull-requests/) to request that the changes you made in `branchname` be merged into the master repository. Navigate to the [CLMM pulls page](https://github.com/LSSTDESC/CLMM/pulls) and click 'New pull request.' Select `branchname`, fill out a name and description for the pull request, and submit for approval by CLMM admins. Once the pull request is approved, it will be merged into the CLMM master branch.
