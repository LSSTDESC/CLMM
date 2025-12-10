## Description

<!-- Briefly describe the purpose of this pull request, or link it to the issue it solves-->

## Main changes
<!-- List the major changes in this PR-->
- Change 1
- Change 2

## Checklist

Besides passing all CI checks and coverage is at 100%, make sure you also checked the following items
(check details in [CONTRIBUTING](https://github.com/LSSTDESC/CLMM/blob/main/CONTRIBUTING.md)).

### For developers

- [ ] **Notebooks:** notebooks related to this PR have been updated and all notebooks can run correctly.
- [ ] **Build the documentation:** All documentation builds correctly.

### For reviewers

- [ ] **Notebooks:** notebooks related to this PR have been updated and all notebooks can run correctly.
- [ ] **Build the documentation:** All documentation builds correctly.

### For developers (part 2)

After the PR has been approved by two reviewers:

- [ ] Update the code version in `clmm/__init__.py`.
- [ ] Keep only relevant points in the squash and merge commit message.
- [ ] If any dependencies have been altered, update `environment.yml`, `pyproject.toml`, `INSTALL.md`, and `README.md`. A maintainer should also be notified to change the requirements on conda-forge.
- [ ] Update `clmm/theory/_ccl_supported_versions.py` and `clmm/theory/ccl.py` if pyccl's version constraints have been altered.
