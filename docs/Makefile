# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = CLMM
SOURCEDIR     = .
BUILDDIR      = _build
RM	      = rm -rf
SPHINXAPIDOC  = sphinx-apidoc
MODULEDIR     = ../clmm

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: html
html: sphinx-apidoc
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(0) -W

.PHONY: clean
clean:
	@$(RM) _build/* api/ api.rst compiled-examples/ index.rst

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -W

sphinx-apidoc:
	@$(SPHINXAPIDOC) --separate --no-toc -f -M -o api $(MODULEDIR)
