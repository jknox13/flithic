PYTHON ?= python
CYTHON ?= cython
PYTEST ?= pytest
CTAGS ?= ctags

all: clean inplace test

clean-ctags:
	rm -f tags

clean: clean-ctags
	$(PYTHON) setup.py clean
	rm -rf dist

in: inplace # just a shortcut
inplace:
	$(PYTHON) setup.py build_ext -i

test-code: in
	$(PYTEST) --showlocals -v flithic

test-coverage:
	rm -rf coverage .coverage
	$(PYTEST) flithic --showlocals -v --cov=flithic --cov-report=html:coverage

test: test-code

trailing-spaces:
	find flithic -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

cython:
	python setup.py build_src

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R flithic

code-analysis:
	flake8 flithic | grep -v __init__ | grep -v external
	pylint -E -i y flithic/ -d E1103,E0611,E1101
