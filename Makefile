PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = $(PYTHON) runtests.py

.PHONY: build clean check check-fast dist init install html

all: build

install: clean build dist
	pip install dist/*.tar.gz

build:
	$(SETUP) build

dist:
	$(SETUP) sdist

check:
	$(RUNTEST)

check-fast:
	$(RUNTEST) -a '!slow'

clean:
	$(SETUP) clean --all

html:
	@cd docs; make html

init:
	pip install -r ./requirements.txt
