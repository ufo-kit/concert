PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = pytest concert/tests

.PHONY: build clean check check-fast dist init install html

all: build

install:
	$(PYTHON) setup.py install

build:
	$(SETUP) build

dist:
	$(SETUP) sdist

check:
	$(RUNTEST) --full-trace

check-fast:
	$(RUNTEST) -m 'not slow'

clean:
	$(SETUP) clean --all

html:
	@cd docs; make html

init:
	pip install -r ./requirements.txt
