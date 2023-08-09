PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = pytest concert/tests
RUN_STATIC_TYPE_CHECK = mypy concert/networking/base.py

.PHONY: build clean check check-fast dist init install html

all: build

install:
	$(PYTHON) setup.py install

build:
	$(SETUP) build

dist:
	$(SETUP) sdist

check:
	$(RUNTEST)

check-fast:
	$(RUNTEST) -m 'not slow'

type-check:
	$(RUN_STATIC_TYPE_CHECK)

clean:
	$(SETUP) clean --all

html:
	@cd docs; make html

init:
	pip install -r ./requirements.txt
