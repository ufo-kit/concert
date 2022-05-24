PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = pytest concert/tests

.PHONY: build clean check check-fast check-without-async dist init install html

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

check-without-async:
	$(RUNTEST) --disable-async

clean:
	$(SETUP) clean --all

html:
	@cd docs; make html

init:
	pip install -r ./requirements.txt
