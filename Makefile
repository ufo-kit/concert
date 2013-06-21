PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = nosetests

.PHONY: build clean check check-fast dist init install

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

init:
	pip install -r ./requirements.txt
