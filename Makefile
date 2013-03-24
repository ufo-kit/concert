PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = nosetests

.PHONY: build clean check check-fast init install

all: build

install: clean build
	$(SETUP) install

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
