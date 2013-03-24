PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = nosetests

.PHONY: clean check check-fast

all: build

install: clean build
	$(SETUP) install

build:
	$(SETUP) build

check:
	$(RUNTEST)

check-fast:
	$(RUNTEST) -a '!slow'

clean:
	$(SETUP) clean --all
