PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = nosetests -v

.PHONY: clean check check-fast

all: build

build:
	$(SETUP) build

check:
	$(RUNTEST)

check-fast:
	$(RUNTEST) -a '!slow'

clean:
	$(SETUP) clean --all
