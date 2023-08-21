PYTHON = python
SETUP = $(PYTHON) setup.py
RUNTEST = pytest concert/tests
RUNTEST_TANGO = pytest concert/tests/integration/tango
RUN_STATIC_TYPE_CHECK = mypy concert/networking/base.py
TANGO_SERVICE_DOCKER = docker-compose -f tango_service.yaml

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

check-tango:
	$(RUNTEST_TANGO)

type-check:
	$(RUN_STATIC_TYPE_CHECK)

clean:
	$(SETUP) clean --all

html:
	@cd docs; make html

init:
	pip install -r ./requirements.txt

tango-service-up:
	$(TANGO_SERVICE_DOCKER) up -d

tango-service-down:
	$(TANGO_SERVICE_DOCKER) down