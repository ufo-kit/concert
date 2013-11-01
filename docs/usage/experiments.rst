===========
Experiments
===========


Experiments help to conduct real experiments more effectively without needing to worry
about current directory path and creation, saving files and so on. Particular
:class:`.Experiment` subclass defines how the experiment steps are conducted and brings
together data acquisition and processing. The base class provides directory
creation functionality and logging redirection into a file in the current
experiment directory.


.. autoclass:: concert.experiments.base.Experiment
    :members:


Imaging
-------

Imaging experiments all subclass :class:`.Radiography`. Every experiment
consists of taking dark, flat fields and radiographs. If the particular
methods are not provided or overriden, the image type is skipped.


.. autoclass:: concert.experiments.imaging.Radiography
    :members:
