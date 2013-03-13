from concert import __version__
from distutils.core import setup

setup(
    name='concert',
    version=__version__,
    author='John Doe',
    author_email='john.doe@kit.edu',
    packages=['concert',
              'concert/controllers',
              'concert/devices',
              'concert/devices/axes',
              'concert/devices/storagerings/',
              'concert/events',
              'concert/feedback',
              'concert/processes',
              ],
    scripts=['bin/autofocus',
             'bin/concert'],
    description="Lightweight control of heterogeneous environment",
)
