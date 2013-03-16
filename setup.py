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
              'concert/devices/cameras',
              'concert/devices/storagerings/',
              'concert/events',
              'concert/measures',
              'concert/optimization',
              'concert/processes',
              'concert/ui',
              ],
    scripts=['bin/autofocus',
             'bin/concert'],
    data_files=[('/etc/bash_completion.d', ['extras/completion/concert'])],
    description="Lightweight control of heterogeneous environment",
)
