import os
from concert import __version__
from setuptools import setup


# Install Bash completion script only if installation is run as root
if os.geteuid() != 0:
    data_files = []
else:
    data_files = [('/etc/bash_completion.d', ['extras/completion/concert.sh'])]


setup(
    name='concert',
    version=__version__,
    author='John Doe',
    author_email='john.doe@kit.edu',
    packages=['concert',
              'concert/devices',
              'concert/devices/cameras',
              'concert/devices/controllers',
              'concert/devices/io',
              'concert/devices/motors',
              'concert/devices/storagerings',
              'concert/devices/shutters',
              'concert/feedbacks',
              'concert/measures',
              'concert/optimization',
              'concert/processes',
              'concert/ui',
              ],
    scripts=['bin/autofocus',
             'bin/concert'],
    data_files=data_files,
    exclude_package_data={'': ['README.rst']},
    description="Lightweight control of heterogeneous environment",
    install_requires=['argparse',
                      'quantities',
                      'pyxdg',
                      'logbook',
                      'futures',
                      'prettytable'],
    test_suite='concert.tests'
)
