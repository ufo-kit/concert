import os
from concert import __version__
from distutils.core import setup


# Install Bash completion script only if installation is run as root
if os.geteuid() != 0:
    data_files = []
else:
    data_files = [('/etc/bash_completion.d', ['extras/completion/concert'])]


setup(
    name='concert',
    version=__version__,
    author='John Doe',
    author_email='john.doe@kit.edu',
    packages=['concert',
              'concert/controllers',
              'concert/devices',
              'concert/devices/cameras',
              'concert/devices/motors',
              'concert/devices/storagerings/',
              'concert/events',
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
                      'logbook']
)
