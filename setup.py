import os
from concert import __version__
from setuptools import setup, find_packages


# Install Bash completion script only if installation is run as root
data_files = []

if hasattr(os, 'geteuid') and os.geteuid() == 0:
    data_files = [('/etc/bash_completion.d', ['extras/completion/concert.sh'])]


setup(
    name='concert',
    version=__version__,
    author='Matthias Vogelgesang',
    author_email='matthias.vogelgesang@kit.edu',
    url='http://github.com/ufo-kit/concert',
    license='LGPL',
    packages=find_packages(exclude=['*.tests']),
    scripts=['bin/concert'],
    data_files=data_files,
    exclude_package_data={'': ['README.rst']},
    description="Lightweight beamline control system",
    long_description=open('README.rst').read(),
    install_requires=['futures',
                      'numpy',
                      'prettytable',
                      'pyxdg',
                      'pint',
                      'scipy',
                      'six'],
    tests_require=['nose',
                   'testfixtures'],
    test_suite='concert.tests',
)
