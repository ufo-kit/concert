# About

$FOO is a light-weight control system interface to control Tango and native
devices.


# Installation

## Global installation

Simply run

    sudo python setup.py install

or (in the far future)

    sudo pip install $FOO


## Installing into a virtualenv

$FOO can be installed into a virtualenv without polluting your global Python
environment. Install virtualenv and virtualenvwrapper and follow these steps:

1. `mkvirtualenv $FOO`
2. `workon $FOO`
3. `cd $FOO`
4. `pip -r requirements.txt -e .`
5. `nosetests`
