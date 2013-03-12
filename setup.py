from distutils.core import setup

setup(
    name='concert',
    version='0.0.1',
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
