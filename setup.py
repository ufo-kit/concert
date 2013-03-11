from distutils.core import setup

setup(
    name='bossa',
    version='0.0.1',
    author='John Doe',
    author_email='john.doe@kit.edu',
    packages=['control',
              'control/controllers',
              'control/devices',
              'control/devices/axes',
              'control/devices/storagerings/',
              'control/events',
              'control/feedback',
              'control/processes',
              ],
    scripts=['bin/autofocus'],
    description="ANKA control system",
)
