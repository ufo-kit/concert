from distutils.core import setup

setup(
    name='bossa',
    version='0.0.1',
    author='John Doe',
    author_email='john.doe@kit.edu',
    packages=['control',
              'control/connections',
              'control/devices',
              'control/devices/motion',
              'control/devices/motion/axes',
              'control/devices/storagerings/',
              'control/events',
              'control/measure',
              'control/processes',
              ],
    description="ANKA control system",
)
