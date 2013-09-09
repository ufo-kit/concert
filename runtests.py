import nose.core
from concert.ext.noseplugin import DisableAsync
import sys

if __name__ == '__main__':
    sys.exit(int(not nose.core.run(addplugins=[DisableAsync()])))
