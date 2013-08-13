import nose.core
from concert.ext.noseplugin import DisableAsync

if __name__ == '__main__':
    nose.core.run(addplugins=[DisableAsync()])
