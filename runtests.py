import nose
from concert.ext.noseplugin import DisableAsync, DisableGevent
import sys

if __name__ == '__main__':
    sys.exit(nose.main(addplugins=[DisableAsync(), DisableGevent()]))
