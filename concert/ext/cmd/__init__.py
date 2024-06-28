plugins = []

from . import spyder, tango

plugins.append(spyder.SpyderCommand())
plugins.append(tango.TangoCommand())
