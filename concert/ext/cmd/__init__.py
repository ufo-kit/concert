plugins = []

try:
    from . import spyder

    plugins.append(spyder.SpyderCommand())
except ImportError:
    pass
