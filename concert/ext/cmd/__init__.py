plugins = []

try:
    import spyder

    plugins.append(spyder.SpyderCommand())
except ImportError:
    pass
