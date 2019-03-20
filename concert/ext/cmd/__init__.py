plugins = []

try:
    import spyder_plugin

    plugins.append(spyder_plugin.SpyderCommand())
except ImportError:
    pass
