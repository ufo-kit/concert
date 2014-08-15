"""__init__.py"""
__version__ = '0.9.0'


def get_canonical_version(version=None):
    """Return a version without any dev-suffixes"""
    if not version:
        version = __version__

    return version[:-3] if version.endswith('dev') else version


def require(version):
    """
    Check if the *version* string matches the installed version. If it is
    higher than the installed version, a RuntimeError is raised.
    """
    def get_triple(v):
        return tuple([int(x) for x in get_canonical_version(v).split('.')])

    current_triple = get_triple(__version__)
    required_triple = get_triple(version)

    if current_triple < required_triple:
        import warnings
        msg = "Concert {} is required but only version {} is installed"
        warnings.warn(msg.format(version, __version__), RuntimeWarning)
