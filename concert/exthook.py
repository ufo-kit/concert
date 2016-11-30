"""
Extension hook system based on Flask's exthook.py, originally written by Armin
Ronacher and licensed under BSD license.
"""


import sys


class ExtensionImporter(object):

    def __init__(self, wrapper_module):
        self.wrapper_module = wrapper_module
        self.prefix = wrapper_module + '.'
        self.prefix_cutoff = wrapper_module.count('.') + 1

    def __eq__(self, other):
        return self.__class__.__module__ == other.__class__.__module__ and \
            self.__class__.__name__ == other.__class__.__name__ and \
            self.wrapper_module == other.wrapper_module

    def __ne__(self, other):
        return not self.__eq__(other)

    def install(self):
        sys.meta_path[:] = [x for x in sys.meta_path if self != x] + [self]

    def find_module(self, fullname, path=None):
        if fullname.startswith(self.prefix):
            return self

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        modname = fullname.split('.', self.prefix_cutoff)[self.prefix_cutoff]
        realname = 'concert_{}'.format(modname)
        __import__(realname)
        module = sys.modules[fullname] = sys.modules[realname]

        if '.' not in modname:
            setattr(sys.modules[self.wrapper_module], modname, module)

        return module
        raise ImportError('No module named %s' % fullname)
