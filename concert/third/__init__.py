def setup():
    from ..exthook import ExtensionImporter
    importer = ExtensionImporter(__name__)
    importer.install()


setup()
del setup
