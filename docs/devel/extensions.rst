==========
Extensions
==========

Concert allows third-party extensions to reside under a common namespace
``concert.third.*`` similar to the Flask extension system. To achieve this,
extensions must be modules or packages named ``concert_name`` and be installed
with setuptools like this::

    from setuptools import setup

    setup(
        name='Concert-Foo',
        version='1.0',
        url='...',
        author='...',
        py_modules=['concert_foo'],
        zip_safe=False,
        install_requires=[
            'concert',
        ]
    )

After successful installation, the user can import a third-party extension
simply like this::

    from concert.third.foo import SomeClass, some_func
