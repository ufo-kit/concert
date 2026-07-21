import inspect
from concert.base import AsyncObject, AsyncType
from concert.tests import TestCase


# Sync Inheritance Tree


class SyncRoot:
    def __init__(self, arg, kwarg=None):
        self.sync_arg = arg
        self.sync_kwarg = kwarg
        self.sync_root_called = True


class SyncClassA(SyncRoot):
    def __init__(self, arg, kwarg=None):
        self.arg = arg
        self.kwarg = kwarg
        self.sync_class_a_called = True
        super().__init__(arg, kwarg=kwarg)


class SyncClassB(SyncRoot):
    def __init__(self, arg, kwarg=None):
        self.arg = arg
        self.kwarg = kwarg
        self.sync_class_b_called = True
        super().__init__(arg, kwarg=kwarg)


class SyncClassC(SyncClassA, SyncClassB):
    """Sync counterpart of AsyncClassC."""
    def __init__(self, arg, kwarg=None):
        self.arg = arg
        self.kwarg = kwarg
        self.sync_class_c_called = True
        super().__init__(arg, kwarg=kwarg)


# Async Inheritance Tree


class AsyncRoot(AsyncObject):
    async def __ainit__(self, arg, kwarg=None, **kwargs):
        self.async_arg = arg
        self.async_kwarg = kwarg
        self.async_root_called = True
        await super().__ainit__(**kwargs)


class AsyncClassA(AsyncRoot):
    async def __ainit__(self, arg, kwarg=None, **kwargs):
        self.arg = arg
        self.kwarg = kwarg
        self.async_class_a_called = True
        await super().__ainit__(arg=arg, kwarg=kwarg)


class AsyncClassB(AsyncRoot):
    async def __ainit__(self, arg, kwarg=None, **kwargs):
        self.arg = arg
        self.kwarg = kwarg
        self.async_class_b_called = True
        await super().__ainit__(arg=arg, kwarg=kwarg)


class AsyncClassC(AsyncClassA, AsyncClassB):
    r"""
    Class with diamond inheritance for testing cooperative inheritance.

    Topology:

             AsyncObject
                  |
              AsyncRoot
                 /\
                /  \
               /    \
              /      \
             /        \
      AsyncClassA AsyncClassB
             \        /
              \      /
               \    /
                \  /
                 \/
             AsyncClassC
    """

    async def __ainit__(self, arg, kwarg=None, **kwargs):
        self.arg = arg
        self.kwarg = kwarg
        self.async_class_c_called = True
        await super().__ainit__(arg=arg, kwarg=kwarg, **kwargs)


class AsyncMixed(AsyncClassC, SyncClassC):
    async def __ainit__(self, **kwargs):
        super().__init__('sync', kwarg='skw')
        await super().__ainit__(arg='async', kwarg='akw', **kwargs)


class AsyncMixedInverted(SyncClassC, AsyncClassC):
    """Force mro to start with the sync classes, still everything must be initialized."""
    async def __ainit__(self, **kwargs):
        super().__init__('sync', kwarg='skw')
        await super().__ainit__(arg='async', kwarg='akw', **kwargs)


# Tests


class TestAsyncType(TestCase):
    async def test_new_class_init_defined(self):
        """This must raise a TypeError."""
        with self.assertRaises(TypeError):
            # It doesn't matter that __init__ is not a function
            AsyncType.__new__(AsyncType, 'Cls', (), {'__init__': ''})

    async def test_class_signature(self):
        sig = inspect.signature(AsyncRoot)
        self.assertTrue('arg' in sig.parameters)
        self.assertTrue('kwarg' in sig.parameters)

        # No __ainit__ must not prevent class construction
        AsyncType.__new__(AsyncType, 'Cls', (), {})


class TestAsyncObject(TestCase):
    async def test_new_undefined_ainit_undefined(self):
        """Trivial inheritance must work"""
        class NoRedefinitions(AsyncObject):
            pass

        with self.assertRaises(TypeError):
            # __new__ of AsyncObject must complain about not taking arguments
            await NoRedefinitions(1)

        # Trivial construction must pass
        obj = await NoRedefinitions()
        self.assertTrue(isinstance(obj, NoRedefinitions))

    async def test_new_defined_ainit_undefined(self):
        class NewRedefined(AsyncObject):
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls, *args, **kwargs)

        with self.assertRaises(TypeError):
            # __new__ of AsyncObject must complain about not taking arguments
            await NewRedefined(1)

        # Trivial construction which re-uses AsyncObject.__new__ without arguments must pass
        obj = await NewRedefined()
        self.assertTrue(isinstance(obj, NewRedefined))

    async def test_new_undefined_ainit_defined(self):
        class AinitRedefined(AsyncObject):
            async def __ainit__(self, **kwargs):
                await super().__ainit__(**kwargs)

        # Trivial construction must pass
        obj = await AinitRedefined()
        self.assertTrue(isinstance(obj, AinitRedefined))

    async def test_new_defined_ainit_defined(self):
        class BadNew(AsyncObject):
            def __new__(cls, *args, **kwargs):
                return super().__new__(cls, *args, **kwargs)

            async def __ainit__(self, arg, **kwargs):
                pass

        class BadAinit(AsyncObject):
            def __new__(cls, arg, *args, **kwargs):
                return super().__new__(cls, *args, **kwargs)

            async def __ainit__(self, **kwargs):
                pass

        class Correct(AsyncObject):
            async def __ainit__(self, arg, **kwargs):
                self.ainit_arg = arg

            def __new__(cls, arg, *args, **kwargs):
                obj = super().__new__(cls, *args, **kwargs)
                obj.new_arg = arg + 1

                return obj

        with self.assertRaises(TypeError):
            # __ainit__ must complain about the missing arg
            await BadNew()

        with self.assertRaises(TypeError):
            # __new__ must complain about the missing arg
            await BadAinit()

        with self.assertRaises(TypeError):
            # __new__ must complain about the missing arg
            await Correct()

        obj = await Correct(0)
        self.assertTrue(isinstance(obj, Correct))
        self.assertEqual(obj.new_arg, 1)
        self.assertEqual(obj.ainit_arg, 0)

    async def test_not_isinstance_construction(self):
        """A crippled class where __new__ is called but does not return an instance of its class."""
        class NewNone(AsyncObject):
            called = False

            def __new__(cls, *args, **kwargs):
                NewNone.called = True
                return None

        nn = await NewNone()
        self.assertEqual(nn, None)
        self.assertTrue(NewNone.called)

    async def test_ainit_not_coroutinefunction(self):
        class AsyncNotCoroutineFunction(AsyncObject):
            def __ainit__(self, **kwargs):
                async def foo():
                    self.called = True

                return foo()

        obj = await AsyncNotCoroutineFunction()
        self.assertTrue(obj.called)

    async def test_ainit_not_awaitable(self):
        class AsyncNotAwaitable(AsyncObject):
            def __ainit__(self, **kwargs):
                pass

        with self.assertRaises(TypeError):
            await AsyncNotAwaitable()

    async def test_ainit_returns(self):
        class AsyncReturning(AsyncObject):
            async def __ainit__(self, **kwargs):
                await super().__ainit__(**kwargs)
                return 0

        with self.assertRaises(TypeError):
            await AsyncReturning()

    async def test_trivial_construction(self):
        """Just and empty class even without an __ainit__."""
        coro = AsyncObject()
        self.assertTrue(inspect.isawaitable(coro))
        self.assertTrue(isinstance(await coro, AsyncObject))

    async def test_normal_construction(self):
        obj = await AsyncClassA(1, kwarg='kw')
        self.assertEqual(obj.arg, 1)
        self.assertEqual(obj.kwarg, 'kw')

    async def test_inheritance(self):
        async def _test_class(cls):
            obj = await cls()
            self.assertTrue(obj.async_class_a_called)
            self.assertTrue(obj.async_class_b_called)
            self.assertTrue(obj.async_class_c_called)
            self.assertTrue(obj.async_root_called)

            self.assertTrue(obj.sync_class_a_called)
            self.assertTrue(obj.sync_class_b_called)
            self.assertTrue(obj.sync_class_c_called)
            self.assertTrue(obj.sync_root_called)

            self.assertEqual(obj.sync_arg, 'sync')
            self.assertEqual(obj.sync_kwarg, 'skw')
            self.assertEqual(obj.async_arg, 'async')
            self.assertEqual(obj.async_kwarg, 'akw')

        await _test_class(AsyncMixed)
        await _test_class(AsyncMixedInverted)

    async def test_diamond_varied_ainit(self):
        # Define a diamond hierarchy with differing __ainit__ signatures
        class RootA(AsyncObject):
            async def __ainit__(self, a, **kwargs):
                self.a = a
                await super().__ainit__(**kwargs)

        class RootB(AsyncObject):
            async def __ainit__(self, b, extra=None, **kwargs):
                self.b = b
                self.extra = extra
                await super().__ainit__(**kwargs)

        class Mid(RootA, RootB):
            async def __ainit__(self, a, b, extra='default', **kwargs):
                self.mid = True
                await super().__ainit__(a=a, b=b, extra=extra, **kwargs)

        class Leaf(Mid):
            async def __ainit__(self, **kwargs):
                self.leaf = True
                await super().__ainit__(**kwargs)

        # Instantiate leaf with appropriate arguments
        obj = await Leaf(a='val_a', b='val_b', extra='val_extra')
        self.assertTrue(obj.leaf)
        self.assertTrue(obj.mid)
        self.assertEqual(obj.a, 'val_a')
        self.assertEqual(obj.b, 'val_b')
        self.assertEqual(obj.extra, 'val_extra')
