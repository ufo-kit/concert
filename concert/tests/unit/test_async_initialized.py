import inspect
from concert.base import AsyncInitialized, AsyncMeta
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


class AsyncRoot(AsyncInitialized):
    async def __ainit__(self, arg, kwarg=None):
        self.async_arg = arg
        self.async_kwarg = kwarg
        self.async_root_called = True


class AsyncClassA(AsyncRoot):
    async def __ainit__(self, arg, kwarg=None):
        self.arg = arg
        self.kwarg = kwarg
        self.async_class_a_called = True
        await super().__ainit__(arg, kwarg=kwarg)


class AsyncClassB(AsyncRoot):
    async def __ainit__(self, arg, kwarg=None):
        self.arg = arg
        self.kwarg = kwarg
        self.async_class_b_called = True
        await super().__ainit__(arg, kwarg=kwarg)


class AsyncClassC(AsyncClassA, AsyncClassB):
    """
    Class with diamond inheritance for testing cooperative inheritance.

    Topology:

            AsyncInitialized
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

    async def __ainit__(self, arg, kwarg=None):
        self.arg = arg
        self.kwarg = kwarg
        self.async_class_c_called = True
        await super().__ainit__(arg, kwarg=kwarg)


class AsyncMixed(AsyncClassC, SyncClassC):
    async def __ainit__(self):
        super().__init__('sync', kwarg='skw')
        await super().__ainit__('async', kwarg='akw')


class AsyncMixedInverted(SyncClassC, AsyncClassC):
    """Force mro to start with the sync classes, still everything must be initialized."""
    async def __ainit__(self):
        super().__init__('sync', kwarg='skw')
        await super().__ainit__('async', kwarg='akw')


# Tests


class TestAsyncInitialized(TestCase):
    async def test_new_class_init_defined(self):
        """This must raise a TypeError."""
        with self.assertRaises(TypeError):
            # It doesn't matter that __init__ is not a function
            AsyncMeta.__new__(AsyncMeta, 'Cls', (), {'__init__': ''})

    async def test_custom_new(self):
        """A full class where __new__ and __ainit__ must be called."""
        class AsyncCustomNew(AsyncInitialized):
            async def __ainit__(self, arg):
                self.ainit_arg = arg

            def __new__(cls, arg, *args, **kwargs):
                obj = super().__new__(cls, *args, **kwargs)
                obj.new_arg = arg + 1

                return obj

        obj = await AsyncCustomNew(0)
        self.assertTrue(isinstance(obj, AsyncCustomNew))
        self.assertEqual(obj.new_arg, 1)
        self.assertEqual(obj.ainit_arg, 0)

    async def test_not_isinstance_construction(self):
        """A crippled class where __new__ is called but does not return an instance of its class."""
        class NewNone(AsyncInitialized):
            called = False

            def __new__(cls, *args, **kwargs):
                NewNone.called = True
                return None

        nn = await NewNone()
        self.assertEqual(nn, None)
        self.assertTrue(NewNone.called)

    async def test_no_ainit_construction(self):
        class NoAinit(metaclass=AsyncMeta):
            pass

        obj = await NoAinit()
        self.assertTrue(isinstance(obj, NoAinit))

    async def test_ainit_not_coroutinefunction(self):
        class AsyncNotCoroutineFunction(AsyncInitialized):
            def __ainit__(self):
                async def foo():
                    self.called = True

                return foo()

        obj = await AsyncNotCoroutineFunction()
        self.assertTrue(obj.called)

    async def test_ainit_not_awaitable(self):
        class AsyncNotAwaitable(AsyncInitialized):
            def __ainit__(self):
                pass

        with self.assertRaises(TypeError):
            await AsyncNotAwaitable()

    async def test_ainit_returns(self):
        class AsyncReturning(AsyncInitialized):
            async def __ainit__(self):
                return 0

        with self.assertRaises(TypeError):
            await AsyncReturning()

    async def test_trivial_construction(self):
        """Just and empty class even without an __ainit__."""
        coro = AsyncInitialized()
        self.assertTrue(inspect.isawaitable(coro))
        self.assertTrue(isinstance(await coro, AsyncInitialized))

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
