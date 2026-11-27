"""Qt-independent ownership and validation of asynchronous parameter operations."""
import asyncio
import time

from concert.base import QuantityValue


UNREAD = object()


async def validate_quantity(param, value):
    """Validate fresh soft limits before submitting a value to the device."""
    value = value.to(param.unit)
    lower, upper = await asyncio.gather(param.get_lower(), param.get_upper())
    if lower is not None and value < lower:
        raise ValueError(f'{value} is below the lower limit {lower}')
    if upper is not None and value > upper:
        raise ValueError(f'{value} is above the upper limit {upper}')
    return value


class ParameterController:
    """Own one operation at a time, independently of callers and GUI lifetime.

    Closing or hiding a UI cancels reads, not submitted hardware writes. Shutdown
    waits for writes to finish. Hardware abort remains a separate Concert action.
    Observers receive (event, controller); they must not access deleted widgets.
    """

    def __init__(self, param):
        self.param = param
        self.value = UNREAD
        self.last_read_at = None
        self.target = UNREAD
        self.error = None
        self.metadata_error = None
        self.lower = self.upper = None
        self.limits = {}
        self.written_value = UNREAD
        self.updated_limit = None
        self.operation = None
        self.closed = False
        self._task = None
        self._observers = []

    @property
    def busy(self):
        return self._task is not None and not self._task.done()

    @property
    def help(self):
        return self.param.help or ''

    def subscribe(self, observer):
        self._observers.append(observer)

    def unsubscribe(self, observer):
        if observer in self._observers:
            self._observers.remove(observer)

    def _emit(self, event):
        for observer in tuple(self._observers):
            observer(event, self)

    def request(self, operation, value=UNREAD):
        """Coalesce all requests while busy; claim ownership before scheduling."""
        if self.closed or self.busy:
            return None
        if operation not in ('read', 'write', 'limits'):
            raise ValueError('Unknown parameter operation')
        loop = asyncio.get_running_loop()
        self.operation = operation
        self._task = loop.create_task(self._run(operation, value))
        return self._task

    async def _run(self, operation, value):
        self._emit('started')
        try:
            if operation == 'limits':
                if not isinstance(self.param, QuantityValue):
                    raise ValueError('Only quantities have limits')
                if self.param.limits_locked:
                    raise ValueError('Limits are locked')
                name, bound = value
                if name not in ('lower', 'upper'):
                    raise ValueError('Unknown limit')
                await getattr(self.param, 'set_' + name)(bound)
                self.updated_limit = name
                names = ('lower', 'upper', 'lower_user', 'upper_user',
                         'lower_external', 'upper_external')
                results = await asyncio.gather(*(getattr(self.param, 'get_' + key)()
                                                 for key in names), return_exceptions=True)
                self.limits = {key: result for key, result in zip(names, results)
                               if not isinstance(result, Exception)}
                self.lower = self.limits.get('lower')
                self.upper = self.limits.get('upper')
                self.metadata_error = next((result for result in results
                                            if isinstance(result, Exception)), None)
                self._emit('limits')
                if self.metadata_error:
                    self._emit('metadata_error')
            elif operation == 'write':
                if not self.param.writable:
                    raise ValueError('Parameter is read-only')
                if self.param.locked:
                    raise ValueError('Parameter is locked')
                if isinstance(self.param, QuantityValue):
                    value = await validate_quantity(self.param, value)
                await self.param.set(value)
                self.written_value = value
                self._emit('written')
            else:
                value = await self.param.get()
                # Some Concert getters swallow cancellation and return None.
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
                self.value = value
                self.last_read_at = time.monotonic()
                self.target = UNREAD
                self.metadata_error = None
                readers = {}
                if isinstance(self.param, QuantityValue):
                    readers = {name: getattr(self.param, 'get_' + name)() for name in
                               ('lower', 'upper', 'lower_user', 'upper_user',
                                'lower_external', 'upper_external')}
                if self.param.target_readable:
                    readers['target'] = self.param.get_target()
                results = await asyncio.gather(*readers.values(), return_exceptions=True)
                self.limits = {}
                for name, result in zip(readers, results):
                    if isinstance(result, Exception):
                        self.metadata_error = result
                    elif name == 'target':
                        self.target = result
                    else:
                        self.limits[name] = result
                self.lower = self.limits.get('lower')
                self.upper = self.limits.get('upper')
                if self.metadata_error:
                    self._emit('metadata_error')
                if asyncio.current_task().cancelling():
                    raise asyncio.CancelledError
                self._emit('value')
            self.error = None
        except Exception as error:
            self.error = error
            self._emit('error')
        finally:
            self._task = None
            self._emit('finished')

    async def read(self):
        task = self.request('read')
        if task is None and self.busy and self.operation == 'read':
            # Explicit refresh callers can await an automatic/polling read
            # already in progress without submitting a duplicate request.
            task = self._task
        if task is not None:
            await asyncio.shield(task)

    async def write(self, value):
        task = self.request('write', value)
        if task is not None:
            # Cancelling a caller must not implicitly abort a hardware write.
            await asyncio.shield(task)

    def cancel_read(self):
        if self.busy and self.operation == 'read':
            self._task.cancel()

    def close(self):
        self.closed = True
        self.cancel_read()

    async def shutdown(self):
        self.close()
        if self._task is not None:
            await asyncio.gather(self._task, return_exceptions=True)
