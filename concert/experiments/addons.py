"""Add-ons for acquisitions are standalone extensions which can be applied to them. They operate on
the acquired data, e.g. write images to disk, do tomographic reconstruction etc.
"""
from concert.coroutines.filters import queue
from concert.coroutines.sinks import Accumulate


class Addon(object):

    """A base addon class. An addon can be attached, i.e. its functionality is applied to the
    specified *acquisitions* and detached.

    .. py:attribute:: acquisitions

    A list of :class:`~concert.experiments.base.Acquisition` objects. The addon attaches itself on
    construction.

    """

    def __init__(self, acquisitions):
        self.acquisitions = acquisitions
        self.attach()

    def attach(self):
        """attach adds the addon to an experiment. This means all the necessary operations which
        provide the addon functionality should be implemented in this method. This mostly means
        appending consumers to acquisitions.
        """
        pass

    def detach(self):
        """Unattach removes the addon from an experiment. This means all the necessary operations
        which provide the addon functionality should be undone by this method. This mostly means
        removing consumers from acquisitions.
        """
        pass


class Consumer(Addon):

    """An addon which applies a specific coroutine-based consumer to acquisitions.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: consumer

    A callable which returns a coroutine which processes the incoming data from acquisitions

    """

    def __init__(self, acquisitions, consumer):
        self.consumer = consumer
        super(Consumer, self).__init__(acquisitions)

    def attach(self):
        """attach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.append(self.consumer)

    def detach(self):
        """Unattach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self.consumer)


class Accumulator(Addon):

    """An addon which accumulates data.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: shapes

    a list of shapes for different acquisitions

    .. py:attribute:: dtype

    the numpy data type
    """

    def __init__(self, acquisitions, shapes=None, dtype=None):
        self._accumulators = {}
        self._shapes = shapes
        self._dtype = dtype
        self.items = {}
        super(Accumulator, self).__init__(acquisitions)

    def attach(self):
        """attach all acquisitions."""
        shapes = (None,) * len(self.acquisitions) if self._shapes is None else self._shapes

        for i, acq in enumerate(self.acquisitions):
            self._accumulators[acq] = Accumulate(shape=shapes[i], dtype=self._dtype)
            self.items[acq] = self._accumulators[acq].items
            acq.consumers.append(self._accumulators[acq])

    def detach(self):
        """Unattach all acquisitions."""
        self.items = {}
        for acq in self.acquisitions:
            acq.consumers.remove(self._accumulators[acq])

        self._accumulators = {}


class ImageWriter(Addon):

    """An addon which writes images to disk.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: walker

    A :class:`~concert.storage.Walker` instance

    .. py:attribute:: async

    If True write images asynchronously
    """

    def __init__(self, acquisitions, walker, async=True):
        self.walker = walker
        self._async = async
        self._writers = {}
        super(ImageWriter, self).__init__(acquisitions)

    def attach(self):
        """attach all acquisitions."""
        for acq in self.acquisitions:
            block = True if acq == self.acquisitions[-1] else False
            self._writers[acq] = self._write_sequence(acq, block)
            acq.consumers.append(self._writers[acq])

    def detach(self):
        """Unattach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self._writers[acq])
            del self._writers[acq]

    def _write_sequence(self, acquisition, block):
        """Wrap the walker and write data."""
        def wrapped_writer():
            """Returned wrapper."""
            try:
                self.walker.descend(acquisition.name)
                coro = self.walker.write()
                if self._async:
                    coro = queue(coro, process_all=True, block=block)
                return coro
            finally:
                self.walker.ascend()

        return wrapped_writer


class AddonError(Exception):

    """Addon errors."""

    pass
