"""Add-ons for acquisitions are standalone extensions which can be applied to them. They operate on
the acquired data, e.g. write images to disk, do tomographic reconstruction etc.
"""


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


class ImageWriter(Addon):

    """An addon which writes images to disk.

    .. py:attribute:: acquisitions

    a list of :class:`~concert.experiments.base.Acquisition` objects

    .. py:attribute:: walker

    A :class:`~concert.storage.Walker` instance
    """

    def __init__(self, acquisitions, walker):
        self.walker = walker
        self._writers = {}
        super(ImageWriter, self).__init__(acquisitions)

    def attach(self):
        """attach all acquisitions."""
        for acq in self.acquisitions:
            self._writers[acq] = self._write_sequence(acq)
            acq.consumers.append(self._writers[acq])

    def detach(self):
        """Unattach all acquisitions."""
        for acq in self.acquisitions:
            acq.consumers.remove(self._writers[acq])
            del self._writers[acq]

    def _write_sequence(self, acquisition):
        """Wrap the walker and write data."""
        def wrapped_writer():
            """Returned wrapper."""
            try:
                self.walker.descend(acquisition.name)
                return self.walker.write()
            finally:
                self.walker.ascend()

        return wrapped_writer


class AddonError(Exception):

    """Addon errors."""

    pass
