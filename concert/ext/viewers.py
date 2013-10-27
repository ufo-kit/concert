"""Opening images in external programs."""
import atexit
import os
import signal
import tempfile
import time
try:
    from Queue import Empty
except ImportError:
    from queue import Empty
import logging
import numpy as np
from subprocess import Popen
from multiprocessing import Queue as MultiprocessingQueue, Process
from concert.helpers import coroutine, threaded
from concert.storage import write_tiff


LOG = logging.getLogger(__name__)
_PYPLOT_VIEWERS = []
_ORIG_SIGINT_HANDLER = signal.getsignal(signal.SIGINT)


def _terminate_pyplotviewers():
    """Terminate all :py:class:`PyplotViewer` isntances."""
    for viewer in _PYPLOT_VIEWERS:
        if viewer is not None:
            viewer.terminate()


def _sigint_handler(signum, frame):
    """
    Handle the interrupt signal in order to exit gracefully
    by terminating all the :py:class:`PyplotViewer` processes.
    """
    _terminate_pyplotviewers()
    # Call the original handler, but first check if it
    # actually can be called (depends on OS)
    if hasattr(_ORIG_SIGINT_HANDLER, "__call__"):
        _ORIG_SIGINT_HANDLER(signum, frame)


# Register sigint handler for closing all PyplotViewer instances
signal.signal(signal.SIGINT, _sigint_handler)
# Register termination of all the PyplotViewer isntances also on
# normal exit
atexit.register(_terminate_pyplotviewers)


def _start_command(program, image, writer=write_tiff):
    """
    Create a tmp file for dumping the *image* and use *program*
    to open that image. Use *writer* for writing the iamge to the disk.
    """
    tmp_file = tempfile.mkstemp()[1]
    try:
        full_path = writer(tmp_file, image).result()
        process = Popen([program, full_path])
        process.wait()
    finally:
        os.remove(full_path)


@threaded
def imagej(image, path="imagej", writer=write_tiff):
    """
    Open *image* in ImageJ found by *path*. *writer* specifies
    the written image file type.
    """
    _start_command(path, image, writer)


class PyplotViewer(object):

    """
    A base class for data viewer which sends commands to a matplotlib updater
    which runs in a separate process.

    .. py:attribute:: view_function

        The function which updates the figure based on the changed data. Its
        nomenclature has to be::

            foo(data, force=False)

        Where *force* determines whether the redrawing must be done or not. If
        it is False, the redrawing takes place if the data queue contains only
        the current data item. This prevents the actual drawer from being
        overwhelmed by the amount of incoming data.
    """

    def __init__(self, view_function):
        self._queue = MultiprocessingQueue()
        self._paused = False
        self._terminated = False
        self._coroutine = None
        self._proc = None
        # The udater is implementation-specific and must be provided by
        # the subclass by calling self._set_updater
        self._updater = None
        self.view_function = view_function

    def __call__(self, size=None):
        """
        Display a dynamic image in a separate thread in order not to
        stall program execution. If *size* is specified, the redrawing stops
        when *size* images come.
        """
        if self._coroutine is None:
            self._coroutine = self._update(size=size)

        return self._coroutine

    @coroutine
    def _update(self, size=None):
        """
        Display data image in a separate thread in order not to stall program
        execution. If *size* is specified, the redrawing stops when *size*
        data items come. This method does not force the redrawing unless the
        last item has come. The subclass must provide the view function
        :py:attr:`PyplotViewer.view_function`.
        """
        i = 0

        while True:
            item = yield
            if size is None or i < size:
                if size is not None and i == size - 1:
                    # Maximum number of items has come, end redrawing
                    self.view_function(item, force=True)
                else:
                    self.view_function(item)
                # This helps with the smoothness of drawing
                time.sleep(0.001)

            i += 1

    def _set_updater(self, updater):
        """
        Set the *updater*, now the process can start. This has to be called
        by the subclasses.
        """
        self._updater = updater
        self._proc = Process(target=self._run)
        self._proc.start()
        _PYPLOT_VIEWERS.append(self)

    def terminate(self):
        """Close all communication and terminate child process."""
        if not self._terminated:
            self._queue.close()
            self._proc.terminate()

    def pause(self):
        """Pause, no images are dispayed but image commands work."""
        self._paused = True

    def resume(self):
        """Resume the viewer."""
        self._paused = False

    def _run(self):
        """
        Run the process, i.e. wait for an image to come to the queue
        and dispaly it. This method is executed in a separate process.
        """
        # This import *must* be here, otherwise it doesn't work on Linux
        from matplotlib import pyplot as plt
        from matplotlib.animation import FuncAnimation

        try:
            figure = plt.figure()
            _ = FuncAnimation(figure, self._updater.process, interval=5,
                              blit=True)
            plt.show()

        except KeyboardInterrupt:
            plt.close("all")


class PyplotImageViewer(PyplotViewer):

    """Dynamic image viewer using matplotlib."""

    def __init__(self, imshow_kwargs=None, colorbar=True):
        super(PyplotImageViewer, self).__init__(self.show)
        self._has_colorbar = colorbar
        self._imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        self._make_imshow_defaults()
        self._set_updater(_PyplotImageUpdater(self._queue,
                                              self._imshow_kwargs,
                                              self._has_colorbar))

    def show(self, item, force=False):
        """
        show *item* into the redrawing queue. The item is truly inserted
        only if the queue is empty in order to guarantee that the newest
        image is drawn or if the *force* is True.
        """
        if not self._paused and (self._queue.empty() or force):
            self._queue.put((_PyplotImageUpdater.IMAGE, item))

    def set_limits(self, clim):
        """
        Update the colormap limits by *clim*, which is a (lower, upper)
        tuple.
        """
        self._queue.put((_PyplotImageUpdater.CLIM, clim))

    def set_colormap(self, colormap):
        """Set colormp of the shown image to *colormap*."""
        self._queue.put((_PyplotImageUpdater.COLORMAP, colormap))

    def _make_imshow_defaults(self):
        """Override matplotlib's image showing defafults."""
        from matplotlib import cm

        if "cmap" not in self._imshow_kwargs:
            self._imshow_kwargs["cmap"] = cm.gray
        if "interpolation" not in self._imshow_kwargs:
            self._imshow_kwargs["interpolation"] = "nearest"


class _PyplotUpdater(object):

    """
    Base class for animating a matploblib figure in a separate process.

    .. py:attribute:: queue

        A multiprocessing queue for receiving commands
    """

    def __init__(self, queue):
        self.queue = queue
        self.first = True
        # A dictionary in form command: method which tells the class what to do
        # for every received command
        self.commands = {}

    def process(self, iteration):
        """
        Update function which is going to be called by matplotlib's
        FuncAnimation instance.
        """
        try:
            if self.first:
                # Wait as much time as it takes for the first
                # time beacuse we don't want to show a window
                # with no image in it.
                item = self.queue.get()
                self.first = False
            else:
                item = self.queue.get(timeout=0.1)
            command, data = item
            self.commands[command](data)
        except Empty:
            pass
        finally:
            return self.get_artists()

    def get_artists(self):
        """
        Abstract function for getting all matplotlib artists which we want
        to redraw. Needs to be implemented by the subclass.
        """
        raise NotImplementedError


class _PyplotImageUpdater(_PyplotUpdater):

    """
    Private class for updating a matplotlib figure with an image stream.
    """

    IMAGE = "image"
    CLIM = "clim"
    COLORMAP = "colormap"

    def __init__(self, queue, imshow_kwargs, has_colorbar):
        super(_PyplotImageUpdater, self).__init__(queue)
        self.imshow_kwargs = imshow_kwargs
        self.has_colorbar = has_colorbar
        self.mpl_image = None
        self.colorbar = None
        self.clim = None
        self.colormap = None
        self.commands = {_PyplotImageUpdater.IMAGE: self.process_image,
                         _PyplotImageUpdater.CLIM: self.update_limits,
                         _PyplotImageUpdater.COLORMAP: self.update_colormap}

    def get_artists(self):
        """Get artists to return for matplotlib's animation."""
        retval = [] if self.mpl_image is None else [self.mpl_image]

        return retval

    def process_image(self, image):
        """Process the incoming *image*."""
        if self.mpl_image is not None and \
                self.mpl_image.get_size() != image.shape:
            # When the shape changes the axes needs to be reset
            self.mpl_image.axes.clear()
            self.mpl_image = None

        if self.mpl_image is None:
            # Either removed by shape change or first time drawing
            self.make_image(image)
        else:
            self.update_all(image)

    def update_limits(self, clim):
        """
        Update current colormap limits by *clim*, which is a tuple
        (lower, upper). If *clim* is None, the limit is reset to the span of
        the current image.
        """
        from matplotlib import pyplot as plt
        self.clim = clim

        if clim is None and self.mpl_image is not None:
            # Restore the full clim
            clim = self.mpl_image.get_array().min(), \
                self.mpl_image.get_array().max()

        if self.mpl_image is not None and clim is not None:
            self.mpl_image.set_clim(clim)
            self.update_colorbar()
            plt.draw()

    def update_colormap(self, colormap):
        """Update colormap to *colormap*."""
        from matplotlib import pyplot as plt

        self.colormap = colormap

        if self.mpl_image is not None:
            self.mpl_image.set_cmap(self.colormap)
        if "cmap" in self.imshow_kwargs:
            self.imshow_kwargs["cmap"] = self.colormap
        if self.colorbar is not None:
            self.colorbar.set_cmap(self.colormap)
            self.colorbar.draw_all()
        plt.draw()

    def update_all(self, image):
        """Update image and colorbar."""
        from matplotlib import pyplot as plt

        self.mpl_image.set_data(image)
        if self.clim is None:
            # If the limit is not set to a value we autoscale
            new_lower = float(image.min())
            new_upper = float(image.max())
            if self.limits_changed(new_lower, new_upper):
                self.mpl_image.set_clim(new_lower, new_upper)
                self.update_colorbar()
                plt.draw()

    def update_colorbar(self):
        """Update the colorbar (rescale and redraw)."""
        self.colorbar.set_clim(self.mpl_image.get_clim())
        self.colorbar.draw_all()

    def make_colorbar(self):
        """Make colorbar according to the current colormap."""
        from matplotlib import pyplot as plt

        colormap = None
        if self.colormap is not None:
            colormap = self.colormap
        elif "cmap" in self.imshow_kwargs:
            colormap = self.imshow_kwargs["cmap"]

        self.colorbar = plt.colorbar(cmap=colormap)

    def make_image(self, image):
        """Create an image with colorbar"""
        from matplotlib import pyplot as plt
        self.mpl_image = plt.imshow(image, **self.imshow_kwargs)
        if self.clim is not None:
            self.mpl_image.set_clim(self.clim)
        if self.has_colorbar:
            if self.colorbar is None:
                self.make_colorbar()
            else:
                self.update_colorbar()
        plt.draw()

    def limits_changed(self, lower, upper):
        """
        Determine whether the colormap limits changed enough for colorbar
        redrawing.
        """
        new_range = upper - lower
        lower_ratio = np.abs(lower - self.mpl_image.get_clim()[0]) / new_range
        upper_ratio = np.abs(upper - self.mpl_image.get_clim()[1]) / new_range

        return lower_ratio > 0.1 or upper_ratio > 0.1
