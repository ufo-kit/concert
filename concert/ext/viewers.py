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

    """Image viewer which updates the plot in a separate process."""

    def __init__(self, imshow_kwargs=None, colorbar=True):
        self._has_colorbar = colorbar
        self._imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        self._queue = MultiprocessingQueue()
        self._paused = False
        self._make_imshow_defaults()
        self._terminated = False
        self._coroutine = None
        self._proc = Process(target=self._run)
        self._proc.start()
        _PYPLOT_VIEWERS.append(self)

    def __call__(self, size=None):
        """
        Display a dynamic image in a separate thread in order not to
        stall program execution. If *size* is specified, the redrawing stops
        when *size* images come.
        """
        if self._coroutine is None:
            self._coroutine = self._updater(size=size)

        return self._coroutine

    @coroutine
    def _updater(self, size=None):
        """
        Display a dynamic image in a separate thread in order not to
        stall program execution. If *size* is specified, the redrawing stops
        when *size* images come.
        """
        i = 0

        while True:
            image = yield
            if size is None or i < size:
                if size is not None and i == size - 1:
                    # Maximum number of images has come, end redrawing
                    self.show(image, force=True)
                    self.stop()
                else:
                    self.show(image)
                # This helps with the smoothness of drawing
                time.sleep(0.001)

            i += 1

    def terminate(self):
        """Close all communication and terminate child process."""
        if not self._terminated:
            self._queue.close()
            self._proc.terminate()

    def show(self, item, force=False):
        """
        show *item* into the redrawing queue. The item is truly inserted
        only if the queue is empty in order to guarantee that the newest
        image is drawn or if the *force* is True.
        """
        if not self._paused and (self._queue.empty() or force):
            self._queue.put((_PyplotUpdater.IMAGE, item))

    def pause(self):
        """Pause, no images are dispayed but image commands work."""
        self._paused = True

    def resume(self):
        """Resume the viewer."""
        self._paused = False

    def set_limits(self, clim):
        """
        Update the colormap limits by *clim*, which is a (lower, upper)
        tuple.
        """
        self._queue.put((_PyplotUpdater.CLIM, clim))

    def set_colormap(self, colormap):
        """Set colormp of the shown image to *colormap*."""
        self._queue.put((_PyplotUpdater.COLORMAP, colormap))

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
            updater = _PyplotUpdater(self._queue, self._imshow_kwargs,
                                     self._has_colorbar)
            _ = FuncAnimation(figure, updater.process, interval=5,
                              blit=True)
            plt.show()

        except KeyboardInterrupt:
            plt.close("all")

    def _make_imshow_defaults(self):
        """Override matplotlib's image showing defafults."""
        from matplotlib import cm

        if "cmap" not in self._imshow_kwargs:
            self._imshow_kwargs["cmap"] = cm.gray
        if "interpolation" not in self._imshow_kwargs:
            self._imshow_kwargs["interpolation"] = "nearest"


class _PyplotUpdater(object):

    """
    Private class for updating a matplotlib figure with an image stream.
    """

    IMAGE = "image"
    CLIM = "clim"
    COLORMAP = "colormap"

    def __init__(self, queue, imshow_kwargs, has_colorbar):
        self.queue = queue
        self.imshow_kwargs = imshow_kwargs
        self.has_colorbar = has_colorbar
        self.mpl_image = None
        self.lower = None
        self.upper = None
        self.colorbar = None
        self.first = True
        self.image = None
        self.clim = None
        self.image = None
        self.colormap = None
        self.commands = {_PyplotUpdater.IMAGE: self.process_image,
                         _PyplotUpdater.CLIM: self.update_limits,
                         _PyplotUpdater.COLORMAP: self.update_colormap}

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
            retval = [] if self.mpl_image is None else [self.mpl_image]
            return retval

    def process_image(self, image):
        """Process the incoming *image*."""
        if self.image is not None and self.image.shape != image.shape:
            # When the shape changes the axes needs to be reset
            self.mpl_image.axes.clear()
            self.mpl_image = None

        self.image = image

        if self.mpl_image is None:
            # Either removed by shape change or first time drawing
            self.make_image()
        else:
            self.update_all()

    def update_limits(self, clim):
        """
        Update current colormap limits by *clim*, which is a tuple
        (lower, upper). If *clim* is None, the limit is reset to the span of
        the current image.
        """
        self.clim = clim

        if clim is None and self.image is not None:
            # Restore the full clim
            clim = self.image.min(), self.image.max()

        if self.mpl_image is not None and clim is not None:
            self.mpl_image.set_clim(clim)

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

    def update_all(self):
        """Update image and colorbar."""
        self.mpl_image.set_data(self.image)
        if self.clim is None:
            new_lower = float(self.image.min())
            new_upper = float(self.image.max())
        else:
            new_lower = self.clim[0]
            new_upper = self.clim[1]

        if self.lower is None:
            self.lower = new_lower
            self.upper = new_upper
        elif self.limits_changed(new_lower, new_upper):
            self.lower = new_lower
            self.upper = new_upper
            self.update_colorbar()

    def update_colorbar(self):
        """Update the colorbar (rescale and redraw)."""
        from matplotlib import pyplot as plt
        lower, upper = self.mpl_image.get_clim()
        self.colorbar.set_clim(lower, upper)
        self.colorbar.draw_all()
        plt.draw()

    def make_colorbar(self):
        """Make colorbar according to the current colormap."""
        from matplotlib import pyplot as plt

        colormap = None
        if self.colormap is not None:
            colormap = self.colormap
        elif "cmap" in self.imshow_kwargs:
            colormap = self.imshow_kwargs["cmap"]

        self.colorbar = plt.colorbar(cmap=colormap)

    def make_image(self):
        """Create an image with colorbar"""
        from matplotlib import pyplot as plt
        self.mpl_image = plt.imshow(self.image, **self.imshow_kwargs)
        if self.clim is not None:
            self.mpl_image.set_clim(self.clim)
        if self.has_colorbar and self.colorbar is None:
            self.make_colorbar()
        plt.draw()

    def limits_changed(self, lower, upper):
        """
        Determine whether the colormap limits changed enough for colorbar
        redrawing.
        """
        new_range = upper - lower
        lower_ratio = np.abs(lower - self.lower) / new_range
        upper_ratio = np.abs(upper - self.upper) / new_range

        return lower_ratio > 0.1 or upper_ratio > 0.1
