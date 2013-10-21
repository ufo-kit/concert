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
        self._stopped = False
        self._make_imshow_defaults()
        self._terminated = False
        self._proc = Process(target=self._run)
        self._proc.start()
        _PYPLOT_VIEWERS.append(self)

    @coroutine
    def __call__(self, size=None):
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
        if self._stopped:
            raise ValueError("Cannot add images to a stopped viewer")

        if self._queue.empty() or force:
            self._queue.put(item)

    def stop(self):
        """Stop, no more images will be displayed from now on."""
        if not self._stopped:
            LOG.debug("Stopping viewer")
            self._queue.put(None)
            self._stopped = True

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
            _ = FuncAnimation(figure, updater.update, interval=5,
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
    def __init__(self, queue, imshow_kwargs, has_colorbar):
        self.queue = queue
        self.imshow_kwargs = imshow_kwargs
        self.has_colorbar = has_colorbar
        self.mpl_image = None
        self.shape = None
        self.lower = None
        self.upper = None
        self.colorbar = None
        self.first = True

    def update(self, iteration):
        """
        Update function which is going to be called by matplotlib's
        FuncAnimation instance.
        """
        try:
            if self.first:
                # Wait as much time as it takes for the first
                # time beacuse we don't want to show a window
                # with no image in it.
                image = self.queue.get()
                self.first = False
            else:
                image = self.queue.get(timeout=0.1)
            if image is not None:
                if self.shape is not None and self.shape != image.shape:
                    # When the shape changes the axes needs to be reset
                    self.mpl_image.axes.clear()
                    self.mpl_image = None

                if self.mpl_image is None:
                    # Either removed by shape change or first time drawing
                    self.make_image(image)
                else:
                    self.update_all(image)
                # Remember the shape because when it changes we need to
                # recreate the matplotlib image.
                self.shape = image.shape
        except Empty:
            pass
        finally:
            retval = [] if self.mpl_image is None else [self.mpl_image]
            return retval

    def update_all(self, image):
        """Update image and colorbar."""
        self.mpl_image.set_data(image)
        new_lower = float(image.min())
        new_upper = float(image.max())
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

    def make_image(self, image):
        """Create an image with colorbar"""
        from matplotlib import pyplot as plt
        self.mpl_image = plt.imshow(image, **self.imshow_kwargs)
        if self.has_colorbar and self.colorbar is None:
            self.colorbar = plt.colorbar()
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
