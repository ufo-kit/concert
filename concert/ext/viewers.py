"""Opening images in external programs."""
import atexit
import os
import tempfile
from multiprocessing import Queue as MultiprocessingQueue, Process
from subprocess import Popen
import logbook
from concert.helpers import coroutine, threaded
from concert.storage import write_tiff


LOG = logbook.Logger(__name__)


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


class PyplotViewer(Process):

    """Image viewer which updates the plot in a separate process."""

    def __init__(self, imshow_kwargs=None, colorbar=True):
        super(PyplotViewer, self).__init__()
        self._has_colorbar = colorbar
        self._imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        self._queue = MultiprocessingQueue()
        self._stopped = False
        self._make_imshow_defaults()
        # Make sure the window doesn't hang when the parent process exits
        atexit.register(self.stop)

    @coroutine
    def __call__(self, size=None):
        """
        Display a dynamic image in a separate thread in order not to
        stall program execution. If *size* is specified, the redrawing stops
        when *size* images come.
        """
        i = 0
        if not self.is_alive():
            self.start()

        while True:
            image = yield
            if size is None or i < size:
                self.show(image)

                if size is not None and i == size - 1:
                    # Maximum number of images has come, end redrawing
                    self.show(image, force=True)
                    self.stop()

            i += 1

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

        if not self.is_alive():
            self.start()

    def stop(self):
        """Stop, no more images will be displayed from now on."""
        if not self._stopped:
            LOG.debug("Stopping viewer")
            self._queue.put(None)
            self._stopped = True

    def run(self):
        """
        Run the process, i.e. wait for an image to come to the queue
        and dispaly it.
        """
        # This import *must* be here, otherwise it doesn't work on Linux
        from matplotlib import pyplot as plt
        plt.ion()

        mpl_image = None
        shape = None

        while True:
            image = self._queue.get()

            if image is None:
                break

            if shape is not None and shape != image.shape:
                # When the shape changes the axes needs to be reset
                mpl_image.axes.clear()
                mpl_image = None

            if mpl_image is None:
                mpl_image = plt.imshow(image, **self._imshow_kwargs)
                if self._has_colorbar and shape is None:
                    plt.colorbar()
            else:
                mpl_image.set_data(image)
                # Rescale the image and colorbar (updates the figure)
                mpl_image.set_clim(image.min(), image.max())

            plt.draw()
            shape = image.shape

        plt.ioff()
        plt.show()

    def _make_imshow_defaults(self):
        """Override matplotlib's image showing defafults."""
        from matplotlib import cm

        if "cmap" not in self._imshow_kwargs:
            self._imshow_kwargs["cmap"] = cm.gray
        if "interpolation" not in self._imshow_kwargs:
            self._imshow_kwargs["interpolation"] = "nearest"
