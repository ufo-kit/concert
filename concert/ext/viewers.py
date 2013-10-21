"""Opening images in external programs."""
import os
import tempfile
try:
    from Queue import Empty
except ImportError:
    from queue import Empty
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


class PyplotViewer(object):

    """Image viewer which updates the plot in a separate process."""

    def __init__(self, imshow_kwargs=None, colorbar=True):
        self._has_colorbar = colorbar
        self._imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        self._queue = MultiprocessingQueue()
        self._stopped = False
        self._make_imshow_defaults()
        self._proc = Process(target=self._run)
        self._proc.start()

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
                self.show(image)

                if size is not None and i == size - 1:
                    # Maximum number of images has come, end redrawing
                    self.show(image, force=True)
                    self.stop()

            i += 1

    def terminate(self):
        """Close all communication and terminate child process."""
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
        import numpy as np
        from matplotlib import pyplot as plt
        from matplotlib.animation import FuncAnimation
        try:
            figure = plt.figure()

            def update_image(iteration):
                try:
                    image = self._queue.get(timeout=0.1)
                    if image is not None:
                        if update_image.shape is not None and \
                                update_image.shape != image.shape:
                            # When the shape changes the axes needs to be reset
                            update_image.mpl_image.axes.clear()
                            update_image.mpl_image = None

                        if update_image.mpl_image is None:
                            update_image.mpl_image = \
                                plt.imshow(image,
                                           **self._imshow_kwargs)
                            if self._has_colorbar and update_image.colorbar \
                                    is None:
                                update_image.colorbar = plt.colorbar()
                            plt.draw()
                        else:
                            update_image.mpl_image.set_data(image)
                            new_lower = image.min()
                            new_upper = image.max()
                            if update_image.lower is None:
                                update_image.lower = new_lower
                                update_image.upper = new_upper
                            else:
                                range = new_upper - new_lower
                                lower_ratio = \
                                    np.abs(new_lower -
                                           update_image.lower) / range
                                upper_ratio = \
                                    np.abs(new_upper -
                                           update_image.upper) / range
                                # If the lower or upper bound changed more
                                # than 10 % of the whole range we update the
                                # colorbar.
                                if lower_ratio > 0.1 or upper_ratio > 0.1:
                                    update_image.lower = new_lower
                                    update_image.upper = new_upper
                                    # We need to redraw in order to update
                                    # the ticklabels
                                    update_image.colorbar.set_clim(new_lower,
                                                                   new_upper)
                                    update_image.colorbar.draw_all()
                                    plt.draw()
                        update_image.shape = image.shape
                except Empty:
                    pass
                finally:
                    return update_image.mpl_image,

            update_image.mpl_image = None
            update_image.shape = None
            update_image.lower = None
            update_image.upper = None
            update_image.colorbar = None
            _ = FuncAnimation(figure, update_image, interval=5,
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
