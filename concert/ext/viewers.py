"""Opening images in external programs."""
import collections
import atexit
import os
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
from concert.async import threaded
from concert.quantities import q
from concert.storage import write_tiff
from concert.coroutines.base import coroutine


LOG = logging.getLogger(__name__)
_PYPLOT_VIEWERS = []


def _terminate_pyplotviewers():
    """Terminate all :py:class:`PyplotViewerBase` isntances."""
    for viewer in _PYPLOT_VIEWERS:
        if viewer is not None:
            viewer.terminate()


# Register termination of all the PyplotViewerBase instances also on
# normal exit
atexit.register(_terminate_pyplotviewers)


def _start_command(program, image, writer=write_tiff):
    """
    Create a tmp file for dumping the *image* and use *program*
    to open that image. Use *writer* for writing the iamge to the disk.
    """
    tmp_file = tempfile.mkstemp()[1]
    try:
        full_path = writer(tmp_file, image)
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


class PyplotViewerBase(object):

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

    .. py:attribute:: blit

        True if faster redrawing based on canvas blitting should be used.
    """

    def __init__(self, view_function, blit=False):
        self._queue = MultiprocessingQueue()
        self._paused = False
        self._terminated = False
        self._coroutine = None
        self._proc = None
        # The udater is implementation-specific and must be provided by
        # the subclass by calling self._set_updater
        self._updater = None
        self.view_function = view_function
        self._blit = blit

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
        :py:attr:`PyplotViewerBase.view_function`.
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
            # The underscore must stay, otherwise it doesn't work, magic?
            _ = FuncAnimation(figure, self._updater.process, interval=5,
                              blit=self._blit)
            plt.show()

        except KeyboardInterrupt:
            plt.close("all")


class PyplotViewer(PyplotViewerBase):

    """
    Dynamic plot viewer using matplotlib.

    .. py:attribute:: style

        One of matplotlib's linestyle format strings

    .. py:attribute:: plt_kwargs

        Keyword arguments accepted by matplotlib's plot()

    .. py:attribute:: autoscale

        If True, the axes limits will be expanded as needed by the new data,
        otherwise the user needs to rescale the axes

    """

    def __init__(self, style="o", plot_kwargs=None, autoscale=True, title=""):
        super(PyplotViewer, self).__init__(self.plot)
        self._autoscale = autoscale
        self._style = style
        self._iteration = 0
        self._set_updater(_PyplotUpdater(self._queue, style,
                                         plot_kwargs, autoscale,
                                         title=title))

    def plot(self, x, y=None, force=False):
        """
        Plot *x* and *y*, if *y* is None and *x* is a scalar the real y is
        given by *x* and x is the current iteration of the plotting command,
        if *x* is an iterable then it is interpreted as y data array and x is
        a span [0, len(x)]. If both *x* and *y* are given, they are plotted as
        they are. If *force* is True the plotting is guaranteed, otherwise it
        might be skipped for the sake of plotting speed.

        Note: if x is not given, the iteration starts at 0.
        """
        if not self._paused and (self._queue.empty() or force):
            if y is None:
                if isinstance(x, q.Quantity) and isinstance(x.magnitude, collections.Iterable) or\
                   not isinstance(x, q.Quantity) and isinstance(x, collections.Iterable):
                    x_data = np.arange(len(x))
                    y_data = x
                else:
                    x_data = self._iteration
                    y_data = x
                    self._iteration += 1
            else:
                x_data = x
                y_data = y
            self._queue.put((_PyplotUpdater.PLOT, (x_data, y_data)))

    @property
    def style(self):
        return self._style

    @style.setter
    def style(self, style):
        """Set line style to *style*."""
        self._style = style
        self._queue.put((_PyplotUpdater.STYLE, style))

    def clear(self):
        """Clear the plotted data."""
        self._iteration = 0
        self._queue.put((_PyplotUpdater.CLEAR, None))

    @property
    def autoscale(self):
        return self._autoscale

    @autoscale.setter
    def autoscale(self, autoscale):
        """Set *autoscale* on the axes, can be True or False."""
        self._autoscale = autoscale
        self._queue.put((_PyplotUpdater.AUTOSCALE, autoscale))


class PyplotImageViewer(PyplotViewerBase):

    """Dynamic image viewer using matplotlib."""

    def __init__(self, imshow_kwargs=None, colorbar=True, title=""):
        super(PyplotImageViewer, self).__init__(self.show, blit=True)
        self._has_colorbar = colorbar
        self._imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        self._make_imshow_defaults()
        self._set_updater(_PyplotImageUpdater(self._queue,
                                              self._imshow_kwargs,
                                              self._has_colorbar,
                                              title=title))

    def show(self, item, force=False):
        """
        show *item* into the redrawing queue. The item is truly inserted
        only if the queue is empty in order to guarantee that the newest
        image is drawn or if the *force* is True.
        """
        if not self._paused and (self._queue.empty() or force):
            self._queue.put((_PyplotImageUpdater.IMAGE, item))

    @property
    def limits(self):
        raise NotImplementedError

    @limits.setter
    def limits(self, clim):
        """
        Update the colormap limits by *clim*, which is a (lower, upper)
        tuple.
        """
        self._queue.put((_PyplotImageUpdater.CLIM, clim))

    @property
    def colormap(self):
        return self._colormap

    @colormap.setter
    def colormap(self, colormap):
        """Set colormp of the shown image to *colormap*."""
        self._queue.put((_PyplotImageUpdater.COLORMAP, colormap))

    def _make_imshow_defaults(self):
        """Override matplotlib's image showing defafults."""
        from matplotlib import cm

        if "cmap" not in self._imshow_kwargs:
            self._imshow_kwargs["cmap"] = cm.gray
            self._colormap = cm.gray
        else:
            self._colormap = self._imshow_kwargs["cmap"]
        if "interpolation" not in self._imshow_kwargs:
            self._imshow_kwargs["interpolation"] = "nearest"


class _PyplotUpdaterBase(object):

    """
    Base class for animating a matploblib figure in a separate process.

    .. py:attribute:: queue

        A multiprocessing queue for receiving commands
    """

    def __init__(self, queue, title=""):
        self.queue = queue
        self.first = True
        self.title = title
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
        except Exception as exc:
            LOG.error("Pyplot Exception: {}".format(exc))
        finally:
            return self.get_artists()

    def get_artists(self):
        """
        Abstract function for getting all matplotlib artists which we want
        to redraw. Needs to be implemented by the subclass.
        """
        raise NotImplementedError


class _PyplotUpdater(_PyplotUpdaterBase):

    """
    Private class for updating a matplotlib figure with a 1D data stream.
    The arguments are the same as by :py:class:`PyplotViewer`.
    """
    CLEAR = "clear"
    PLOT = "plot"
    STYLE = "style"
    AUTOSCALE = "autoscale"

    def __init__(self, queue, style="o", plot_kwargs=None, autoscale=True,
                 title=""):
        super(_PyplotUpdater, self).__init__(queue, title=title)
        self.data = [[], []]
        self.line = None
        self.style = style
        self.plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        if 'color' not in self.plot_kwargs:
            # Matplotlib changes colors all the time by default
            self.plot_kwargs['color'] = 'b'
        self.autoscale = autoscale
        self.commands = {_PyplotUpdater.PLOT: self.plot,
                         _PyplotUpdater.STYLE: self.change_style,
                         _PyplotUpdater.CLEAR: self.clear,
                         _PyplotUpdater.AUTOSCALE: self.set_autoscale}

    def get_artists(self):
        """Get the artists for the drawing."""
        return [] if self.line is None else [self.line]

    def make_line(self):
        """Create the line based on current settings."""
        from matplotlib import pyplot as plt

        self.line = plt.plot(self.data[0], self.data[1], self.style,
                             **self.plot_kwargs)[0]
        self.line.axes.set_title(self.title)

    def clear(self, data):
        """Clear everything from the plot."""
        from matplotlib import pyplot as plt

        self.data = [[], []]
        if self.line is not None:
            self.line.axes.clear()
        self.make_line()

        plt.draw()

    def plot(self, data):
        """Plot *data*, which is an (x, y) tuple."""
        from matplotlib import pyplot as plt

        def get_magnitude_and_unit(value):
            if hasattr(value, "magnitude"):
                dimless = value.magnitude
                unit = value.units
            else:
                dimless = value
                unit = None

            return dimless, unit

        x_item, x_units = get_magnitude_and_unit(data[0])
        y_item, y_units = get_magnitude_and_unit(data[1])

        first = len(self.data[0]) == 0
        is_iterable = isinstance(x_item, collections.Iterable)
        if is_iterable:
            self.data = [list(x_item), list(y_item)]
        else:
            self.data[0].append(x_item)
            self.data[1].append(y_item)

        if first:
            self.make_line()
            if not is_iterable:
                # Make sure the limits are set properly for the first time
                self.line.axes.set_xlim(self.data[0][0] - 1e-7, self.data[0][0] + 1e-7)
                self.line.axes.set_ylim(self.data[1][0] - 1e-7, self.data[1][0] + 1e-7)
            if x_units is not None:
                self.line.axes.get_xaxis().set_label_text(str(x_units))
            if y_units is not None:
                self.line.axes.get_yaxis().set_label_text(str(y_units))

        self.line.set_data(*self.data)

        if self.autoscale or first:
            # Draw for the first time or on limit change to update ticks and
            # labels
            self.autoscale_view()
        plt.draw()

    def change_style(self, style):
        """Change line style to *style*."""
        self.style = style
        if self.line is not None:
            # Just redrawing doesn't work
            self.line.axes.clear()
            self.make_line()

    def set_autoscale(self, autoscale):
        """If *autoscale* is True, the plit is rescaled when needed."""
        from matplotlib import pyplot as plt

        self.autoscale = autoscale
        if self.autoscale:
            self.autoscale_view()
            plt.draw()

    def autoscale_view(self):
        """Autoscale axes limits."""
        if self.line is not None:
            self.line.axes.relim()
            # For some reason the relim itself doesn't work, so we set the
            # limits to the new values explicitly
            if len(self.data[0]) > 1:
                self.line.axes.set_xlim(min(self.data[0]), max(self.data[0]))
                self.line.axes.set_ylim(min(self.data[1]), max(self.data[1]))
            self.line.axes.autoscale_view()


class _PyplotImageUpdater(_PyplotUpdaterBase):

    """
    Private class for updating a matplotlib figure with an image stream.
    """

    IMAGE = "image"
    CLIM = "clim"
    COLORMAP = "colormap"

    def __init__(self, queue, imshow_kwargs, has_colorbar, title=""):
        super(_PyplotImageUpdater, self).__init__(queue, title=title)
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
        self.mpl_image.axes.set_title(self.title)
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
        if lower >= upper:
            return False
        new_range = upper - lower
        lower_ratio = np.abs(lower - self.mpl_image.get_clim()[0]) / new_range
        upper_ratio = np.abs(upper - self.mpl_image.get_clim()[1]) / new_range

        return lower_ratio > 0.1 or upper_ratio > 0.1
