"""Opening images in external programs."""
import asyncio
import collections
import multiprocessing as mp
import time
import logging
from queue import Empty
from typing import Callable
import numpy as np
from concert.base import Parameterizable, Parameter
from concert.coroutines.base import background, run_in_executor
from concert.quantities import q


LOG = logging.getLogger(__name__)
_MP_CTX = mp.get_context('spawn')


def _start_command(program, image):
    """
    Create a tmp file for dumping the *image* and use *program*
    to open that image. Use *writer* for writing the iamge to the disk.
    """
    import os
    import tempfile
    from subprocess import Popen
    from concert.storage import write_tiff

    tmp_file = tempfile.mkstemp()[1]
    try:
        full_path = write_tiff(tmp_file, image)
        with Popen([program, full_path]) as process:
            process.wait()
    finally:
        os.remove(full_path)


def imagej(image, path="imagej"):
    """
    imagej(image, path="imagej")

    Open *image* in ImageJ found by *path*.
    """
    # Do not make a daemon process to make sure the interpreter waits for it to exit, i.e. the
    # *finally* will be called and temp file deleted.
    proc = _MP_CTX.Process(target=_start_command, args=(path, image), daemon=False)
    proc.start()


class ViewerBase(Parameterizable):

    """
    A base class for data viewer which sends commands to a backend-specific updater which runs in a
    separate process.
    """

    force = Parameter(help='Make sure every item is displayed')

    async def __ainit__(self, force: bool = False):
        await super().__ainit__()
        self._force = force
        self._queue = _MP_CTX.Queue()
        # This prevents hanging in the case we exit the session after something is put in the queue
        # and before it is consumed.
        self._queue.cancel_join_thread()
        self._paused = False
        # To be set up by an actual implementation which runs the drawing backed in a separate
        # process
        self._proc = None
        # __del__ is not going to help because it's never called from our concert session

    async def _set_force(self, value):
        self._force = value

    async def _get_force(self):
        return self._force

    @background
    async def __call__(self, producer, size=0, force=None):
        """
        Display stream from *producer*. If *size* is specified, stop after displaying *size* items.
        If *force* is True make sure the item is displayed, if False it may be skipped if there is
        something in the queue waiting to be shown, if it is None, the viewer's *force* parameter is
        used.
        """
        i = 0

        async for item in producer:
            if not size or i < size:
                await self.show(item, force=self._force if force is None else force)

            i += 1

    @background
    async def show(self, item, force=False):
        """Push *item* to the queue for display in a separate proces. If *force* is True make sure
        the item is displayed, otherwise it may be skipped if there is something in the queue
        waiting to be shown.
        """
        # If the circumstances allow it, push the item to the queue for display
        # This must happen before instantiation of the updater below because _show may raise
        # exception, in which case we don't want the updater to have started yet.
        if not self._paused and (not self._queue.qsize() or force or not self._proc):
            await run_in_executor(self._show, item)

        # If there is no updater or it has been stopped, instantiate it and start it in a process.
        if not (self._proc and self._proc.is_alive()):
            updater = self._make_updater()
            self._proc = _MP_CTX.Process(target=updater.run, daemon=True)
            self._proc.start()
            # Make sure that all control commands, like changing colormap, have been processed
            while self._queue.qsize():
                await asyncio.sleep(0.01)

    def pause(self):
        """Pause, no images are dispayed but image commands work."""
        self._paused = True

    def resume(self):
        """Resume the viewer."""
        self._paused = False

    def _show(self, item):
        """Implementation of pushing *item* to the display queue."""
        raise NotImplementedError

    def _make_updater(self):
        """Updater factory method."""
        raise NotImplementedError


class PyplotViewer(ViewerBase):

    """
    Dynamic plot viewer using matplotlib.

    .. py:attribute:: style

        One of matplotlib's linestyle format strings

    .. py:attribute:: plt_kwargs

        Keyword arguments accepted by matplotlib's plot()

    .. py:attribute:: autoscale

        If True, the axes limits will be expanded as needed by the new data,
        otherwise the user needs to rescale the axes

    .. py:attribute:: title

        Plot title
    """

    style = Parameter(help='Line style')
    autoscale = Parameter(help='Autoscale view')

    async def __ainit__(self, style: str = "o", plot_kwargs: dict = None, autoscale: bool = True,
                        title: str = "", force: bool = False):
        await super().__ainit__(force=force)
        self._autoscale = autoscale
        self._style = style
        self._plot_kwargs = plot_kwargs
        self._title = title

    def _show(self, item):
        """Unravel the *item* for x and y so that it is plotted correctly."""
        try:
            if len(item) != 2:
                raise ValueError('Plotting accepts only (x, y) pairs')
        except TypeError as exc:
            raise ValueError('Plotting accepts only (x, y) pairs') from exc

        if isinstance(item, q.Quantity):
            item = item.magnitude
        first, second = item
        if isinstance(first, q.Quantity):
            first = first.magnitude
        if isinstance(second, q.Quantity):
            second = second.magnitude
        item = (first, second)

        self._queue.put(('plot', item))

    def _make_updater(self):
        return _PyplotUpdater(self._queue, self._style, self._plot_kwargs,
                              self._autoscale, title=self._title)

    def reset(self):
        """Clear the plotted data."""
        self._queue.put(('clear', None))

    async def _get_style(self):
        return self._style

    async def _set_style(self, style):
        """Set line style to *style*."""
        self._queue.put(('style', style))
        self._style = style

    async def _get_autoscale(self):
        return self._autoscale

    async def _set_autoscale(self, autoscale):
        """Set *autoscale* on the axes, can be True or False."""
        self._queue.put(('autoscale', autoscale))
        self._autoscale = autoscale


class ImageViewerBase(ViewerBase):

    """Backend-free base class for displaying images.

    .. py:attribute:: limits

        minimum and maximum gray value (black and white points). Can be a tuple (min, max), 'auto'
        or 'stream'. When 'auto', limits are adjusted for every shown image, when 'stream', limits
        are adjusted on every __call__.

    .. py:attribute:: downsampling

        Display every n-th pixel, which can speed up the viewer

    .. py:attribute:: title

        Image title

    .. py:attribute:: show_refresh_rate

        Whether or not to show refresh rate text directly embedded into the displayed image
    """

    show_refresh_rate = Parameter(help='Display current refresh rate')
    limits = Parameter(help='Black and white point')
    downsampling = Parameter(help='Display only every n-th pixel')

    async def __ainit__(self, limits: str = 'stream', downsampling: int = 1, title: str = "",
                        show_refresh_rate: bool = False, force: bool = False):
        await super().__ainit__(force=force)
        self._show_refresh_rate = show_refresh_rate
        self._title = title
        self._downsampling = downsampling
        self._limits = limits

    @background
    async def __call__(self, producer: Callable, size: int = None, force: bool = None):
        # In case limits are set to 'stream' we need to reset clim
        self._queue.put(('clim', self._limits))
        return await super().__call__(producer, size=None, force=force)

    def _show(self, item):
        self._queue.put(('image', item[::self._downsampling, ::self._downsampling]))

    async def _get_downsampling(self):
        return self._downsampling

    async def _set_downsampling(self, value):
        if value not in range(1, 21):
            raise ValueError('Downsampling must be from interval [1, 20]')
        self._downsampling = value

    async def _get_limits(self):
        return self._limits

    async def _set_limits(self, limits):
        """
        Minimum and maximum gray value (black and white points). Can be a tuple (min, max), 'auto'
        or 'stream'. When 'auto', limits are adjusted for every shown image, when 'stream', limits
        are adjusted on every __call__.
        """
        if not (limits == 'auto' or limits == 'stream' or len(limits) == 2):
            raise ViewerError("limits can be a tuple (min, max), 'auto' or 'stream'")
        self._queue.put(('clim', limits))
        self._limits = limits

    async def _get_show_refresh_rate(self):
        return self._show_refresh_rate

    async def _set_show_refresh_rate(self, value):
        if not isinstance(value, bool):
            raise ValueError('boolean value expected')
        self._queue.put(('show-fps', value))
        self._show_refresh_rate = value


class PyQtGraphViewer(ImageViewerBase):

    """Dynamic image viewer using PyQtGraph."""

    def _make_updater(self):
        return _PyQtGraphUpdater(self._queue, limits=self._limits, title=self._title,
                                 show_refresh_rate=self._show_refresh_rate)


class PyplotImageViewer(ImageViewerBase):

    """Dynamic image viewer using matplotlib.

    .. py:attribute:: imshow_kwargs

        matplotlib's imshow keyword arguments

    .. py:attribute:: fast

        Whether to use the fast version without colorbar

    """

    colormap = Parameter(help='Colormap')

    async def __ainit__(self, imshow_kwargs: dict = None, fast: bool = True, limits: str = 'stream',
                        downsampling: int = 1, title: str = "", show_refresh_rate: bool = False,
                        force: bool = False):
        await super().__ainit__(limits=limits, downsampling=downsampling, title=title,
                                show_refresh_rate=show_refresh_rate, force=force)
        self._has_colorbar = not fast
        self._imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        self._make_imshow_defaults()

    def _make_updater(self):
        if self._has_colorbar:
            return _PyplotImageUpdater(
                self._queue, self._imshow_kwargs, limits=self._limits, title=self._title,
                show_refresh_rate=self._show_refresh_rate
            )
        return _SimplePyplotImageUpdater(
            self._queue, self._imshow_kwargs, limits=self._limits, title=self._title,
            show_refresh_rate=self._show_refresh_rate
        )

    def reset(self):
        """Reset the viewer's state."""
        self._queue.put(('reset', None))

    def _make_imshow_defaults(self):
        """Override matplotlib's image showing defafults."""
        if "cmap" not in self._imshow_kwargs:
            self._imshow_kwargs["cmap"] = 'gray'
        self._colormap = self._imshow_kwargs["cmap"]
        if "interpolation" not in self._imshow_kwargs:
            self._imshow_kwargs["interpolation"] = "nearest"

    async def _get_colormap(self):
        return self._colormap

    async def _set_colormap(self, colormap):
        """Set colormp of the shown image to *colormap*."""
        self._queue.put(('colormap', colormap))


class _PyQtGraphUpdater:

    """Fast PyQtGraph-based image viewing backend."""

    def __init__(self, queue: mp.Queue, limits: str = 'stream', title: str = "",
                 show_refresh_rate: bool = False, force: bool = False):
        self.queue = queue
        self.title = title
        self.clim = limits
        self.show_refresh_rate = show_refresh_rate
        self.text = None
        self.plot = None
        # main graphics window
        self.view = None
        self.last_text_time = time.perf_counter()
        self.last_time = time.perf_counter()
        self.commands = {'image': self.proces_image,
                         'clim': self.update_limits,
                         'show-fps': self.toggle_show_refresh_rate}

    def process(self):
        """Process commands from queue."""
        try:
            cmd, item = self.queue.get(timeout=0.01)
            self.commands[cmd](item)
        except Empty:
            pass

    def update_all(self, image):
        """Display *image*."""
        now = time.perf_counter()
        self.view.imageItem.setImage(image, autoLevels=self.clim == 'auto', autoDownsample=True)
        if self.clim == 'stream':
            self.clim = (image.min(), image.max())
            self.sync_image_and_clim()

        if self.show_refresh_rate:
            if now - self.last_text_time > 0.5:
                fps = 1 / (now - self.last_time)
                self.text.setPos(image.shape[1] / 2, 0)
                self.text.setText(f'{fps:5.1f} FPS')
                self.last_text_time = now

        if not self.view.isVisible():
            self.view.show()

        self.last_time = now

    def _pg_mouse_moved(self, ev):
        if self.view.imageItem.sceneBoundingRect().contains(ev):
            image = self.view.imageItem.image
            pos = self.view.imageItem.mapFromScene(ev)
            x = int(pos.x() + 0.5)
            y = int(pos.y() + 0.5)
            if y < image.shape[0] and x < image.shape[1]:
                self.view.view.setTitle(
                    f'x={x} y={y} [{self.view.imageItem.image[y, x]:g}]',
                    bold=True
                )
        else:
            self.view.view.setTitle('')

    def proces_image(self, image):
        """Process current *image* including window setup if it is a first image."""
        import pyqtgraph as pg
        first = False

        if not self.view:
            first = True
            self.plot = pg.PlotItem(title=self.title)
            self.view = pg.ImageView(view=self.plot)
            self.view.imageItem.scene().sigMouseMoved.connect(self._pg_mouse_moved)
            self.make_refresh_rate_text()

        self.update_all(image)

        if first:
            self.update_limits(self.clim)

    def update_limits(self, clim):
        """Update limits (black and white point)."""
        # Store no matter if there is an image already or not
        self.clim = clim

        if not self.view or self.clim == 'stream':
            # No image has been displayed yet or the limits will be set with the next image
            return

        if clim == 'auto':
            # Synchronize histogram range and current image range drawn as lines
            self.view.imageItem.setImage(self.view.imageItem.image, autoLevels=True,
                                         autoDownsample=True)
            # Adjust histogram range
            self.view.ui.histogram.autoHistogramRange()
            return

        self.sync_image_and_clim()

    def sync_image_and_clim(self):
        self.view.imageItem.setLevels(self.clim)
        # Synchronize histogram range and current image range drawn as lines
        self.view.imageItem.setImage(self.view.imageItem.image, autoLevels=False,
                                     autoDownsample=True)
        self.view.ui.histogram.setHistogramRange(*self.clim)

    def toggle_show_refresh_rate(self, value):
        """Show refresh rate or not controlled by *value*."""
        self.show_refresh_rate = value

        if not self.view:
            return

        if value:
            self.make_refresh_rate_text()
        else:
            if self.text:
                self.view.removeItem(self.text)
            self.text = None

    def make_refresh_rate_text(self):
        """Make text for showing the refresh rate."""
        import pyqtgraph as pg

        if self.text or not self.show_refresh_rate:
            return

        self.text = pg.TextItem(anchor=(0.5, 0), color=(220, 220, 220), fill=(0, 0, 0))
        self.text.setOpacity(0.5)
        if self.view:
            self.view.addItem(self.text)

    def run(self):
        """Start drawing."""
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore
        import signal

        signal.signal(signal.SIGINT, signal.SIG_IGN)

        app = pg.Qt.mkQApp()
        # row-major for not transposing the array
        pg.setConfigOptions(antialias=True, imageAxisOrder='row-major')

        timer = QtCore.QTimer()
        timer.timeout.connect(self.process)
        # 0 ms delay -> be as fast as possible
        timer.start(0)

        app.exec_()


class _PyplotUpdaterBase:

    """
    Base class for animating a matploblib figure in a separate process.

    .. py:attribute:: queue

        A multiprocessing queue for receiving commands
    """

    def __init__(self, queue: mp.Queue, title: str = ""):
        self.queue = queue
        self.first = True
        self.title = title
        # A dictionary in form command: method which tells the class what to do
        # for every received command
        self.commands = {}

    def process(self, iteration):
        """Get item from queue and process it."""
        try:
            if self.first:
                # Wait as much time as it takes for the first
                # time beacuse we don't want to show a window
                # with no image in it.
                item = self.queue.get()
                self.first = False
            else:
                item = self.queue.get(timeout=0.01)
            cmd, data = item
            self.commands[cmd](data)
            return self.get_artists()
        except Empty:
            self.on_empty()
            return self.get_artists()

    def on_empty(self):
        """Callback on queue timeout."""

    def get_artists(self):
        """
        Abstract function for getting all matplotlib artists which we want
        to redraw. Needs to be implemented by the subclass.
        """
        raise NotImplementedError

    def run(self):
        """
        Run the process, i.e. wait for an image to come to the queue
        and dispaly it. This method is executed in a separate process.
        """
        # This import *must* be here, otherwise it doesn't work on Linux
        from matplotlib import pyplot as plt
        from matplotlib.animation import FuncAnimation
        # KeyboardInterrupt is ignored in order not to keep hanging somewhere...
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        figure = plt.figure()
        animation = FuncAnimation(figure, self.process, interval=200, blit=False)
        plt.show()


class _PyplotUpdater(_PyplotUpdaterBase):

    """Plotting backend. The arguments are the same as by :py:class:`PyplotViewer`."""

    def __init__(self, queue: mp.Queue, style: str = "o", plot_kwargs: dict = None,
                 autoscale: bool = True, title: str = ""):
        super().__init__(queue, title=title)
        self.data = [[], []]
        self.line = None
        self.style = style
        self.plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        if 'color' not in self.plot_kwargs:
            # Matplotlib changes colors all the time by default
            self.plot_kwargs['color'] = 'b'
        self.autoscale = autoscale
        self.commands = {'plot': self.plot,
                         'style': self.change_style,
                         'clear': self.clear,
                         'autoscale': self.set_autoscale}

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
        self.data = [[], []]
        if self.line is not None:
            self.line.axes.clear()
        self.make_line()

    def plot(self, data):
        """Plot *data*, which is an (x, y) tuple."""
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

    def change_style(self, style):
        """Change line style to *style*."""
        self.style = style
        if self.line is not None:
            # Just redrawing doesn't work
            self.line.axes.clear()
            self.make_line()

    def set_autoscale(self, autoscale):
        """If *autoscale* is True, the plit is rescaled when needed."""
        self.autoscale = autoscale
        if self.autoscale:
            self.autoscale_view()

    def autoscale_view(self):
        """Autoscale axes limits."""
        if self.line is not None:
            self.line.axes.relim()
            # For some reason the relim itself doesn't work, so we set the
            # limits to the new values explicitly
            if len(self.data[0]) > 1:
                self.line.axes.set_xlim(min(self.data[0]) - 1e-7, max(self.data[0]) + 1e-7)
                self.line.axes.set_ylim(min(self.data[1]) - 1e-7, max(self.data[1]) + 1e-7)
            self.line.axes.autoscale_view()


class _PyplotImageUpdaterBase(_PyplotUpdaterBase):

    """Common class for various image viewing backends."""

    def __init__(self, queue: mp.Queue, imshow_kwargs: dict, limits: str = 'stream',
                 title: str = "", show_refresh_rate: bool = False):
        super().__init__(queue, title=title)
        self.imshow_kwargs = imshow_kwargs
        self.mpl_image = None
        self.clim = limits
        self.text = None
        self.show_refresh_rate = show_refresh_rate
        self.last_text_time = time.perf_counter()
        self.last_time = time.perf_counter()
        self.commands = {'image': self.process_image,
                         'clim': self.update_limits,
                         'colormap': self.update_colormap,
                         'reset': self.reset,
                         'show-fps': self.toggle_show_refresh_rate}

    def process_image(self, image):
        """Display *image*."""
        if self.mpl_image is not None and self.mpl_image.get_size() != image.shape:
            self.reset()

        if self.mpl_image:
            # Either removed by shape change or first time drawing
            self.update_all(image)
        else:
            self.make_image(image)

    def make_image(self, image):
        """Setup everything and display *image* for the first time."""
        raise NotImplementedError

    def update_all(self, image):
        """Update everything which needs to be updated when new *image* arrives."""
        self.mpl_image.set_data(image)
        self.update_refresh_rate_text()
        if self.clim in ['auto', 'stream']:
            # If the limit is not set to a value we autoscale
            new_lower = float(image.min())
            new_upper = float(image.max())
            if self.limits_changed(new_lower, new_upper):
                self.mpl_image.set_clim(new_lower, new_upper)
            if self.clim == 'stream':
                self.clim = self.mpl_image.get_clim()

    def update_limits(self, clim):
        """Update limits (black and white point)."""
        # Store no matter if there is an image already or not
        self.clim = clim

        if self.mpl_image is None or clim == 'stream':
            return

        if clim == 'auto':
            image = self.mpl_image.get_array()
            clim = (image.min(), image.max())

        self.mpl_image.set_clim(clim)

    def update_colormap(self, colormap):
        """Update colormap."""
        raise NotImplementedError

    def reset(self, *args):
        """Reset state."""
        if self.mpl_image:
            self.mpl_image.remove()
            self.mpl_image = None
        if self.text:
            self.text.remove()
            self.text = None

    def toggle_show_refresh_rate(self, value):
        """Show refresh rate or not controlled by *value*."""
        self.show_refresh_rate = value

        if self.show_refresh_rate:
            if self.mpl_image:
                self.make_refresh_rate_text()
        else:
            if self.text:
                self.text.remove()
                self.text = None

    def get_artists(self):
        """Needed by matplotlib's FuncAnimation."""
        artists = []
        if self.mpl_image:
            artists.append(self.mpl_image)
        if self.text:
            artists.append(self.text)

        return artists

    def update_refresh_rate_text(self):
        """Update refresh rate text."""
        if not (self.show_refresh_rate and self.text):
            return

        current = time.perf_counter()
        if current - self.last_text_time > 0.5:
            # Don't update FPS text that often to be readable
            fps = 1 / (current - self.last_time)
            self.text.set_text(f'{fps:5.1f} FPS')
            self.last_text_time = current
        self.last_time = current

    def make_refresh_rate_text(self, animated=False):
        """Make text for showing the refresh rate."""
        import matplotlib.pyplot as plt

        if self.text or not self.show_refresh_rate:
            return

        self.text = plt.text(0.5, 0.975, '', alpha=0.5, animated=animated,
                             ha='center', va='center', backgroundcolor='black',
                             color='lightgray', transform=self.mpl_image.axes.transAxes)
        self.text.get_bbox_patch().set_alpha(0.5)

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


class _PyplotImageUpdater(_PyplotImageUpdaterBase):

    """Image viewing backend."""

    def __init__(self, queue: mp.Queue, imshow_kwargs: dict, limits: str = 'stream',
                 title: str = "", show_refresh_rate: bool = False):
        super().__init__(queue, imshow_kwargs, limits=limits, title=title,
                         show_refresh_rate=show_refresh_rate)
        self.colorbar = None

    def get_artists(self):
        """Get artists to return for matplotlib's animation."""
        artists = super().get_artists()
        if self.colorbar:
            artists.append(self.colorbar)

        return artists

    def update_limits(self, clim):
        super().update_limits(clim)
        if self.mpl_image:
            self.update_colorbar()

    def update_colormap(self, colormap):
        """Update colormap to *colormap*."""
        if self.mpl_image is not None:
            self.mpl_image.set_cmap(colormap)

        # Save for later
        self.imshow_kwargs["cmap"] = colormap

    def update_all(self, image):
        """Update image and colorbar."""
        super().update_all(image)
        self.update_colorbar()

    def reset(self, *args):
        # We need to get rid of the colorbar as well
        self.mpl_image.axes.figure.clear()
        self.mpl_image = None
        self.colorbar = None

    def update_colorbar(self):
        """Update the colorbar (rescale and redraw)."""
        shape = self.mpl_image.get_size()
        if (shape[1] > shape[0] and self.colorbar.orientation == 'vertical'
                or shape[0] >= shape[1] and self.colorbar.orientation == 'horizontal'):
            self.colorbar.remove()
            self.make_colorbar()

    def make_colorbar(self):
        """Make colorbar according to the current colormap."""
        from matplotlib import pyplot as plt

        colormap = self.imshow_kwargs.get("cmap")
        shape = self.mpl_image.get_size()
        orientation = 'horizontal' if shape[1] > shape[0] else 'vertical'
        self.colorbar = plt.colorbar(cmap=colormap, orientation=orientation)

    def make_image(self, image):
        """Create an image with colorbar"""
        from matplotlib import pyplot as plt
        self.mpl_image = plt.imshow(image, **self.imshow_kwargs)
        self.mpl_image.axes.set_title(self.title)
        self.make_refresh_rate_text(animated=False)
        if self.colorbar is None:
            self.make_colorbar()
        self.update_limits(self.clim)


class _SimplePyplotImageUpdater(_PyplotImageUpdaterBase):

    """Simple image viewing backend optimized for speed, no colorbar available."""

    def __init__(self, queue: mp.Queue, imshow_kwargs: dict, limits: str = 'stream',
                 title: str = "", show_refresh_rate: bool = False):
        super().__init__(queue, imshow_kwargs, limits=limits, title=title,
                         show_refresh_rate=show_refresh_rate)
        self.fig = None
        self.axes = None
        self.background = None
        self.closed = True

    def run(self):
        """There is no animation, we handle everything ourselves so just keep getting images from
        the queue.
        """
        # KeyboardInterrupt is ignored in order not to keep hanging somewhere...
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        while True:
            self.process(None)

    def on_empty(self):
        """Let the image window be responsive and run the GUI event loop for a while.

        Note:
        plt.pause makes the window steal focus aggresively (mpl 3.1.2) and after minimizing it pops
        right back up. This keeps happening until user manipulates the image via the toolbar. Doing
        things manually via draw_idle and starting the event loop seems to do the trick, for now...
        """
        # plt.pause(0.01)
        if self.fig:
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(0.01)

    def on_close(self, event):
        """Called when the window is closed. Note it and re-open on next image."""
        self.closed = True

    def on_draw(self, event):
        """Called when size is changed and we need to copy the new bbox."""
        if event is not None:
            if event.canvas != self.fig.canvas:
                raise RuntimeError

        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        if self.mpl_image:
            self.axes.draw_artist(self.mpl_image)
        if self.text:
            self.axes.draw_artist(self.text)

    def make_image(self, image):
        """Create a new matplotlib image."""
        import matplotlib.pyplot as plt

        self.mpl_image = self.axes.imshow(image, animated=True, **self.imshow_kwargs)
        if self.clim not in ['auto', 'stream']:
            self.mpl_image.set_clim(self.clim)
        self.make_refresh_rate_text(animated=True)
        # This makes sure axes BBox fits the newly displayed image
        self.axes.relim()
        # In case we were panning/zooming, this will reset the position
        self.axes.autoscale()
        # This will reset toolbar buttons
        self.axes.figure.canvas.toolbar.update()
        plt.show(block=False)
        plt.pause(0.1)
        self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def update_limits(self, clim):
        super().update_limits(clim)
        if self.mpl_image:
            self.redraw()

    def update_colormap(self, colormap):
        """Update colormap to *colormap*."""
        if self.mpl_image is not None:
            self.mpl_image.set_cmap(colormap)
            self.redraw()

        # Save for after reset
        self.imshow_kwargs["cmap"] = colormap

    def update_all(self, image):
        self.fig.canvas.restore_region(self.background)
        super().update_all(image)
        self.redraw()

    def redraw(self):
        """Redraw scene."""
        self.axes.draw_artist(self.mpl_image)
        if self.text:
            self.axes.draw_artist(self.text)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def process_image(self, image):
        import matplotlib.pyplot as plt

        if self.closed:
            # Start from scratch after close or first invocation
            self.fig, self.axes = plt.subplots()
            self.mpl_image = None
            self.axes.set_title(self.title)
            self.fig.canvas.mpl_connect('draw_event', self.on_draw)
            self.fig.canvas.mpl_connect('close_event', self.on_close)
            self.closed = False

        super().process_image(image)


class ViewerError(Exception):
    """Viewer errors."""
