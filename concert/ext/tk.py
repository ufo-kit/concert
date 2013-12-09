"""Tk Widget."""
import Queue
import Tkinter as tk
from concert.async import dispatcher


class DeviceWidget(tk.Toplevel):

    """Wrap a device in a top level Tk widget.

    *device* must be an object derived from :class:`.Device`. *timeout*
    specifies how much time in milliseconds passes between to parameter
    queries.
    """

    def __init__(self, device, timeout=1500):
        tk.Toplevel.__init__(self, padx=6, pady=6)

        self.title(device.__class__.__name__)
        self.grid()
        self.queues = {}
        self.values = {}
        self.timeout = timeout

        grid_opts = {'padx': 3, 'pady': 3}

        for i, param in enumerate(device):
            name_label = tk.Label(self, text=param.name)
            name_label.grid(row=i, column=0, **grid_opts)

            if param.is_readable():
                value = tk.DoubleVar()
                value.set(param.get().result())

                state = tk.NORMAL if param.is_writable() else 'readonly'
                entry = tk.Entry(self, textvariable=value, state=state)
                entry.grid(row=i, column=1, **grid_opts)

                self.queues[param] = Queue.Queue()
                self.values[param] = value

                dispatcher.subscribe(param, 'changed', self._update_value)
                self.after(self.timeout, self._poll, param)

            if param.unit:
                unit_name = param.unit.dimensionality.string
                unit_label = tk.Label(self, text=unit_name)
                unit_label.grid(row=i, column=2, **grid_opts)

    def _update_value(self, param):
        self.queues[param].put(param.get().result())

    def _poll(self, param):
        try:
            result = self.queues[param].get(False)
            self.values[param].set(result)
        except Queue.Empty:
            pass

        self.after(self.timeout, self._poll, param)
