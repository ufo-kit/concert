"""
NeXus file format related utitilities.

This module provides convenience utilities to create NeXus-compatible data
sets. It uses NeXpy_ to interface with NeXus.

.. _NeXpy: http://wiki.nexusformat.org/NeXpy
"""
import numpy as np
from concert.base import Quantity
from concert.devices import cameras, monochromators
from concert.async import async

try:
    from nexpy.api import nexus as nx
except ImportError:
    print("NeXpy or libnexus are not installed")


def get_detector(_camera):
    """Return an nx.NXdetector instance of *camera*."""
    return nx.NXdetector()


def get_monochromator(monochromator):
    """Return an nx.NXmonochromator instance of *monochromator*."""
    return nx.NXmonochromator(energy=monochromator.energy,
                              wavelength=monochromator.wavelength)


def get_instrument(device):
    """Tries to return a corresponding nx.NXinstrument for *device* or
    *None*."""
    type_map = {
        cameras.base.Camera: get_detector,
        monochromators.base.Monochromator: get_monochromator
    }

    for tval, convert in type_map.items():
        if isinstance(device, tval):
            return convert(device)

    return None


def quantity_unit(quantity):
    """Return string representation of *quantity*'s unit."""
    return str(quantity.dimensionality)


def add_quantity(nx_object, name, quantity):
    """Add *quantity* value as *name* to *nx_object*.

    The magnitude of *quantity* is used as a value and the string
    representation of the dimensionality of *quantity* is added to the field as
    *units* ::

        sample = nx.NXdetector()
        add_quantity(sample, 'x_pixel_size', 5 * q.micrometer)

        assert sample.x_pixel_size == 5
        assert sample.x_pixel_size.units = 'um'
    """
    setattr(nx_object, name, quantity.magnitude)
    attr = getattr(nx_object, name)
    setattr(attr, 'units', quantity_unit(quantity))


@async
def get_scan_result(scanner):
    """Create an nx.NXdata tuple from a scan.

    Runs *scanner* and turns the result into two nx.SDSs and returns an
    nx.NXdata tuple from both SDSs.
    """
    x_raw, y_raw = scanner.run().result()
    x_nx = nx.SDS(x_raw,
                  name=scanner.param.name,
                  units=quantity_unit(scanner.minimum))

    if isinstance(scanner.feedback, Quantity):
        y_nx = nx.SDS(y_raw,
                      name=scanner.feedback.name,
                      units=quantity_unit(scanner.feedback.unit))
    else:
        y_nx = nx.SDS(y_raw)

    return nx.NXdata(y_nx, [x_nx])


@async
def get_tomo_scan_result(tomo_scanner, title='Tomo scan'):
    """Create and return a NXtomophase data set from the scan results of
    *tomo_scanner*."""
    future = tomo_scanner.run()
    detector = tomo_scanner.camera

    root = nx.NXentry(title=title)

    n_projections = tomo_scanner.num_projections
    step = tomo_scanner.angle.magnitude
    angles = np.arange(n_projections * step, step=step)
    zeros = np.zeros(n_projections)
    sample = nx.NXsample(rotation_angle=angles,
                         x_translation=zeros,
                         y_translation=zeros,
                         z_translation=zeros)

    sample.rotation_angle.units = quantity_unit(tomo_scanner.angle)
    sample.x_translation.units = "m"
    sample.y_translation.units = "m"
    sample.z_translation.units = "m"

    root.sample = sample
    root.instrument = nx.NXinstrument()

    darks, flats, projections = future.result()

    dark_field = nx.NXdetector(sequence_number=len(darks),
                               data=darks)
    bright_field = nx.NXdetector(sequence_number=len(flats),
                                 data=flats)
    sample = nx.NXdetector(sequence_number=len(projections),
                           data=projections)

    root.instrument.dark_field = dark_field
    root.instrument.bright_field = bright_field
    root.instrument.sample = sample

    add_quantity(sample, 'x_pixel_size', detector.sensor_pixel_width)
    add_quantity(sample, 'y_pixel_size', detector.sensor_pixel_height)

    counts = [np.sum(frame) for frame in darks]
    counts.extend([np.sum(frame) for frame in flats])
    counts.extend([np.sum(frame) for frame in projections])

    control = nx.NXmonitor(integral=counts)
    root.control = control

    return root
