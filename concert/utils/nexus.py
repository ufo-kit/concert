from nexpy.api import nexus as nx
from concert.base import Parameter
from concert.devices import cameras
from concert.asynchronous import async


def get_detector(camera):
    """Return an nx.NXdetector instance of *camera*."""
    instrument = nx.NXdetector()
    return instrument


def get_instrument(device):
    """Tries to return a corresponding nx.NXinstrument for *device* or
    *None*."""
    type_map = {cameras.base.Camera: get_detector}

    for t, convert in type_map.items():
        if isinstance(device, t):
            return convert(device)

    return None


@async
def get_scan_result(scanner):
    """Create an nx.NXdata tuple from a scan.

    Runs *scanner* and turns the result into two nx.SDSs and returns an
    nx.NXdata tuple from both SDSs.
    """

    def get_unit_string(quantity):
        return str(quantity.dimensionality)

    x_raw, y_raw = scanner.run().result()
    x_nx = nx.SDS(x_raw,
                  name=scanner.param.name,
                  units=get_unit_string(scanner.minimum))

    if isinstance(scanner.feedback, Parameter):
        y_nx = nx.SDS(y_raw,
                      name=scanner.feedback.name,
                      units=get_unit_string(scanner.feedback.unit))
    else:
        y_nx = nx.SDS(y_raw)

    return nx.NXdata(y_nx, [x_nx])
