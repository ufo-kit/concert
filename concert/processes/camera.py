import numpy as np


def noise_scan(camera, param_name, minimum, maximum, n_intervals=50):
    """Calculate the noise level depending on variable parameter.

    *camera* is a camera object that supports :meth:`.grab`, *param_name*
    is a parameter name of *camera* that should be varied according to
    *minimum*, *maximum* and *n_intervals*.

    Returns a tuple (thresholds, noises, min_threshold, dark_frame)."""
    thresholds = np.linspace(minimum, maximum, n_intervals)
    noises = np.zeros(thresholds.shape)
    dark_frame = None
    min_threshold = 0
    min_noise = np.finfo('d').max

    for i, x in enumerate(thresholds):
        camera[param_name].set(x).wait()
        frame = camera.grab()

        # Should this be over whole frame or just those pixels above 0?
        noise = np.mean(frame)
        noises[i] = noise

        if noise < min_noise:
            dark_frame = np.array(frame)
            min_noise = noise
            min_threshold = x

    return (thresholds, noises, min_threshold, dark_frame)


def photon_transfer(camera, dark_frame, minimum, maximum, n_intervals):
    """Calculate the photon transfer according to procedure described by M.
    Caselle and F. Beckmann.

    *camera* is a camera object that supports the :meth:`.grab`, *dark_frame*
    is an array-like with the same dimensions as the camera frames. *minimum*,
    *maximum* and *n_intervals* describe the scannable range.

    Returns a tuple *(n_photons, corrected_adcs)*
    """
    num_photons = np.log(np.linspace(minimum, maximum, n_intervals))
    corrected_adcs = np.zeros(num_photons.shape)

    for i, x in enumerate(num_photons):
        frame = camera.grab()
        corrected_adcs[i] = np.log(np.sum(np.abs(frame - dark_frame)))

    return (num_photons, corrected_adcs)
