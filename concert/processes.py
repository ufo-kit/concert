import numpy as np
import logging
from scipy.ndimage.filters import gaussian_filter
from concert.async import async, wait
from concert.quantities import q
from concert.measures import rotation_axis
from concert.optimization import halver, optimize_parameter
from concert.imageprocessing import center_of_mass, flat_correct
from concert.coroutines.base import coroutine
from concert.helpers import expects, Numeric
from concert.devices.motors.base import LinearMotor, ContinuousLinearMotor
from concert.devices.motors.base import RotationMotor, ContinuousRotationMotor
from concert.devices.cameras.base import Camera


LOG = logging.getLogger(__name__)


def _pull_first(tuple_list):
        for tup in tuple_list:
            yield tup[0]


def scan(param, feedback, minimum=None, maximum=None, intervals=64, convert=lambda x: x):
    """
    Scan the parameter object in *intervals* steps between *minimum* and
    *maximum* and call *feedback* at each step. If *minimum* or *maximum* is
    ``None``, :attr:`.ParameterValue.lower` or :attr:`.ParameterValue.upper` is
    used.

    Set *convert* to a callable that transforms the parameter value prior to
    setting it.

    Generates futures which resolve to tuples containing the set and feedback
    values *(x, y)*.
    """
    if minimum is None:
        minimum = param.lower
    if maximum is None:
        maximum = param.upper

    xss = np.linspace(minimum, maximum, intervals)

    @async
    def get_value(x, previous):
        if previous:
            previous.join()

        param.set(convert(x)).join()
        return (x, feedback())

    future = None

    for xval in xss:
        future = get_value(xval, future)
        yield future


def scan_param_feedback(scan_param, feedback_param, minimum=None, maximum=None, intervals=64,
                        convert=lambda x: x):
    """
    Convenience function to scan one parameter and measure another.

    Scan the *scan_param* object and measure *feedback_param* at each of the
    *intervals* steps between *minimum* and *maximum*.

    Returns a tuple *(x, y)* with scanned parameter and measured values.
    """
    def feedback():
        return feedback_param.get().result()

    return scan(scan_param, feedback, minimum, maximum, intervals, convert)


def ascan(param_list, n_intervals, handler, initial_values=None):
    """
    For each of the *n_intervals* and for each of the *(parameter, start,
    stop)* tuples in *param_list*, calculate a set value from *(stop - start) /
    n_intervals* and set *parameter* to it::

        ascan([(motor['position'], 0 * q.mm, 2 * q.mm)], 5, handler)

    When all devices have reached the set point *handler* is called with a list
    of the parameters as its first argument.

    If *initial_values* is given, it must be a list with the same length as
    *devices* containing start values from where each device is scanned.
    """
    parameters = [param for param in _pull_first(param_list)]

    if initial_values:
        if len(param_list) != len(initial_values):
            raise ValueError("*initial_values* must match *parameter_list*")
    else:
        initial_values = []

        for param in parameters:
            if param.unit:
                initial_values.append(0 * param.unit)
            else:
                initial_values.append(0)

    initialized_params = list(map(lambda x: x[0] + (x[1],),
                                  zip(param_list, initial_values)))

    for i in range(n_intervals + 1):
        futures = []

        for param, start, stop, init in initialized_params:
            step = (stop - start) / n_intervals
            value = init + start + i * step
            futures.append(param.set(value))

        wait(futures)
        handler(parameters)


def dscan(parameter_list, n_intervals, handler):
    """
    For each of the *n_intervals* and for each of the *(parameter, start,
    stop)* tuples in *param_list*, calculate a set value from *(stop - start) /
    n_intervals* and set *parameter*.
    """
    initial_values = [param.get().result()
                      for param in _pull_first(parameter_list)]

    return ascan(parameter_list, n_intervals, handler, initial_values)


@expects(Camera, LinearMotor, measure=np.std, opt_kwargs=None,
         plot_consumer=None, frame_consumer=None, output=Numeric(1))
def focus(camera, motor, measure=np.std, opt_kwargs=None,
          plot_consumer=None, frame_consumer=None):
    """
    Focus *camera* by moving *motor*. *measure* is a callable that computes a
    scalar that has to be maximized from an image taken with *camera*.
    *opt_kwargs* are keyword arguments sent to the optimization algorithm.
    *plot_consumer* is fed with y values from the optimization and
    *frame_consumer* is fed with the incoming frames.

    This function is returning a future encapsulating the focusing event. Note,
    that the camera is stopped from recording as soon as the optimal position
    is found.
    """
    if opt_kwargs is None:
        opt_kwargs = {'initial_step': 0.5 * q.mm,
                      'epsilon': 1e-2 * q.mm}

    def get_measure():
        camera.trigger()
        frame = camera.grab()
        if frame_consumer:
            frame_consumer.send(frame)
        return - measure(frame)

    @coroutine
    def filter_optimization():
        """
        Filter the optimization's (x, y) subresults to only the y part,
        otherwise the the plot update is not lucid.
        """
        while True:
            tup = yield
            if plot_consumer:
                plot_consumer.send(tup[1])

    camera.trigger_mode = camera.trigger_modes.SOFTWARE
    camera.start_recording()
    f = optimize_parameter(motor['position'], get_measure, motor.position,
                           halver, alg_kwargs=opt_kwargs,
                           consumer=filter_optimization())
    f.add_done_callback(lambda unused: camera.stop_recording())
    return f


@async
@expects(Camera, RotationMotor, x_motor=RotationMotor, z_motor=RotationMotor,
         measure=rotation_axis, num_frames=Numeric(1), absolute_eps=Numeric(1, q.deg),
         max_iterations=Numeric(1), flat=None, dark=None,
         frame_consumer=None, output=Numeric(1))
def align_rotation_axis(camera, rotation_motor, x_motor=None, z_motor=None,
                        measure=rotation_axis, num_frames=10, absolute_eps=0.1 * q.deg,
                        max_iterations=5, flat=None, dark=None, frame_consumer=None):
    """
    align_rotation_axis(camera, rotation_motor, x_motor=None, z_motor=None,
    measure=rotation_axis, num_frames=10, absolute_eps=0.1 * q.deg, max_iterations=5,
    flat=None, dark=None, frame_consumer=None)

    Align rotation axis. *camera* is used to obtain frames, *rotation_motor*
    rotates the sample around the tomographic axis of rotation, *x_motor*
    turns the sample around x-axis, *z_motor* turns the sample around z-axis.
    *measure* provides axis of rotation angular misalignment data (a callable),
    *num_frames* defines how many frames are acquired and passed to the *measure*.
    *absolute_eps* is the threshold for stopping the procedure. If *max_iterations*
    is reached the procedure stops as well. *flat* and *dark* are the normalization
    frames applied on the acquired frames. *frame_consumer* is a coroutine which will
    receive the frames acquired at different sample positions.

    The procedure finishes when it finds the minimum angle between an
    ellipse extracted from the sample movement and respective axes or the
    found angle drops below *absolute_eps*. The axis of rotation after
    the procedure is (0,1,0), which is the direction perpendicular
    to the beam direction and the lateral direction.
    """
    if not x_motor and not z_motor:
        raise ValueError("At least one of the x, z motors must be given")

    def get_frames():
        frames = []
        for i in range(num_frames):
            rotation_motor.move(step).join()
            camera.trigger()
            frame = camera.grab().astype(np.float)
            if flat is not None:
                frame = flat_correct(frame, flat, dark=dark)
            if frame_consumer:
                frame_consumer.send(frame)
            frames.append(frame)

        return frames

    step = 2 * np.pi / num_frames * q.rad
    x_angle, z_angle, center = None, None, None

    if camera.state == 'recording':
        camera.stop_recording()
    camera['trigger_mode'].stash().join()
    camera.trigger_mode = camera.trigger_modes.SOFTWARE
    camera.start_recording()

    try:
        # Sometimes both z-directions need to be tried out because of the
        # projection ambiguity.
        z_direction = -1
        i = 0

        x_last = None
        z_last = None
        z_turn_counter = 0

        while True:
            x_angle, z_angle, center = measure(get_frames())

            x_better = True if z_motor is not None and\
                (x_last is None or np.abs(x_angle) < x_last) else False
            z_better = True if x_motor is not None and\
                (z_last is None or np.abs(z_angle) < z_last) else False

            if x_motor:
                if z_better:
                    z_turn_counter = 0
                elif z_turn_counter < 1:
                    # We might have rotated in the opposite direction because
                    # of the projection ambiguity. However, that must be picked up
                    # in the next iteration, so if the two consequent angles
                    # are worse than the minimum, we have the result.
                    z_better = True
                    z_direction = -z_direction
                    z_turn_counter += 1

            x_future, z_future = None, None
            if z_better and np.abs(z_angle) >= absolute_eps:
                x_future = x_motor.move(z_direction * z_angle)
            if x_better and np.abs(x_angle) >= absolute_eps:
                z_future = z_motor.move(x_angle)
            elif (np.abs(z_angle) < absolute_eps or not z_better):
                # The newly calculated angles are worse than the previous
                # ones or the absolute threshold has been reached,
                # stop iteration.
                break

            wait([future for future in [x_future, z_future] if future is not None])

            x_last = np.abs(x_angle)
            z_last = np.abs(z_angle)

            i += 1

            if i == max_iterations:
                # If we reached maximum iterations we consider it a failure
                # because the algorithm was not able to get to the desired
                # solution within the max_iterations limit.
                raise ProcessException("Maximum iterations reached")
    finally:
        camera.stop_recording()
        # No side effects
        camera['trigger_mode'].restore().join()

    # Return the last known ellipse fit
    return x_angle, z_angle, center


@expects(Camera, LinearMotor, LinearMotor, Numeric(2, q.um), Numeric(2, q.mm), Numeric(2, q.mm),
         xstep=Numeric(1), zstep=Numeric(1), thres=Numeric(1), output=Numeric(1))
def find_beam(cam, xmotor, zmotor, pixelsize, xborder, zborder,
              xstep=None, zstep=None, thres=1000):
    """
    Scans the area defined by xborder and zborder for the beam until
    beam_visible returns True.
    Startpoint is the current motor-position if this position is inside the
    defined area else it start from the center of that area.
    It searches in a spiral around the startpoint.

    *cam* is the camera-device, *xmotor* the motor-device horizontally aligned
    to the image and *zmotor* the motor-device vertically aligned to the image.
    *pixelsize* determines the realworld size of an image pixels (scalar or
    2-element array-like, e.g. [4*q.um, 5*q.um]). *xborder* and *zborder*
    define the search area. Each constructed with a start- and an end-value
    (e.g. [-1.2*q.mm, 5.5*q.mm]).

    Optional arguments *xstep* and *zstep* define the length of one movement
    in the specific direction. Defaults are calculated from cam_img.shape and
    *pixelsize*.
    Optional argument *thres* will be past to beam_visible().
    """

    def beam_visible(img, thres):
        """
        Simple grayvalue-threshold that defines the beam to be visible in *img*.
        """
        return (img >= thres).any()

    def check(rel_move):
        """Move to new position and check if the beam is visible."""
        fut_0 = zmotor.move(rel_move[0]*q.um)
        fut_1 = xmotor.move(rel_move[1]*q.um)
        wait([fut_0, fut_1])
        cam.trigger()
        img = gaussian_filter(cam.grab().astype(np.float32), 40.0)
        img_shape[0] = img.shape
        bv = beam_visible(img, thres)
        return bv

    def spiral_scan():
        """
        Generator for returning rel. movement to the next position of the
        scanpath.
        """

        def trydir(dr):
            """
            Check if position in direction *dr* is unchecked or outside the
            search-area.
            Result:   2 - ok; 1 - outside; 0 - inside but already used
            """
            tmp = sp_pos + dir2rpos[dr]
            inside = (tmp >= 0).all() and (tmp < sp_shape).all()
            unused = (not scanned_points[tmp[0], tmp[1]]) if inside else False
            return 1 if not inside else 2 if unused else 0

        def next_move(pos, dr, rel_move=True):
            """
            Moves relativ in the direction *dr* (or to position *dr* if
            rel_move==False) and marks new position as scanned.
            Clips movement at border of the scanarea if needed.
            """
            old = np.array(pos).copy()
            if rel_move:
                pos += dir2rpos[dr]
            else:
                pos[0], pos[1] = dr[0], dr[1]
            scanned_points[pos[0], pos[1]] = True
            if (old == 0).any() or (old == sp_shape-1).any() or \
                    (pos == 0).any() or (pos == sp_shape-1).any():
                new_pos = np.array([zpos, xpos]) + step * (pos-[nz0, nx0])
                clip = np.append((new_pos < [zborder[0], xborder[0]]),
                                 (new_pos > [zborder[1], xborder[1]]))
                new_pos[0] = zborder[1] if clip[2] else \
                    zborder[0] if clip[0] else new_pos[0]
                new_pos[1] = xborder[1] if clip[3] else \
                    xborder[0] if clip[1] else new_pos[1]
                old_pos = np.array([zmotor.position.to(q.um.units).magnitude,
                                    xmotor.position.to(q.um.units).magnitude])
                return new_pos - old_pos
            else:
                if rel_move:
                    return step * dir2rpos[dr]
                else:
                    return step * (pos - old)

        def decide_dir():
            """
            Picks a direction to go and updates direction of search-rotation.
            """
            dr = np.array(map(trydir, range(4))) == 2
            r = np.arange(4)[dr][0]
            w0 = sp_pos-[nz0, nx0]
            w1 = w0 + dir2rpos[r]
            w = np.arctan2(w1[0], w1[1]) - np.arctan2(w0[0], w0[1])
            search_rot[0] = int((w % (2*np.pi)) < np.pi)
            return r

        xpos = xmotor.position.to(q.um.units).magnitude
        zpos = zmotor.position.to(q.um.units).magnitude
        search_rot = [0]
        rdir2try = [[1, 0, 3], [3, 0, 1]]
        dir2rpos = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        step = img_shape[0] * \
            np.array([x.to(q.um.units).magnitude for x in ps])
        step[0] = step[0] if zstep is None else zstep.to(q.um.units).magnitude
        step[1] = step[1] if xstep is None else xstep.to(q.um.units).magnitude
        nz0 = int((zpos-zborder[0])/step[0]) + \
            ((zpos-zborder[0]) % step[0] > step[0]/2.)
        nz1 = int((zborder[1]-zpos)/step[0]) + \
            ((zborder[1]-zpos) % step[0] > step[0]/2.)
        nx0 = int((xpos-xborder[0])/step[1]) + \
            ((xpos-xborder[0]) % step[1] > step[1]/2.)
        nx1 = int((xborder[1]-xpos)/step[1]) + \
            ((xborder[1]-xpos) % step[1] > step[1]/2.)
        sp_shape = np.array([nz0+nz1+1, nx0+nx1+1])
        scanned_points = np.zeros(sp_shape, np.bool_)
        sp_pos = np.array([nz0, nx0])
        scanned_points[sp_pos[0], sp_pos[1]] = True

        while not scanned_points.all():
            skip = False
            try:
                sdir = decide_dir()
                yield next_move(sp_pos, sdir)
            except IndexError:
                skip = True

            while (not scanned_points.all()) and (not skip):
                sr = search_rot[0]
                for rd in rdir2try[sr]:
                    rd = (sdir + rd) % 4
                    c = trydir(rd)
                    if c == 2:
                        break
                    elif c == 1:
                        search_rot[0] = (search_rot[0]+1) % 2
                if c == 2:
                    sdir = rd
                    yield next_move(sp_pos, sdir)
                    continue
                break

            if not scanned_points.all():
                ind = np.indices(sp_shape)
                mask = np.logical_not(scanned_points)
                ind = np.array([ind[0][mask], ind[1][mask]])
                dist = ((ind-np.array([nz0, nx0]).reshape(2, 1))**2).sum(0)
                sort_pos = ind[:, np.argsort(dist)]
                n_pos = sort_pos[:, 0]
                r = next_move(sp_pos, n_pos, False)
                yield r

    units = (u for u in pixelsize.units if pixelsize.units[u] > 0.0)
    units = q.parse_expression(' * '.join(units))
    ps = np.tile(pixelsize.magnitude, 2)[:2] * units
    xborder = np.array([xborder[0].to(q.um.units).magnitude,
                        xborder[1].to(q.um.units).magnitude])
    zborder = np.array([zborder[0].to(q.um.units).magnitude,
                        zborder[1].to(q.um.units).magnitude])

    # startpoint
    img_shape = [0]
    xpos = xmotor.position.to(q.um.units).magnitude
    zpos = zmotor.position.to(q.um.units).magnitude
    outside = (xpos < xborder[0]) or (xpos > xborder[1]) or \
        (zpos < zborder[0]) or (zpos > zborder[1])
    rmov = (zborder.mean() - zpos, xborder.mean() - xpos) if outside \
        else (0*zpos, 0*xpos)
    if check(rmov):
        return True

    # scan
    for rmov in spiral_scan():
        if check(rmov):
            return True

    return False


def drift_to_beam(cam, xmotor, zmotor, pixelsize, tolerance=5,
                  max_iterations=100):
    """
    Moves the camera *cam* with motors *xmotor* and *zmotor* until the
    center of mass is nearer than *tolerance*-pixels to the center of the
    frame or *max_iterations* is reached.

    To convert pixelcoordinates to realworld-coordinates of the motors the
    *pixelsize* (scalar or 2-element array-like, e.g. [4*q.um, 5*q.um]) is
    needed.
    """
    def take_frame():
        cam.trigger()
        return gaussian_filter(cam.grab().astype(np.float32), 40.0)

    units = (u for u in pixelsize.units if pixelsize.units[u] > 0.0)
    units = q.parse_expression(' * '.join(units))
    ps = np.tile(pixelsize.magnitude, 2)[:2] * units

    img = take_frame()
    frm_center = (np.array(img.shape)-1)/2
    d = center_of_mass(img-img.min()) - frm_center

    iter_ = 0
    while ((d**2).sum() > tolerance**2) and (iter_ < max_iterations):
        fut_0 = zmotor.move(d[0]*ps[0])
        fut_1 = xmotor.move(d[1]*ps[1])
        wait([fut_0, fut_1])
        img = take_frame()
        if img.sum() == 0:
            LOG.debug("drift_to_beam: Frame is empty (sum == 0). " +
                      "Can't follow center of mass.")
            raise ProcessException("There is nothing to see! "
                                   "Can't follow the center of mass.")
        d = center_of_mass(img-img.min()) - frm_center
        iter_ += 1

    if iter_ < max_iterations:
        return True
    else:
        return False


@async
def center_to_beam(cam, xmotor, zmotor, pixelsize, xborder, zborder,
                   xstep=None, zstep=None, thres=1000, tolerance=5,
                   max_iterations=100):
    """
    Tries to center the camera *cam* to the beam by moving with the motors
    *xmotor* and *zmotor*. It starts by searching the beam inside the
    search-area defined by *xborder* and *zborder*. Argument *pixelsize* is
    needed to convert pixelcoordinates into realworld-coordinates of the
    motors.
    Exceptions are raised on fail.

    Optional arguments *xstep*, *zstep*, *thres*, *tolerance* and
    *max_iterations* are passed to the functions 'find_beam(...)' and
    'center2beam(...)'.
    """
    cam.trigger_mode = cam.trigger_modes.SOFTWARE
    cam.start_recording()

    try:
        with cam, xmotor, zmotor:
            if not find_beam(cam, xmotor, zmotor, pixelsize, xborder, zborder,
                             xstep, zstep, thres):
                message = "Unable to find the beam"
                LOG.debug('center_to_beam: '+message)
                raise ProcessException(message)
            else:
                LOG.debug('center_to_beam: switch to drift_to_beam')
                if not drift_to_beam(cam, xmotor, zmotor, pixelsize, tolerance,
                                     max_iterations):
                    message = "Maximum iterations reached"
                    LOG.debug('center_to_beam: '+message)
                    raise ProcessException(message)
    finally:
        cam.stop_recording()


class ProcessException(Exception):

    """
    Exception raised by a process when something goes wrong with the procedure
    it tries to accomplish, e.g. cannot focus, cannot align rotation axis, etc.

    """

    pass
