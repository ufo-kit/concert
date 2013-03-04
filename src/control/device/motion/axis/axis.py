from controlobject import Identifiable
import numpy
import eventgenerator
from event import Event
from control.events import eventtype
from threading import Thread
from connection import TangoConnection, SocketConnection
import time


#===============================================================================
# Calibration
#===============================================================================


class _Calibration(object):
    def to_user(self, value):
        raise NotImplementedError

    def to_steps(self, value):
        raise NotImplementedError


class LinearCalibration(_Calibration):
    def __init__(self, units_per_step, offset):
        self._units_per_step = units_per_step
        self._offset = offset

    def to_user(self, value_in_steps):
        return value_in_steps*self._units_per_step + self._offset

    def to_steps(self, value):
        return (value.rescale(self._units_per_step.units) + self._offset)/\
                                    self._units_per_step
                                    

#===============================================================================
# Axis
#===============================================================================


class LimitReached(Exception):
    def __init__(self, limit):
        self._limit = limit
        
    def __str__(self):
        return repr(self._limit)


class Limit(object):
    HARD = 0
    SOFT = 1
    def __init__(self, limit_type):
        self._type = limit_type
        
    @property
    def limit_type(self):
        return self._type
    
    def __repr__(self):
        if self._type == Limit.HARD:
            limit_type_str = "HARD"
        else:
            limit_type_str = "SOFT"
        return "Limit(type=%s)" % (limit_type_str)
    
    def __str__(self):
        return repr(self)


class Movable(Identifiable):
    """Base class for everything that moves."""
    def __init__(self, calibration):
        self._calibration = calibration

    def home(self):
        pass
    
    def stop(self, blocking=False):
        def _stop_with_event():
            self._stop_real()
            eventgenerator.fire(Event(eventtype.Motion.STOP, self))
        
        if blocking:
            _stop_with_event()
        else:
            # Create a thread and execute asynchronously.
            # TODO: make use of threads pool of whatever that improves
            # performance.
            t = Thread(target=_stop_with_event)
            # We don't care if the desired position is reached after the
            # process terminates.
            t.daemon = True
            t.start()
    
    def is_out_of_limits(self, value, limit):
        return limit and value < limit[0] or value > limit[1]    
    
    def _stop_real(self):
        """Stop the physical axis. This method must be always blocking
        in order to provide appropriate events at appropriate times.
        
        """
        raise NotImplementedError
    
    def __del__(self):
        # Be profound.
        self.stop()


class DiscretelyMovable(Movable):
    def __init__(self, calibration, position_limit=(-numpy.inf, numpy.inf)):
        super(DiscretelyMovable, self).__init__(calibration)
        self._position_limit = position_limit
        self._position = None
    
    def get_position(self):
        return self._calibration.to_user(self._get_position_real())
    
    def set_position(self, position, blocking=False):
        """Set position of the device.
        
        @param position: position in user units.
        @param blocking: True if the call will block until the movement is done.
        
        """
        position = self._calibration.to_steps(position)
        if self.is_out_of_limits(position, self._position_limit):
            limit = Limit(Limit.SOFT)
            eventgenerator.fire(Event(eventtype.Motion.LIMIT_BREACH,
                                      self, limit))
            raise LimitReached(limit)

        def _set_position_with_events(position):
            # Fire appropriate events and move the device.
            eventgenerator.fire(Event(eventtype.Motion.START, self))
            self._set_position_real(position)
            if self._is_hard_position_limit_reached():
                eventgenerator.fire(Event(eventtype.Motion.LIMIT_BREACH,
                                          self, Limit(Limit.HARD)))
            eventgenerator.fire(Event(eventtype.Motion.STOP, self))
            
        if blocking:
            _set_position_with_events(position)
        else:
            # Create a thread and execute asynchronously.
            # TODO: make use of threads pool of whatever that improves
            # performance.
            t = Thread(target=_set_position_with_events, args=(position,))
            # We don't care if the actual stopping happens after the process
            # terminates.
            t.daemon = True
            t.start()
            
    def _is_hard_position_limit_reached(self):
        raise NotImplementedError
    
    def _set_position_real(self, position):
        """Call to the device itself for physical position setting. This
        method must be always blocking in order to provide appropriate events
        at appropriate times.
        
        """
        raise NotImplementedError
    
    def _get_position_real(self):
        """Call to the device itself for physical position query."""
        raise NotImplementedError


class ContinuouslyMovable(Movable):
    def __init__(self, calibration, velocity_limit=(-numpy.inf, numpy.inf)):
        super(DiscretelyMovable, self).__init__(calibration)
        self._velocity_limit = velocity_limit
        self._velocity = None

    def get_velocity(self):
        return self._calibration.to_user(self._get_velocity_real())

    def set_velocity(self, velocity, blocking=False):
        velocity = self._calibration.to_steps(velocity)
        if self.is_out_of_limits(velocity, self.velocity_limit):
            limit = Limit(Limit.SOFT)
            eventgenerator.fire(Event(eventtype.Motion.LIMIT_BREACH,
                                      self, limit))
            raise LimitReached(limit)

        def _set_velocity_with_events(velocity):
            # Fire appropriate events and move the device.
            eventgenerator.fire(Event(eventtype.Motion.START, self))
            self._set_velocity_real(velocity)
            if self._is_hard_velocity_limit_reached():
                eventgenerator.fire(Event(eventtype.Motion.LIMIT_BREACH,
                                          self, Limit(Limit.HARD)))
            else:
                eventgenerator.fire(Event(
                            eventtype.ContinuousMotion.VELOCITY_STEADY, self))
            
        if blocking:
            _set_velocity_with_events(velocity)
        else:
            # Create a thread and execute asynchronously.
            # TODO: make use of threads pool of whatever that improves
            # performance.
            t = Thread(target=_set_velocity_with_events, args=(velocity,))
            # We don't care if the desired velocity is reached after the
            # process terminates.
            t.daemon = True
            t.start()

    def _is_hard_velocity_limit_reached(self):
        raise NotImplementedError

    def _get_velocity_real(self):
        """Call to the device itself for physical velocity setting."""
        raise NotImplementedError

    def _set_velocity_real(self, velocity):
        """Call to the device itself for physical velocity query. This
        method must be always blocking in order to provide appropriate events
        at appropriate times.
        
        """
        raise NotImplementedError
    
    
#===============================================================================
# Implementations
#===============================================================================


class DummyDiscreteAxis(DiscretelyMovable):
    def __init__(self, calibration, position_limit=(-numpy.inf, numpy.inf)):
        super(DummyDiscreteAxis, self).__init__(calibration, position_limit)
        self._hard_limits = -10, 10
        self._position = 0
        
    def _set_position_real(self, position):
        time.sleep(numpy.random.random())
        self._position = position
        if self._position < self._hard_limits[0]:
            self._position = self._hard_limits[0]
        elif self._position > self._hard_limits[1]:
            self._position = self._hard_limits[1]
        
    def _get_position_real(self):
        return self._position
    
    def _is_hard_position_limit_reached(self):
        return self._position <= self._hard_limits[0] or\
                self._position >= self._hard_limits[1]

class TangoDiscreteAxis(DiscretelyMovable, TangoConnection):
    def __init__(self, uri, calibration,
                 position_limit=(-numpy.inf, numpy.inf),
                 tango_host=None, tango_port=None):
        DiscretelyMovable.__init__(self, calibration, position_limit)
        TangoConnection.__init__(self, uri, tango_host, tango_port)
        
        
class SocketDiscreteAxis(DiscretelyMovable, SocketConnection):
    def __init__(self, uri, calibration, host, port,
                 position_limit=(-numpy.inf, numpy.inf),):
        DiscretelyMovable.__init__(self, calibration, position_limit)
        SocketConnection.__init__(self, uri, host, port)