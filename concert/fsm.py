# Code based heavily on django-fsm

from collections import defaultdict
from functools import wraps


class TransitionNotAllowed(Exception):
    pass


class StateValue(object):

    def __init__(self, default):
        self._current = default
        self._error = None

    @property
    def value(self):
        return self._current

    @property
    def error(self):
        return self._error

    def is_currently(self, state):
        return self._current == state

    def _set_value(self, value):
        self._current = value

    def _set_error(self, error):
        self._error = error


class State(object):
    def __init__(self, default):
        self.default = default

    def __get__(self, instance, owner):
        if not hasattr(instance, '_concert_state'):
            setattr(instance, '_concert_state', StateValue(self.default))

        return getattr(instance, '_concert_state')


class Meta(object):
    def __init__(self):
        self.transitions = defaultdict()
        self.state_name = None

    def get_state_attribute(self, instance):
        if not self.state_name:
            for name in dir(instance):
                if isinstance(getattr(instance, name), StateValue):
                    self.state_name = name

        return getattr(instance, self.state_name)

    def add_transition(self, source, target, immediate):
        if source in self.transitions and self.transitions[source] != target:
            raise AssertionError("Duplicate transition for {}".format(source))

        if immediate:
            self.transitions[source] = immediate
            self.transitions[immediate] = target
        else:
            self.transitions[source] = target

    def do_transition(self, instance, target):
        current = self.get_state_attribute(instance).value
        next_state = None

        try:
            next_state = self.transitions[current]
        except KeyError:
            next_state = self.transitions.get('*')

        if current is None or next_state != target:
            msg = "Invalid transition from {} to {}".format(current, target)
            raise TransitionNotAllowed(msg)

        attr = self.get_state_attribute(instance)
        attr._set_value(target)

    def signal_error(self, instance, msg):
        attr = self.get_state_attribute(instance)
        attr._set_value('error')
        attr._set_error(msg)


def transition(source='*', target=None, immediate=None):
    def inner_transition(func):
        if not hasattr(func, '_concert_fsm'):
            setattr(func, '_concert_fsm', Meta())

        if isinstance(source, (list, tuple)):
            for state in source:
                func._concert_fsm.add_transition(state, target, immediate)
        else:
            func._concert_fsm.add_transition(source, target, immediate)

        @wraps(func)
        def _change_state(instance, *args, **kwargs):
            meta = func._concert_fsm

            if immediate:
                meta.do_transition(instance, immediate)

            try:
                result = func(instance, *args, **kwargs)
                meta.do_transition(instance, target)
            except Exception as e:
                meta.signal_error(instance, str(e))
                raise e

            return result

        return _change_state

    return inner_transition
