'''
Dummy attenuator box
'''
from concert.devices.attenuatorboxes import base
from concert.base import Selection


class AttenuatorBox(base.AttenuatorBox):
    '''Attenuator class implementation'''
    attenuator = Selection([None, 'Al_1mm'])

    def __init__(self):
        super(AttenuatorBox, self).__init__()
        self._filter = None

    def _set_attenuator_real(self, attenuator):
        self._filter = attenuator

    def _get_attenuator_real(self):
        return self._filter
