'''
Dummy attenuator box
'''
from concert.devices.attenuatorboxes import base
from concert.base import check, Selection


class AttenuatorBox(base.AttenuatorBox):
    '''Attenuator class implementation'''
    attenuator = Selection([None, 'Al_1mm'], check=check(source='standby', target='standby'))

    def __init__(self):
        super(AttenuatorBox, self).__init__()
        self._filter = None

    async def _set_attenuator(self, attenuator):
        self._filter = attenuator

    async def _get_attenuator(self):
        return self._filter
