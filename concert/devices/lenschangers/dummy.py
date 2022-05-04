from concert.base import check, Selection
from concert.devices.lenschangers.base import LensChanger as BaseLensChanger


class LensChanger(BaseLensChanger):
    '''Lens changer class implementation'''
    objective = Selection(['objective_10x', 'objective_5x'], help='objective',
                          check=check(source='standby', target='standby'))

    async def __ainit__(self):
        await super(LensChanger, self).__ainit__()
        self._objective = 'objective_10x'

    async def _set_objective(self, objective):
        self._objective = objective

    async def _get_objective(self):
        return self._objective
