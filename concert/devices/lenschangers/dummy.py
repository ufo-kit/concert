from concert.base import Selection
from concert.devices.lenschangers.base import LensChanger as BaseLensChanger


class LensChanger(BaseLensChanger):
    '''Lens changer class implementation'''
    objective = Selection(['objective_10x', 'objective_5x'])

    def __init__(self):
        super(LensChanger, self).__init__()
        self._objective = 'objective_10x'

    def _set_objective_real(self, objective):
        self._objective = objective

    def _get_objective_real(self):
        return self._objective
