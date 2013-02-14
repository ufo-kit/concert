import controller.crio
from process.gradientmaximizer import GradientMaximizer
from measure.maximization import Maximizer
import logging

from gi.repository import Ufo, Uca


if __name__ == '__main__':
    config = Ufo.Config(paths=['/home/farago/dev/ufo/ufo-filters/build/src'])

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    pm = Uca.PluginManager()
    camera = pm.get_camera('mock')

    sharpness = Maximizer()
    lm = controller.crio.LinearMotor()

    focus = GradientMaximizer(camera, sharpness, lm, config)
    focus.start()
