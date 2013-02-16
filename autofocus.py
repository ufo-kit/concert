import logging
import controller.crio
from process.gradientmaximizer import GradientMaximizer
from measure.optimization import Maximizer

from gi.repository import Ufo, Uca


if __name__ == '__main__':
    config = Ufo.Config()

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    pm = Uca.PluginManager()
    camera = pm.get_camera('mock')

    sharpness = Maximizer()
    lm = controller.crio.LinearMotor()

    focus = GradientMaximizer(camera, sharpness, lm, config)
    focus.start()
