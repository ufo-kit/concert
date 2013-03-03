import logging
import argparse

from gi.repository import Ufo, Uca
from control.measure.optimization import Maximizer
from control.motion.hardware.crio import CrioLinearAxis
from control.process.gradientmaximizer import GradientMaximizer


if __name__ == '__main__':
    config = Ufo.Config()

    parser = argparse.ArgumentParser(description='Run autofocus procedure')
    parser.add_argument('--camera', type=str, default='mock',
                        help='libuca camera identifier')

    args = parser.parse_args()

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    pm = Uca.PluginManager()
    camera = pm.get_camera(args.camera)

    sharpness = Maximizer()
    lm = CrioLinearAxis()

    focus = GradientMaximizer(camera, sharpness, lm, config)
    focus.start()
