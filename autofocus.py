import logging
import argparse
import controller.crio
from process.gradientmaximizer import GradientMaximizer
from measure.optimization import Maximizer

from gi.repository import Ufo, Uca


if __name__ == '__main__':
    config = Ufo.Config(paths=['/home/matthias/dev/ufo-filters/build/src'])

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
    lm = controller.crio.LinearMotor()

    focus = GradientMaximizer(camera, sharpness, lm, config)
    focus.start()
