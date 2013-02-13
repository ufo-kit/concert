import controller.crio
import process.focus
import measure.sharpness
import logging

from gi.repository import Ufo, Uca


if __name__ == '__main__':
    config = Ufo.Config(paths=['/home/matthias/dev/ufo-filters/build/src'])

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    pm = Uca.PluginManager()
    camera = pm.get_camera('mock')

    sharpness = measure.sharpness.Sharpness(camera, config)
    lm = controller.crio.LinearMotor()

    focus = process.focus.Focus(camera, sharpness, lm)
    focus.start()
