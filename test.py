import controller.scan
import controller.crio
import logging

if __name__ == '__main__':
    logger = logging.getLogger('crio')
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)

    lm = controller.crio.LinearMotor() 
    lm.move_to_relative_position(0.5)
    controller.scan.mesh([lm], 15000.0)
