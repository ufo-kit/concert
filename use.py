#!/usr/bin/env python

import controller as ct

def print_parameters(controllers):
    print "---"
    for controller in controllers:
        for k in controller.parameters.keys():
            print "%s: %f" % (k, controller.parameters[k].value)

class PositionLogger(object):
    def __init__(self):
        self.x = []
        self.y = []

    def append_x(self, param, x):
        self.x.append (x)

    def append_y(self, param, y):
        self.y.append (y)


if __name__ == '__main__':
    # Just an example how to perform an ascan that will scan a range of
    # parameters and call print_parameters for each final position.
    lm1 = ct.LinearMotor(None)
    lm2 = ct.LinearMotor(None)
    ct.meshscan([lm1, lm2], 4.0, print_parameters)

    # Log both linear motor's position, because we combine them into one
    # pseudo rotation "motor". 
    pl = PositionLogger()
    l1_param = lm1.parameters['position']
    l2_param = lm2.parameters['position']
    l1_param.add_callback(pl.append_x)
    l2_param.add_callback(pl.append_y)

    # Instead of combining two one-dimensional motors, one two-dimensional
    # motor could also be used
    prm = ct.PseudoRotationMotor(l1_param, l2_param)
    ct.meshscan([prm], 0.2)

    # Show positions of the linear motors 
    import matplotlib.pyplot as plt
    plt.plot(pl.x, pl.y, 'rx')
    plt.show()
