% Control System Interface Design

# Data structures

## Parameters

A parameter is a value that corresponds to a real-world physical attribute of a
motor. It must have a type, a unit and access regulations. Depending on the type
and actual device, the range of possible should be limitable. In Sardana this
translates to a controller's `axis_attributes` dictionary.

Parameters could be changed from the software side (as part of the control
interface) or from outside due to a master-slave relationship to another device.
To account for this, user provided code must be able to observe changes by being
notified about any changes.


## Controllers

A controller represents any controllable device.


## Motor Controller

Minimum interface: move to an absolute position, move relative to the current
position.


## Pseudo Controllers

A pseudo controller encapsulates other controllers or pseudo controllers to
create new combined behaviour. For example, a pseudo controller could combine
two linear motor controllers to describe and restrict to circular paths.


# Architecture

Each controller exposes its parameters via a name

