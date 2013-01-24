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
two linear motor controllers to describe and restrict to circular paths. To
improve latencies, all motors should be able to move _simultaneous_.


## Motor Groups

Often, motors are mounted on top of each other. To indentify relative coordinate
transforms between any two motors, the hierarchical relationship between motors
needs to be captured in a tree structure. This structure should be
serializable, and it should be possible to annotate it with real CAD-data.


## Trajectories

A trajectory describes the final path that the (combined) movement of a motor,
pseudo motor or a motor group traces. It can be a list of positions that are
moved to one by one or a function that is evaluated at discrete sample
positions.

Before executing a trajectory, a collision check should be performed.


# Architecture

Each controller exposes its parameters via a name.

