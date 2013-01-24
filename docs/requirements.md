% Control System Interface Requirements


# Introduction

This document specifies requirements for a software interface to the beamline
control system.


## Glossary

ascan

:   Change a parameter of a motor device in step-wise fashion. [_should this be
    defined in terms of step size or as number of intervals?_]

meshscan

:   Change parameters of a set of motor devices so that a grid is traced.

Control system

:   A low-level software interface to communicate with a hardware device. This
    can be Tango or a proprietary software/hardware combination such as the
    Aerotech controller.


# Requirements


## Device abstractions

* Motors
* Pseudo motors
* Tree-structured motor groups


## Actions

* Parameter scans such as ascan, dscan and meshscan along trajectories.
* Trajectories are defined as discrete positions or a sampled implicit function.
* Persistent logging of all movements must be present.
