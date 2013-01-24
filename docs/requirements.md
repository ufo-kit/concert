% Control System Interface Requirements


# Introduction

This document specifies requirements for a software interface to the beamline
control system.


## Glossary

ascan

:   Change a parameter of a motor device in step-wise fashion. [Question: should
    this be defined in terms of step size or as number of intervals?]

meshscan

:   Change parameters of a set of motor devices so that a grid is traced.

Control system

:   A low-level software interface to communicate with a hardware device. This
    can be Tango or a proprietary software/hardware combination such as the
    Aerotech controller.


# Requirements

* Motors must be actuated directly
* Parameter scans must be performed. This includes ascan, dscan and meshscan.
* There must be pseudo motors that combine physical or other pseudo motors and
  perform a combined action.
* Persistent logging of all movements must be present.
