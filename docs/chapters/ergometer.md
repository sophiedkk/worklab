---
title: Ergometer
---

This guide will explain how the settings of the Esseda wheelchair
ergometer can be used to simulate overground propulsion or set a target
power output for your participants. It assumes a properly operated and
calibrated system.

The Esseda is a wheelchair roller ergometer developed and produced by
Lode BV (Groningen, The Netherlands). It is equipped with a set of
servomotors and loadcells to simulate and measure wheelchair propulsion.
With the servomotors in the ergometer you can make propulsion as light
or as heavy as you would like it to be.

A simple mechanical model is used to simulate overground propulsion.
This model simulates:

-   The inertia of the participant + the wheelchair
-   The rolling resistance of the wheelchair

Both values can be changed accordingly in LEM. Weight can only be set
before every measurement. Friction can be changed during measurements to
allow for ramp or step protocols. Total friction (left + right module)
is calculated by multiplying the weight of the participant and the
wheelchair (m) with the gravitational constant (g) and a rolling
resistance constant (μ):

$$F_{friction} = m * g * \mu$$

# Setting power output

Oftentimes researchers are more interested in power output (W). During
steady-state propulsion, the velocity of the wheelchair oscillates
around a steady point with every push. Additionally, simulated friction
is assumed to be independent of velocity. As such, power output can be
assumed to be the product of the frictional force (F) and mean velocity
(v):

$$P_{out} = F_{friction} * v$$

As such, the expected power output on the Esseda wheelchair ergometer
can be calculated with relative ease.

# Calibrations

## Drift

## Noise

## Calibration

## Acceleration

# References
