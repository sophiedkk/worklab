# Ergometer testing

This guide will explain how the settings of the Esseda wheelchair
ergometer can be used to simulate overground propulsion or set a target
power output for your participants. It assumes a properly operated and
calibrated system.

The Esseda is a wheelchair roller ergometer developed and produced by
Lode BV (Groningen, The Netherlands). It is equipped with a set of
servomotors and load cells to simulate and measure wheelchair propulsion.
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

The Esseda wheelchair ergometer needs to be calibrated before testing.
The system friction of the ergometer is determined using a dynamic
calibration procedure. Every calibration is unique due to the differences
of each participant (mass, distribution, tension straps). The dynamic
calibration procedure starts at 0 m/s and steadily increases in small steps
to 2.5 m/s. Although every calibration procedure is unique for the 
wheelchair-user combination, all calibrations follow a similar pattern.

## Noise

The measurement signal of the Esseda wheelchair ergometer is subject to
a certain noise level. Tension on the straps, wheelchair tyre type and 
changes in posture can all add noise to the signal. Butterworth filters
can be used on the force and velocity signal to filter some noise.

## Acceleration

The use of servomotors in the ergometer leads to a realistic acceleration
and deceleration of the wheelchair-user combination dependent on the
friction coefficient provided. Differentiating the velocity signal results 
in the acceleration signal. The measured forces are also dependent on the
acceleration of the wheelchair-user combination.

# Measurement modes

The Esseda wheelchair ergometer is able to use four different modes to
evaluate wheelchair propulsion performance.

## Isoinertial 
In the isoinertial mode, the rolling resistance coefficient can be set
and the wheelchair user can be tested under realistic conditions. The
rolling resistance coefficient can be adjusted to simulate different
surfaces or gradually increase the intensity of wheelchair propulsion.

## Isokinetic
In the isokinetic mode, a speed limit can be set on the ergometer. It 
will feel like a ceiling effect. Participants can push as hard as they 
can, they will not be able to go beyond the given speed limit. The
servomotor of the ergometer adjusts the friction based on the velocity.

## Isospeed
In the iso speed mode, the wheels can spin at a given velocity.
The velocity is not affected by the applied forces, which could be
used to increase coordination issues in wheelchair propulsion.

## Isometric
In the isometric mode, the rollers of the wheelchair ergometer are 
blocked. This allows the wheelchair-user to complete an isometric strength
test. The wheels can be pushed as a hard as someone can without them moving.

# References
De Klerk, R., Vegter, R. J. K., Veeger, H. E. J., & Van der Woude, L. H. V. (2020). Technical note:
A novel servo-driven dual-roller handrim wheelchair ergometer. IEEE Transactions on Neural
Systems and Rehabilitation Engineering, 28(4), 953–960. https://doi.org/10.1109/TNSRE.2020.
2965281 

Janssen R.J.F., Vegter R.J.K., Houdijk H., Van der Woude L.H.V., de Groot S. Evaluation of a 
standardized test protocol to measure wheelchair-specific anaerobic and aerobic exercise capacity
in healthy novices on an instrumented roller ergometer. PLoS One. 2022 Sep 6;17(9):e0274255.
https://doi.org/10.1371/journal.pone.0274255