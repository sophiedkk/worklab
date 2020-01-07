Overground
==========
In wheelchair propulsion the main source of energy loss is rolling friction, which depends on the wheelchair, user, and
wheelchair-user interface. At high speeds aerodynamic drag rapidly increases as it is a non-linear function of speed.
Wheelchair design, mechanics, and maintenance can significantly alter the resistive forces acting on the wheelchair and
user. In an everyday setting these forces are to be minimised to reduce strain and, in a sports setting, they have to be
minimised to optimize performance. The forces can either be identified through a drag-test or with the coasting
deceleration method. This document will give a short overview of rolling resistance, air drag, and how to determine
them. It will also provide some reference values where possible that can be used for simulation on a treadmill or
wheelchair ergometer. It aims to be as concise as possible.

Determining power output
------------------------
In a coast-down test you let the wheelchair coast-down without the user applying force on the handrims. The user
should assume a position that is representative of the position during wheelchair propulsion. The deceleration is
reflective of the frictional forces acting on the wheelchair-user combination (Figure 2). Usually, this is done in a
back-and-forth manner to compensate for factors such as uneven ground. The coast-down test is a common procedure
to assess the dissipative forces and their coefficients. A number of instrumentation options is available (slope,
time-gates, velocity), but here we will only discuss the method where velocity data is available as many affordable
technologies for gathering velocity data are currently available. This data could, for example, be gathered with a
tachometer, measurement wheel, or an IMU.

Protocol
^^^^^^^^
For a regular handrim wheelchair the following protocol can be used:

- The test subject should remain upright in a neutral position
- The experimenter accelerates the wheelchair to a “high” velocity
- The wheelchair should decelerate to a complete standstill without interference
- Time and velocity data should be measured for the deceleration period

Analysis
^^^^^^^^
When we assume a constant friction, it is relatively easy to determine the frictional forces. This assumption is safe to
make at relatively low speeds, but does become dangerous for most sports applications. A linear regression should be fit
to the linear deceleration. Knowing that there is no outside force acting on the wheels other than friction, it is
relatively easy to determine the total frictional force. The coefficient of friction can then be extracted with:

.. math:: \mu = \frac{Ma}{mg}

An open-source implementation for the analysis of coast-down data has been developed at the University Medical Centre
Groningen (`Coast-down analyzer <https://gitlab.com/Rickdkk/coast_down_test>`_).

When constant friction cannot be assumed due to air drag, i.e. in most sports environments, the analysis becomes a
little more complex. The protocol is identical; however, initial speed should probably be higher. In this case, a
non-linear differential equation needs to be solved and that equation needs to be fit with a curve fitter
(e.g. Levenberg-Marquardt).

IMUs
----

References
----------
