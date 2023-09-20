# Overground testing

In wheelchair propulsion the main source of energy loss is rolling
friction, which depends on the wheelchair, user, and wheelchair-user
interface. At high speeds aerodynamic drag rapidly increases as it is a
non-linear function of speed. Wheelchair design, mechanics, and
maintenance can significantly alter the resistive forces acting on the
wheelchair and user. In an everyday setting these forces are to be
minimised to reduce strain and, in a sports setting, they have to be
minimised to optimize performance. The forces can either be identified
through a drag-test or with a coasting deceleration method. This
document will give a short overview of rolling resistance, air drag, and
how to determine them. It will also provide some reference values where
possible that can be used for simulation on a treadmill or wheelchair
ergometer. It aims to be as concise as possible.

# Determining power output

In a coast-down test you let the wheelchair coast-down without the user
applying force on the hand rims. The user should assume a position that
is representative of the position during wheelchair propulsion. The
deceleration is reflective of the frictional forces acting on the
wheelchair-user combination (Figure 2). Usually, this is done in a
back-and-forth manner to compensate for factors such as uneven ground.
The coast-down test is a common procedure to assess the dissipative
forces and their coefficients. A number of instrumentation options is
available (slope, time-gates, velocity), but here we will only discuss
the method where velocity data is available as many affordable
technologies for gathering velocity data are currently available. This
data could, for example, be gathered with a tachometer, measurement
wheel, or an IMU.

## Protocol

For a regular handrim wheelchair the following protocol can be used:

-   The test subject should remain upright in a neutral position
-   The experimenter accelerates the wheelchair to a "high" velocity
-   The wheelchair should decelerate to a complete standstill without
    interference
-   Time and velocity data should be measured for the deceleration
    period

## Analysis

When we assume a constant friction, it is relatively easy to determine
the frictional forces. This assumption is safe to make at relatively low
speeds, but does become dangerous for most sports applications. A linear
regression should be fit to the linear deceleration. Knowing that there
is no outside force acting on the wheels other than friction, it is
relatively easy to determine the total frictional force. The coefficient
of friction can then be extracted with:

$$\mu = \frac{Ma}{mg}$$

An open-source implementation for the analysis of coast-down data has
been developed at the University Medical Centre Groningen ([Coast-down
analyzer](https://gitlab.com/Rickdkk/coast_down_test)).

When constant friction cannot be assumed due to air drag, i.e. in most
sports environments, the analysis becomes a little more complex. The
protocol is identical; however, initial speed should probably be higher.
In this case, a non-linear differential equation needs to be solved and
that equation needs to be fit with a curve fitter (e.g.
Levenberg-Marquardt).

# IMUs
Inertial measurement units (IMUs) are small sensors that can easily be placed on 
the hub of both wheels and the frame of the wheelchair. IMUs are able to
measure accelerometer, gyroscope and magnetometer data. The gyroscope signal
of the wheels gives us the rotational velocity of both wheels. Since the wheel
size is fixed, the linear velocity can be determined. Using a correction for the
camber angle of the wheels, using the sensor on the frame, an accurate and reliable
velocity signal is created. Taking the slope of the velocity signal will result in
the deceleration profile of the wheelchair-user combination.

# References
De Klerk, R., Vegter, R. J. K., Leving, M. T., De Groot, S., Veeger, H. E. J., & Van der
Woude, L. H. V. (2020). Determining and controlling external power output during regular
handrim wheelchair propulsion. Journal of Visualized Experiments, 156, e60492. 
https://doi.org/10.3791/60492

Hoffman, M. D., Millet, G. Y., Hoch, A. Z., & Candau, R. B. (2003). Assessment of wheelchair drag
resistance using a coasting deceleration technique. American Journal of Physical Medicine and
Rehabilitation, 82(11), 880–889. https://doi.org/10.1097/01.PHM.0000091980.91666.58 

Rietveld, T., Mason, B.S., Goosey-Tolfrey, V.L., van der Woude, L.H.V., de Groot, S.,
Vegter, R.J.K., 2021a. Inertial measurement units to estimate drag forces and power
output during standardised wheelchair tennis coast-down and sprint tests. Sports
BioMech. 1–19. https://doi.org/10.1080/14763141.2021.1902555

Van der Slikke, R. M. A., Berger, M. A. M., Bregman, D. J. J., Lagerberg, A. H., & Veeger, H. E. J.
(2015). Opportunities for measuring wheelchair kinematics in match settings; reliability of
a three inertial sensor configuration. Journal of Biomechanics, 48(12), 3398–3405. 
https://doi.org/10.1016/j.jbiomech.2015.06.001

Van der Woude, L. H. V., Veeger, H. E. J., Dallmeijer, A. J., Janssen, T. W. J., & Rozendaal, L. A.
(2001). Biomechanics and physiology in active manual wheelchair propulsion. Medical
Engineering and Physics, 4(23), 713–733. https://doi.org/10.1016/S1350-4533(01)00083-2