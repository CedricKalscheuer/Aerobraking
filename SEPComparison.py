import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import time as tm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import CheckButtons, Slider
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema

#region User input
# === Rocket parameters ============================
m_0                         = 11745.0            # Initial spacecraft mass in [kg]
m_p                         = 2725.0            # Propellant mass in [kg]
v_inf_values                = [0.42, 0.45, 0.88, 0.9]#, 1]#, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]  # Hyperbolic excess speeds at infinity in [km/s] (5.5km/s for Cassini)
thrust                      = 0                 # Thrust in [N]
Isp                         = 3100              # Specific impulse
c_d                         = 2.2               # Drag coefficient
A                           = 20.0              # Cross-sectional area of the spacecraft in [m^2]
c_q                         = 0.5               # 0.5 for highest Heat Load
# === Orbit parameters ============================
max_crossings               = 8
# === Output parameters ============================
Plot_Trajectory             = True              # Plot spacecraft trajectory?
Plot_Atmosphere             = False             # Plot atmospheric density over height?
Plot_Values                 = True             # Plot panel of values?
Plot_Comparison             = False             # Plot v_inf vs. r_p required?
Plot_HeatingRate            = False             # Plot heating rate maximum and average over time?
#endregion

#region Prepare the program
# Planetary and Orbital constants
g0_earth            = 9.80665               # Earth standard gravitational acceleration in [m/s^2]
# === Saturn ============================
mu_saturn           = 37931207.8            # Gravitational parameter of Saturn in [km^3/s^2]
R_saturn            = 60268.0               # Radius of Saturn in [km]
dist_saturn_titan   = 1221870.0             # Approximate distance between Saturn and Titan in [km]
R_SOI_saturn        = 54.5 * 1e6            # Saturn's Radius of Sphere of Influence in [km]
# === Titan ============================
mu_titan            = 8978.14               # Gravitational parameter of Titan in [km^3/s^2]
R_titan             = 2574.73               # Titan equatorial radius in [km]
R_SOI_titan         = 43321               # Titan's radius of standard Sphere of Influence in [km] SHOULD BE 43321??????
a_titan             = 1221870               # Semi-major axis in km
e_titan             = 0.0288                # Eccentricity
T_titan             = 2 * np.pi * np.sqrt(a_titan**3 / mu_saturn)
# === Enceladus ============================
mu_enceladus        = 7.202e3               # Gravitational parameter of Enceladus in [km^3/s^2]
R_enceladus         = 252.1                 # Enceladus equatorial radius in [km]
R_SOI_enceladus     = 488                   # Enceladus radius of standard Sphere of Influence in [km]
a_enceladus         = 237948                # Semi-major axis of Enceladus' orbit in [km]
e_enceladus         = 0.0047                # Eccentricity of Enceladus' orbit
T_enceladus         = 2 * np.pi * np.sqrt(a_enceladus**3 / mu_saturn)  # Orbital period of Enceladus in seconds
# === Accuracy ============================
tolerance                   = 0.0000001         # Tolerance for calculations
max_step_critical           = 1.0            # Maximum step size during simulation for critical areas
max_step_non_critical       = 800             # Maximum step size during simulation for non-critical areas
critical_distance_titan_sim = R_titan + 10000
critical_distance_saturn_sim= R_saturn
# Start values for later iteration
r_thrust_descending         = 900000         # Altitude where thrust begins for orbit insertion maneuver
thrust_duration             = 0 * 3600       # Thrust duration in seconds
thrust_duration_reduction   = 0 * 3600       # Reduction of thrust_duration in [s]
thrust_duration_limit       = 0 * 3600       # Lower limit of thrust_duration reduction
# === Old Orbit parameters (OUTDATED) ============================
tint                        = 2500               # Integration time in days
tint_short                  = 2500               # Integration time for short runs
Optimize_Thrust_Duration    = False             # Optimize thrust duration? (NOT WORKING ATM/USELESS FUNCTION)
r_p_lowest                  = 100               # Lowest Periapsis tested
r_p_highest                 = 1101              # Highest Periapsis tested
r_p_step_size               = 10                # Step size between r_p_lowest and r_p_highest
r_p_orbit                   = 1500              # Periapsis of desired orbit in [km]
r_a                         = 2000              # Apoapsis of desired orbit in [km]
# === Unit Conversion ============================
tmax                = tint * 86400                  # Convert integration time in [s]
tmax_short          = tint_short * 86400            # Convert integration time in [s]
A                   = A / 1e6                       # Convert cross-sectional area to [km^2]
thrust              = thrust / 1000.0               # Convert thrust to [kg*km/s^2]
ceff                = Isp * g0_earth / 1000.0       # Effective exhaust velocity in [km/s]
r_p_orbit           = R_titan + r_p_orbit           # Periapsis of desired orbit including planet radius
r_a                 = R_titan + r_a                 # Apoapsis of desired orbit including planet radius
r_p_values          = np.arange(R_titan + r_p_lowest, R_titan + r_p_highest, r_p_step_size)

# Initialize mission phase variables
orbit_insertion_maneuver    = True      # Set to False after the first thruster burn
full_aerobraking            = True      # Set to False when r_p is raised to r_p_slow
is_descending               = True      # Track if spacecraft is ascending or descending
thrust_active               = False     # Track when thrust is active for plots
stable_orbit_time           = None      # Track the time when stable orbit is reached
slowdown_start_time         = None
thrust_start_time           = None
stabilization_start_time    = None
aerobrake                   = True      # Use aerobraking for the main simulations
#endregion

#region Atmosphere
atmo_height                 = 1375                                              # Height of the atmosphere boundary in [km]
def atmospheric_density(r_titan_spacecraft):
    height                  = r_titan_spacecraft - R_titan                      # Altitude above Titan's surface in km
    atmo_density            = 5.38 * 1e9                                        # Titan's atmospheric density in [kg/km^3]
    atmo_density_boundary   = 1e-2                                              # Density at 1200 km altitude
    H                       = atmo_height / np.log(atmo_density / atmo_density_boundary)   # Scale height
    return atmo_density * np.exp(-height / H) if height <= atmo_height else 0
#endregion

#region Mission Phases
def handle_orbit_insertion_maneuver(r, m, mdry, t, r_thrust_descend, thrust_duration):
    global is_descending, thrust_active, orbit_insertion_maneuver, thrust_start_time

    # If thrust is active, check if we should stop thrusting
    if thrust_active:
        # Stop thrusting if the thrust duration has elapsed
        if (t - thrust_start_time) >= thrust_duration:
            orbit_insertion_maneuver = False
            thrust_active = False
            return 0.0  # Throttle fully closed
        # Stop thrusting if mass reaches dry mass
        elif m <= mdry:
            orbit_insertion_maneuver = False
            thrust_active = False
            return 0.0  # Throttle fully closed
        else:
            return 1.0  # Continue thrusting

    # If thrust is not active, check if we should start thrusting
    elif is_descending and r < r_thrust_descend and m > mdry:
        thrust_start_time = t
        thrust_active = True
        return 1.0  # Throttle fully open

    else:
        return 0.0  # No thrust

def handle_slowdown_aerobraking(r, m, mdry, apoapsis, periapsis, t, r_p_slow):
    global thrust_active, full_aerobraking, slowdown_start_time

    if r_p_slow - R_titan <= 950:
        dynamic_range = 2700
    elif 951 < r_p_slow - R_titan <= 960:
        dynamic_range = 1800
    elif 961 < r_p_slow - R_titan <= 970:
        dynamic_range = 1200
    elif 971 < r_p_slow - R_titan <= 980:
        dynamic_range = 700
    else:
        dynamic_range = 500

    if r_a < apoapsis <= (r_a + dynamic_range) and abs(r - apoapsis) < 30 and periapsis < r_p_slow and m > mdry:
        if slowdown_start_time is None:
            slowdown_start_time = t
        return 1.0  # Throttle fully open
    elif periapsis >= r_p_slow + 3:
        return 0.0  # Throttle fully closed
    elif apoapsis <= r_a and periapsis > r_p_slow:
        full_aerobraking = False
    return 1.0 if thrust_active else 0.0  # Maintain current state

def handle_orbit_stabilization(r, m, mdry, apoapsis, periapsis, t):
    global thrust_active, stable_orbit_time, stabilization_start_time
    if apoapsis <= r_a and abs(r - apoapsis) < 0.1 and periapsis < r_p_orbit and m > mdry:
        if stabilization_start_time is None:
            stabilization_start_time = t
        return 1.0  # Throttle fully open
    elif periapsis >= (r_p_orbit - 5):
        if stable_orbit_time is None:
            stable_orbit_time = t
        return 0.0  # Throttle fully closed
    return 1.0 if thrust_active else 0.0  # Maintain current state

def handle_thrust_only(r, m, mdry, apoapsis, periapsis, t):
    global stable_orbit_time, full_aerobraking
    if apoapsis >= r_a and abs(r - periapsis) <= 10 and m > mdry:
        return 1.0  # Throttle fully open
    elif apoapsis < r_a and abs(r - periapsis) <= 0.1:
        full_aerobraking = False
    return 0.0
#endregion

#region Equations of Motion
def get_titan_position(t):
    # Orbital period and other parameters
    n = 2 * np.pi / T_titan  # Mean motion (radians per second)
    M = n * t  # Mean anomaly at time t

    # Solve Kepler's equation for Eccentric Anomaly E using Newton's method
    E = M
    for _ in range(10):  # Iterate to refine E
        E = E - (E - e_titan * np.sin(E) - M) / (1 - e_titan * np.cos(E))

    # Calculate True Anomaly ν
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + e_titan) * np.sin(E / 2), np.sqrt(1 - e_titan) * np.cos(E / 2))

    # Calculate Titan's distance from Saturn at time t
    r_titan_saturn = a_titan * (1 - e_titan**2) / (1 + e_titan * np.cos(true_anomaly))

    # Titan's position in Cartesian coordinates
    x_titan = r_titan_saturn * np.cos(true_anomaly)
    y_titan = r_titan_saturn * np.sin(true_anomaly)
    v_titan = np.sqrt(mu_saturn * (2 / r_titan_saturn - 1 / a_titan))

    return x_titan, y_titan, v_titan

def get_enceladus_position(t):
    # Orbital period and mean motion
    n = 2 * np.pi / T_enceladus  # Mean motion (radians per second)
    M = n * t  # Mean anomaly at time t

    # Solve Kepler's equation for eccentric anomaly (E) using Newton's method
    E = M
    for _ in range(10):  # Iterate to refine E
        E = E - (E - e_enceladus * np.sin(E) - M) / (1 - e_enceladus * np.cos(E))

    # Calculate True Anomaly ν
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + e_enceladus) * np.sin(E / 2), np.sqrt(1 - e_enceladus) * np.cos(E / 2))

    # Distance from Saturn to Enceladus
    r_enceladus_saturn = a_enceladus * (1 - e_enceladus**2) / (1 + e_enceladus * np.cos(true_anomaly))

    # Enceladus' position in Cartesian coordinates
    x_enceladus = r_enceladus_saturn * np.cos(true_anomaly)
    y_enceladus = r_enceladus_saturn * np.sin(true_anomaly)
    v_enceladus = np.sqrt(mu_saturn * (2 / r_enceladus_saturn - 1 / a_enceladus))

    return x_enceladus, y_enceladus, v_enceladus

def get_initial_spacecraft_state(v_inf, dist_saturn_titan, R_SOI_saturn, R_titan, mu_titan):
    #  0.15 NOT UP TO DATE
    if v_inf <= 0.10:
        b_i = 53020e3 - dist_saturn_titan
    elif v_inf <= 0.15:
        b_i = 47050e3 - dist_saturn_titan
    elif v_inf <= 0.20:
        b_i = 37080e3 - dist_saturn_titan
    elif v_inf <= 0.30:
        b_i = 27000e3 - dist_saturn_titan
    elif v_inf <= 0.42:
        b_i = 20200e3 - dist_saturn_titan
    elif v_inf <= 0.45:
        b_i = 19000e3 - dist_saturn_titan
    elif v_inf <= 0.55:
        b_i = 15852e3 - dist_saturn_titan
    elif v_inf <= 0.65:
        b_i = 13610e3 - dist_saturn_titan
    elif v_inf <= 0.75:
        b_i = 11930e3 - dist_saturn_titan
    elif v_inf <= 0.85:
        b_i = 10630e3 - dist_saturn_titan
    elif v_inf <= 0.88:
        b_i = 10295e3 - dist_saturn_titan
    elif v_inf <= 0.90:
        b_i = 10080e3 - dist_saturn_titan
    elif v_inf <= 1.00:
        b_i = 9140e3 - dist_saturn_titan
    else:
        b_i = R_titan * np.sqrt(2 * mu_titan / (R_titan * v_inf**2) + 1) + 5500000 / (v_inf**1.1)

    b = b_i

    y_0 = -np.sqrt(R_SOI_saturn**2 - (b + dist_saturn_titan)**2)
    x_0 = b + dist_saturn_titan

    # Calculate the radial distance and angle in polar coordinates
    r_0 = np.sqrt(x_0**2 + y_0**2)
    phi_0 = np.arctan2(y_0, x_0)

    # Calculate initial velocities
    rho_r_0 = v_inf * np.cos(phi_0 - np.pi / 2)
    rho_phi_0 = -v_inf * np.sin(phi_0 - np.pi / 2) / r_0

    return x_0, y_0, r_0, phi_0, rho_r_0, rho_phi_0

def eom(t, state, thrust, ceff, mdry, r_thrust_descend, thrust_duration, r_p_slow):
    global is_descending, thrust_active, orbit_insertion_maneuver, full_aerobraking

    r, phi, rhor, rhophi, m = state

    # Get Titan's position
    x_titan, y_titan, v_titan = get_titan_position(t)

    # Spacecraft's position
    x_spacecraft = r * np.cos(phi)
    y_spacecraft = r * np.sin(phi)
    dx = x_spacecraft - x_titan
    dy = y_spacecraft - y_titan
    r_titan_spacecraft = np.sqrt(dx**2 + dy**2)

    # Calculate spacecraft velocity and flight path angle
    v = np.sqrt(rhor**2 + (r * rhophi)**2)
    alpha = np.arctan2(r * rhophi, rhor)
    E = v**2 / 2 - mu_titan / r
    h = r**2 * rhophi
    a = -mu_titan / (2 * E)
    e = np.sqrt(1 + (2 * E * h**2) / (mu_titan**2))
    apoapsis = a * (1 + e)
    periapsis = a * (1 - e)
    is_descending = rhor < 0
    throttle = 0.0
    atmo_density = atmospheric_density(r)
    D = max(0.5 * c_d * atmo_density * A * v ** 2, 0) if atmo_density > 0 else 0

    # Check if the spacecraft is inside Titan's SOI
    if r_titan_spacecraft <= R_SOI_titan:
        # Inside Titan's SOI: Gravitational force towards Titan
        accel_titan = -mu_titan / r_titan_spacecraft**2

        # Vector components
        titan_dx = (x_spacecraft - x_titan) / r_titan_spacecraft
        titan_dy = (y_spacecraft - y_titan) / r_titan_spacecraft

        # Gravitational acceleration in polar coordinates
        gravitational_acceleration_r_titan = accel_titan * (titan_dx * np.cos(phi) + titan_dy * np.sin(phi))
        gravitational_acceleration_phi_titan = accel_titan * (-titan_dx * np.sin(phi) + titan_dy * np.cos(phi))

        # Saturn's gravitational influence
        x_saturn = 0
        y_saturn = 0
        r_saturn_spacecraft = np.sqrt(x_spacecraft**2 + y_spacecraft**2)

        # Gravitational force towards Saturn
        accel_saturn = -mu_saturn / r_saturn_spacecraft**2

        # Radial and angular components of Saturn's gravity
        gravitational_acceleration_r_saturn = accel_saturn * (x_spacecraft / r_saturn_spacecraft * np.cos(phi) + y_spacecraft / r_saturn_spacecraft * np.sin(phi))
        gravitational_acceleration_phi_saturn = accel_saturn * (-x_spacecraft / r_saturn_spacecraft * np.sin(phi) + y_spacecraft / r_saturn_spacecraft * np.cos(phi))

        # Combine both Titan and Saturn's gravity
        gravitational_acceleration_r = gravitational_acceleration_r_titan + gravitational_acceleration_r_saturn
        gravitational_acceleration_phi = gravitational_acceleration_phi_titan + gravitational_acceleration_phi_saturn
    else:
        # Spacecraft position relative to Saturn
        x = r * np.cos(phi)
        y = r * np.sin(phi)

        # Vector to Saturn
        r_saturn = np.sqrt(x**2 + y**2)

        # Saturn's gravitational force magnitude
        accel_saturn = -mu_saturn / r_saturn**2

        # Normalize vector
        dx_norm = x / r_saturn
        dy_norm = y / r_saturn

        # Gravitational force components
        gravitational_acceleration_r = accel_saturn * (dx_norm * np.cos(phi) + dy_norm * np.sin(phi))
        gravitational_acceleration_phi = accel_saturn * (-dx_norm * np.sin(phi) + dy_norm * np.cos(phi))

    if not aerobrake:
        if orbit_insertion_maneuver:
            beta = alpha + np.pi
            throttle = handle_orbit_insertion_maneuver(r, m, mdry, t, r_thrust_descend, thrust_duration)
        elif not orbit_insertion_maneuver and full_aerobraking:
            beta = alpha + np.pi
            throttle = handle_thrust_only(r, m, mdry, apoapsis, periapsis, t)
        elif not orbit_insertion_maneuver and not full_aerobraking:
            beta = alpha
            throttle = handle_orbit_stabilization(r, m, mdry, apoapsis, periapsis, t)
    else:
        # Choose appropriate thrust logic
        if orbit_insertion_maneuver:
            beta = alpha + np.pi
            throttle = handle_orbit_insertion_maneuver(r, m, mdry, t, r_thrust_descend, thrust_duration)
        elif not orbit_insertion_maneuver and full_aerobraking:
            beta = alpha
            throttle = handle_slowdown_aerobraking(r, m, mdry, apoapsis, periapsis, t, r_p_slow)
        elif not orbit_insertion_maneuver and not full_aerobraking:
            beta = alpha
            throttle = handle_orbit_stabilization(r, m, mdry, apoapsis, periapsis, t)

    # Update equations of motion
    dr = rhor
    dphi = rhophi
    drhor = r * rhophi**2 + gravitational_acceleration_r + throttle * thrust / m * np.cos(beta) - D / m * np.cos(alpha)
    drhophi = (-2 * rhor * rhophi + gravitational_acceleration_phi + throttle * thrust / m * np.sin(beta) - D / m * np.sin(alpha)) / r
    dm = -throttle * thrust / ceff

    thrust_active = throttle > 0.0

    return [dr, dphi, drhor, drhophi, dm]
#endregion

#region Simulation termination conditions
def planet_crash(t, state, *args):
    r, phi, rhor, rhophi, m = state
    altitude = r - R_titan
    return altitude - 20
planet_crash.terminal   = True
planet_crash.direction  = -1

def stable_orbit_reached(t, state, *args):
    global stable_orbit_time
    return t - stable_orbit_time - 86400 if stable_orbit_time is not None else -1
stable_orbit_reached.terminal = True
stable_orbit_reached.direction = 1
#endregion

#region Simulate Trajectory
class Trajectory:
    pass

def simulate_trajectory(v_inf, r_thrust_descend, r_p_slow, thrust_duration, short_run=True,
                        max_step_large=max_step_non_critical, max_step_small=max_step_critical,
                        launch_time=0, critical_distance=critical_distance_titan_sim,
                        stop_after_first_crossing=False, radial_tolerance_km=5):

    global orbit_insertion_maneuver, is_descending, thrust_active, full_aerobraking
    global stable_orbit_time, slowdown_start_time, stabilization_start_time, thrust_start_time
    inside_critical_zone = False

    # Reset mission phase variables
    orbit_insertion_maneuver = True
    full_aerobraking = True
    is_descending = True
    thrust_active = False
    stable_orbit_time = None
    slowdown_start_time = None
    stabilization_start_time = None
    thrust_start_time = None
    crossing_count = 0  # local variable for counting
    crossing_in_progress = False

    # Initial conditions
    x_0, y_0, r_0, phi_0, rho_r_0, rho_phi_0 = get_initial_spacecraft_state(v_inf, dist_saturn_titan, R_SOI_saturn, R_titan, mu_titan)
    init_val = [r_0, phi_0, rho_r_0, rho_phi_0, m_0]
    p = (thrust, ceff, m_0 - m_p, r_thrust_descend, thrust_duration, r_p_slow)

    # Define event functions

    def first_crossing_event(t, state, *args):
        # state = [r, phi, rhor, rhophi, m]
        r, phi, rhor, rhophi, m = state
        r_p_titan = a_titan * (1 - e_titan)

        if r < (1221870.0-50000):
            # The event function should return a value that crosses zero at the crossing
            return r - r_p_titan
        else:
            # If not descending, return a positive value to avoid triggering the event
            return 1.0

    first_crossing_event.terminal = True
    first_crossing_event.direction = 0


    def enter_critical_zone(t, state, *args):
        global thrust_active
        r, phi, rhor, _, _ = state
        x_spacecraft = r * np.cos(phi)
        y_spacecraft = r * np.sin(phi)
        x_titan, y_titan, _ = get_titan_position(t)
        dx = x_spacecraft - x_titan
        dy = y_spacecraft - y_titan
        r_titan_spacecraft = np.sqrt(dx**2 + dy**2)
        distance_to_titan_critical = r_titan_spacecraft - critical_distance
        distance_to_saturn_critical = r - critical_distance_saturn_sim
        r_p_titan = a_titan * (1 - e_titan)
        r_a_titan = a_titan * (1 + e_titan)
        delta_r = 5000
        in_titan_orbit_range = (r >= r_p_titan - delta_r) and (r <= (r_a_titan + delta_r))

        if inside_critical_zone:
            return 1
        else:
            if distance_to_titan_critical <= 0 or in_titan_orbit_range:
                return -1
            elif distance_to_saturn_critical <= 0 and orbit_insertion_maneuver:
                return -1
            else:
                return 1

    enter_critical_zone.terminal = True
    enter_critical_zone.direction = -1

    def exit_critical_zone(t, state, *args):
        global thrust_active
        r, phi, rhor, _, _ = state
        x_spacecraft = r * np.cos(phi)
        y_spacecraft = r * np.sin(phi)
        x_titan, y_titan, _ = get_titan_position(t)
        dx = x_spacecraft - x_titan
        dy = y_spacecraft - y_titan
        r_titan_spacecraft = np.sqrt(dx**2 + dy**2)
        distance_to_titan_critical = r_titan_spacecraft - critical_distance
        distance_to_saturn_critical = r - critical_distance_saturn_sim
        r_p_titan = a_titan * (1 - e_titan)
        r_a_titan = a_titan * (1 + e_titan)
        delta_r = 5000
        in_titan_orbit_range = (r >= r_p_titan - delta_r) and (r <= (r_a_titan + delta_r))

        if not inside_critical_zone:
            return -1
        else:
            if distance_to_titan_critical > 0 and distance_to_saturn_critical > 0 and not in_titan_orbit_range:
                return 1
            else:
                return -1

    exit_critical_zone.terminal = True
    exit_critical_zone.direction = 1

    def multiple_crossings_event(t, state, *args):
        nonlocal crossing_count, crossing_in_progress

        r, phi, rhor, rhophi, m = state
        x_titan, y_titan, _ = get_titan_position(t)
        r_titan_at_phi = np.sqrt(x_titan**2 + y_titan**2)

        radial_diff = abs(r - r_titan_at_phi)

        # If we're "inside" Titan's orbit tolerance:
        if radial_diff <= radial_tolerance_km:
            # Check if this is a brand-new crossing
            if not crossing_in_progress:
                crossing_in_progress = True
                crossing_count += 1
                #print(f"+++ CROSSING #{crossing_count} at t={t/86400:.3f} days +++")

                # If crossing_count >= max_crossings, stop the integration
                if crossing_count >= max_crossings:
                    return 0.0  # triggers the solver to end
        else:
            # We are outside the tolerance region => reset the flag
            crossing_in_progress = False

        # If we haven't triggered a crossing, or we are already "in" the crossing,
        # return a positive non-zero so there's no zero root
        return 1.0

    multiple_crossings_event.terminal = True
    multiple_crossings_event.direction = 0


    # Initialize variables
    t_start = launch_time
    t_end = (tmax_short if short_run else tmax + 86400) + launch_time
    trajectory_segments = []
    max_step_current = max_step_large

    while t_start < t_end:
        t_span = (t_start, t_end)
        # Choose events based on the current state
        if not inside_critical_zone:
            if stop_after_first_crossing:
                event_list = [planet_crash, stable_orbit_reached, enter_critical_zone, first_crossing_event]
            else:
                event_list = [planet_crash, stable_orbit_reached, enter_critical_zone, multiple_crossings_event]
        else:
            if stop_after_first_crossing:
                event_list = [planet_crash, stable_orbit_reached, exit_critical_zone, first_crossing_event]
            else:
                event_list = [planet_crash, stable_orbit_reached, exit_critical_zone, multiple_crossings_event]

        # Perform integration
        sol = solve_ivp(eom, t_span, init_val, args=p, method='RK45', rtol=1e-8, atol=1e-11,max_step=max_step_current, events=event_list)

        # Append this segment to the trajectory
        trajectory_segments.append(sol)

        # Check if an event occurred
        if sol.status == 1 and sol.t_events:
            # Get the event that occurred
            event_occurred = None
            for idx, t_events in enumerate(sol.t_events):
                if t_events.size > 0:
                    event_occurred = event_list[idx]
                    break

            if event_occurred == enter_critical_zone:
                # Switch to small max_step
                max_step_current = max_step_small
                inside_critical_zone = True
                #print(f"    Entering critical zone at t = {sol.t[-1]/86400:.4f} days")
            elif event_occurred == exit_critical_zone:
                # Switch to large max_step
                max_step_current = max_step_large
                inside_critical_zone = False
                #print(f"    Exiting critical zone at t = {sol.t[-1]/86400:.4f} days")
            elif event_occurred in [planet_crash, stable_orbit_reached, first_crossing_event, multiple_crossings_event]:
                # Terminate integration
                break
        else:
            # No event occurred; end integration
            break

        init_val = sol.y[:, -1]
        t_start = sol.t[-1] + 1e-6

    # Combine trajectory segments
    t_combined = np.hstack([seg.t for seg in trajectory_segments])
    y_combined = np.hstack([seg.y for seg in trajectory_segments])

    # Create combined trajectory object
    trajectory = Trajectory()
    trajectory.t = t_combined
    trajectory.y = y_combined
    trajectory.status = trajectory_segments[-1].status
    trajectory.t_events = [event for seg in trajectory_segments for event in seg.t_events]

    # Extract radial distance and mass
    r = trajectory.y[0, :]
    m = trajectory.y[4, :]

    # Find local minima (periapsis) and maxima (apoapsis)
    local_minima_indices = argrelextrema(r, np.less)[0]
    local_maxima_indices = argrelextrema(r, np.greater)[0]

    if len(local_minima_indices) >= 2 and len(local_maxima_indices) >= 1:
        periapsis_2nd_orbit = r[local_minima_indices[1]]
        mass_at_2nd_periapsis = m[local_minima_indices[1]]
        apoapsis_1st_orbit = r[local_maxima_indices[0]]

        return periapsis_2nd_orbit, apoapsis_1st_orbit, trajectory, mass_at_2nd_periapsis, m[-1]
    else:
        return None, None, trajectory, None, m[-1]
#endregion

#region Launch Time
def calculate_launch_delay_iterative(initial_launch_time, r_p, v_inf, radial_tolerance_km=5, initial_delay_time=None):

    r_thrust_descend = R_saturn + r_thrust_descending

    rp_matching_tol = 100
    if initial_delay_time is None:
        # Instead of initial_delays, we go through r_p_configurations[v_inf]
        for cfg in r_p_configurations.get(v_inf, []):
            if abs(cfg["first_crossing_rp"] - r_p) <= rp_matching_tol:
                initial_delay_time = cfg.get("initial_delay", 0)
                break
        else:
            # Fallback if none matched
            initial_delay_time = 1.1740749647 * 86400

    # Now proceed with the rest of your iterative logic
    delay_time = initial_delay_time

    iteration = 1
    max_iterations = 21

    while iteration < max_iterations:
        # "Stop after first crossing" only
        _, _, trajectory, _, _ = simulate_trajectory(
            v_inf, 
            r_thrust_descend, 
            r_p, 
            thrust_duration=0, 
            short_run=True, 
            launch_time=delay_time, 
            critical_distance=critical_distance_titan_sim,
            stop_after_first_crossing=True, 
            radial_tolerance_km=radial_tolerance_km
        )

        crossings_in_iteration = []
        crossing_found = False
        within_tolerance = False

        for i in range(1, len(trajectory.y[0, :])):
            # === Position Spacecraft/Titan ===
            x_titan, y_titan, v_titan = get_titan_position(trajectory.t[i])  # Titan's position
            r_spacecraft = trajectory.y[0, i]  # Spacecraft radial position
            phi_spacecraft = trajectory.y[1, i]  # Spacecraft angular position
            rhor = trajectory.y[2, i]  # Extract radial velocity component rhor
            x_spacecraft = r_spacecraft * np.cos(phi_spacecraft)  # Spacecraft's position in Cartesian coordinates
            y_spacecraft = r_spacecraft * np.sin(phi_spacecraft)

            # === Distance Calculations ===
            distance_to_titan = np.sqrt((x_spacecraft - x_titan)**2 + (y_spacecraft - y_titan)**2)  # Distance between spacecraft and Titan
            distance_to_titan_surface = distance_to_titan - R_titan  # Altitude above Titan's surface
            r_titan_at_phi = np.sqrt(x_titan**2 + y_titan**2)  # Titan's radial distance from Saturn

            # Calculate the difference in radial distances
            radial_difference = abs(r_spacecraft - r_titan_at_phi)

            # Check if the spacecraft is within radial tolerance of Titan's orbit
            if radial_difference <= radial_tolerance_km:
                if not within_tolerance:
                    # The spacecraft has just entered the tolerance region
                    within_tolerance = True
                    crossing_found = True
                    time_at_orbit = trajectory.t[i]
                    phi_spacecraft_at_orbit = phi_spacecraft
                    x_titan_at_orbit = x_titan
                    y_titan_at_orbit = y_titan
                    v_titan_at_orbit = v_titan
                    distance_to_titan_surface_at_orbit = distance_to_titan_surface
                    crossings_in_iteration.append((i, time_at_orbit, phi_spacecraft_at_orbit, x_titan_at_orbit,
                                                   y_titan_at_orbit, v_titan_at_orbit, distance_to_titan_surface_at_orbit))
                # Else, we're already within tolerance, do not record again
            else:
                # The spacecraft is outside the tolerance region
                if within_tolerance:
                    # The spacecraft has just exited the tolerance region
                    within_tolerance = False

        if not crossing_found:
            print(f"WARNING: The spacecraft did not cross Titan's orbit during iteration {iteration}.")
            return delay_time, None, None, None, None  # Return current delay time if no crossing is found

        # Sort the crossings by time to ensure the first crossing is earliest
        crossings_in_iteration.sort(key=lambda x: x[1])  # x[1] is time_at_orbit

        # Print angular difference and distance to Titan for all crossings
        for idx, crossing in enumerate(crossings_in_iteration):
            ci, ctime_at_orbit, cphi_spacecraft_at_orbit, cx_titan_at_orbit, \
            cy_titan_at_orbit, cv_titan_at_orbit, cdistance_to_titan_surface_at_orbit = crossing

            cphi_titan_at_orbit = np.arctan2(cy_titan_at_orbit, cx_titan_at_orbit)
            cdelta_phi = (cphi_spacecraft_at_orbit - cphi_titan_at_orbit + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
            cdelta_phi_deg = np.degrees(cdelta_phi)  # Convert delta_phi to degrees

        # Use the first crossing for delay time adjustment
        crossing_for_adjustment = crossings_in_iteration[0]
        i, time_at_orbit, phi_spacecraft_at_orbit, x_titan_at_orbit, \
        y_titan_at_orbit, v_titan_at_orbit, distance_to_titan_surface_at_orbit = crossing_for_adjustment

        phi_titan_at_orbit = np.arctan2(y_titan_at_orbit, x_titan_at_orbit)
        delta_phi = (phi_spacecraft_at_orbit - phi_titan_at_orbit + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
        delta_phi_deg = np.degrees(delta_phi)  # Convert to degrees

        # Check if spacecraft is at the desired distance at the first crossing
        distance_error = distance_to_titan_surface_at_orbit + R_titan - r_p
        if abs(distance_error) <= tolerance:
            #print(f"          [{v_inf:.2f} km/s] OPTIMAL DELAY: {delay_time / 86400:.12f} days")
            return delay_time, None, None, None, None

        # Adjust delay time based on the angular difference and distance error

        if abs(delta_phi_deg) > 1:  # Rough adjustments for large angular differences
            r_current = np.sqrt(x_titan_at_orbit**2 + y_titan_at_orbit**2)
            omega_avg = v_titan_at_orbit / r_current  # Angular velocity of Titan
            if v_inf <= 1:
                delay_adjustment = delta_phi / omega_avg * 0.946  # Adjust based on angular velocity
            else:
                delay_adjustment = delta_phi / omega_avg * 0.94284  # Adjust based on angular velocity
            delay_time += delay_adjustment  # Apply delay adjustment
            print(f"                    [{v_inf:.2f} km/s] Delay-Iteration {iteration} for r_p = {r_p - R_titan:.6f} km: Spacecraft is {delta_phi_deg:.2f} degrees away from crossing")
            #print(f"  Adjusting delay by {delay_adjustment / 3600:.2f} hours to {delay_time/86400:.10f} days.")
        else:  # Fine adjustments for small angular differences
            if delta_phi_deg > 0:
                if abs(distance_error) >= 1000:
                    delay_adjustment = abs(distance_error) * 0.178 * (1/v_inf)**0.012
                elif abs(distance_error) >= 200:
                    delay_adjustment = abs(distance_error) * 0.174 * (1/v_inf)**0.019 #Increase **0.02 to decrease the delay adjustment
                else:
                    if v_inf <= 0.25:
                        delay_adjustment = abs(distance_error) * 0.179    #Increase **0.05 to  to decrease adjustment and increase gap between v_infs
                    elif v_inf <= 0.50:
                        delay_adjustment = abs(distance_error) * 0.1827 #Increase **0.05 to  to decrease adjustment and increase gap between v_infs
                    elif v_inf <= 0.75:
                        delay_adjustment = abs(distance_error) * 0.191 #Increase **0.05 to  to decrease adjustment and increase gap between v_infs
                    elif v_inf <= 1.0:
                        delay_adjustment = abs(distance_error) * 0.192 #Increase **0.05 to  to decrease adjustment and increase gap between v_infs
                    else:
                        delay_adjustment = abs(distance_error) * 0.174 * v_inf**0.0213
            else:
                if abs(distance_error) >= 10000:
                    delay_adjustment = abs(distance_error) * 0.2222222
                elif abs(distance_error) >= R_titan and abs(distance_error) < 10000:
                    delay_adjustment = abs(distance_error) * 0.33333333
                else:
                    delay_adjustment = abs(distance_error + R_titan) / 2.7

            # Adjust delay based on angular difference and distance to Titan
            if delta_phi_deg > 0:  # Spacecraft is ahead of Titan
                if distance_to_titan_surface_at_orbit + R_titan < r_p:
                    print(f"                    [{v_inf:.2f} km/s] Delay-Iteration {iteration} for r_p = {r_p - R_titan:.6f} km: Spacecraft is {abs(distance_error):.8f} km too close to Titan's surface.")
                    #print(f"  Decreasing delay by {delay_adjustment / 60:.2f} minutes.")
                    delay_time -= delay_adjustment
                else:
                    print(f"                    [{v_inf:.2f} km/s] Delay-Iteration {iteration} for r_p = {r_p - R_titan:.6f} km: Spacecraft is {abs(distance_error):.8f} km too far ahead of Titan.")
                    #print(f"  Increasing delay by {delay_adjustment / 60:.2f} minutes.")
                    delay_time += delay_adjustment
            else:  # Spacecraft is behind Titan
                print(f"                    [{v_inf:.2f} km/s] Delay-Iteration {iteration} for r_p = {r_p - R_titan:.6f} km: Spacecraft is behind Titan ({distance_to_titan_surface_at_orbit:.8f} km), error: {distance_error:.6f} km.")
                #print(f"  Decreasing delay by {delay_adjustment / 60:.2f} minutes.")
                delay_time -= delay_adjustment

        iteration += 1

    # If we reach here, we have either converged or hit the maximum iterations
    print(f"Final launch delay for [v_inf = {v_inf:.2f} km/s] r_p = {r_p - R_titan:.4f} km: {delay_time / 86400:.8f} days")
#endregion

#region Adjust Aerobrake Altitude
def adjust_r_p_to_match_crossing(v_inf, target_crossing_number, target_crossing_distance, r_p_initial, tolerance_angle, tolerance_distance):
    current_r_p = r_p_initial
    iteration = 1
    radial_tolerance_km = 5
    max_iterations = 16

    if target_crossing_number == 3:
        if v_inf <= 0.45:
            k_angle_base = 2.3 * v_inf ** 0.21
            k_distance_base = 0.00012 * v_inf ** 0.1
        elif v_inf <= 1:
            k_angle_base = 2.4 * v_inf ** 0.21
            k_distance_base = 0.000155 * v_inf ** 0.1
        else:
            k_angle_base = 0.8 * (1 / v_inf) ** 1.4
            k_distance_base = 0.000084 * (1 / v_inf) ** 1.72
    elif target_crossing_number == 4:
        if v_inf < 1:
            k_angle_base = 1.8 * v_inf ** 0.2
            k_distance_base = 0.00017 * v_inf ** 0.1
        else:
            k_angle_base = 0.8 * (1 / v_inf) ** 1.4
            k_distance_base = 0.000084 * (1 / v_inf) ** 1.72
    elif target_crossing_number == 5:
        if v_inf <= 0.25:
            k_angle_base = -0.0015 * v_inf ** 0.4
            k_distance_base = -0.000000065 * v_inf ** 0.8
        elif v_inf <= 1:
            k_angle_base = -0.00117 * v_inf ** 0.4
            k_distance_base = -0.00000012 * v_inf ** 0.8
        else:
            k_angle_base = -0.0004 * (1 / v_inf) ** 1.7
            k_distance_base = -0.000000043 * (1 / v_inf) ** 1.75
    elif target_crossing_number == 6:
        if v_inf <= 1:
            k_angle_base = -0.00117 * v_inf ** 0.4
            k_distance_base = -0.000000073 * v_inf ** 0.8
        else:
            k_angle_base = -0.0004 * (1 / v_inf) ** 1.7
            k_distance_base = -0.000000043 * (1 / v_inf) ** 1.75
    elif target_crossing_number == 8:
        if v_inf <= 1:
            k_angle_base = 0.0000045 * v_inf ** 0.4
            k_distance_base = 0.0000000001 * v_inf ** 0.8
        else:
            k_angle_base = 0.0004 * (1 / v_inf) ** 1.7
            k_distance_base = 0.000000043 * (1 / v_inf) ** 1.75
    else:
        k_angle_base = 10 * (1 / v_inf) ** 0.7
        k_distance_base = 0.00015 * (1 / v_inf) ** 1.4

    while iteration < max_iterations:
        # Step 1: Calculate the optimal delay for the first crossing at this r_p
        delay_time, _, _, _, _ = calculate_launch_delay_iterative(0, current_r_p, v_inf)

        r_thrust_descend = R_saturn + r_thrust_descending

        # Step 2: Simulate full trajectory to find all crossings
        _, _, trajectory, _, _ = simulate_trajectory(
            v_inf, r_thrust_descend, current_r_p, thrust_duration=0,
            short_run=True, launch_time=delay_time,
            critical_distance=critical_distance_titan_sim,
            stop_after_first_crossing=False
        )

        within_tolerance = False
        crossings_found = []
        first_apoapsis_found = False
        first_apoapsis_value = None

        r = trajectory.y[0, :]
        phi = trajectory.y[1, :]
        rhor = trajectory.y[2, :]

        for i in range(len(trajectory.t)):
            # Detect periapsis by finding where rhor switches from negative to positive
            if i > 0 and rhor[i - 1] < 0 and rhor[i] > 0:
                if not first_apoapsis_found:
                    # Begin tracking apoapsis after the first periapsis is crossed
                    periapsis_found = r[i]

            # Detect apoapsis by finding where rhor switches from positive to negative
            if i > 0 and rhor[i - 1] > 0 and rhor[i] < 0:
                if not first_apoapsis_found:
                    # Track the first apoapsis
                    first_apoapsis_found = True
                    first_apoapsis_value = r[i]

            # Check for radial tolerance crossings
            x_titan, y_titan, v_titan = get_titan_position(trajectory.t[i])
            r_spacecraft = r[i]
            phi_spacecraft = phi[i]
            x_spacecraft = r_spacecraft * np.cos(phi_spacecraft)
            y_spacecraft = r_spacecraft * np.sin(phi_spacecraft)
            distance_to_titan = np.sqrt((x_spacecraft - x_titan) ** 2 + (y_spacecraft - y_titan) ** 2)
            r_titan_at_phi = np.sqrt(x_titan ** 2 + y_titan ** 2)
            radial_difference = abs(r_spacecraft - r_titan_at_phi)

            if radial_difference <= radial_tolerance_km:
                if not within_tolerance:
                    within_tolerance = True
                    phi_titan = np.arctan2(y_titan, x_titan)
                    delta_phi = (phi_spacecraft - phi_titan + np.pi) % (2 * np.pi) - np.pi
                    delta_phi_deg = np.degrees(delta_phi)
                    crossings_found.append((trajectory.t[i], distance_to_titan, delta_phi_deg))
            else:
                if within_tolerance:
                    within_tolerance = False

        #for idx, crossing in enumerate(crossings_found, 1):
         #   crossing_time, crossing_distance, crossing_angle_deg = crossing
          #  print(f"  Crossing {idx}: Time = {crossing_time/86400:.4f} days")

        # Check if we have enough crossings to reach the desired target crossing number
        if len(crossings_found) < target_crossing_number:
            print(f"[{v_inf:.2f} km/s] NOT ENOUGH CROSSINGS. FOUND: {len(crossings_found)}, REQUIRED: {target_crossing_number}")
            break

        # Get the crossing data for the target crossing
        _, crossing_distance, crossing_angle_deg = crossings_found[target_crossing_number - 1]

        # 4. Check angle and distance conditions
        if abs(crossing_angle_deg) > tolerance_angle or crossing_angle_deg < 0:
            # Adjust r_p based on angle
            if crossing_angle_deg < 0 and abs(crossing_angle_deg) < 1:
                delta_r_p_angle = 7 * k_angle_base * crossing_angle_deg
            else:
                delta_r_p_angle = k_angle_base * crossing_angle_deg
            print(f"[{v_inf:.2f} km/s] Error: {crossing_angle_deg:.2f} deg | CROSSING #{target_crossing_number} | R_P {current_r_p - R_titan:.8f} -> {current_r_p + delta_r_p_angle - R_titan:.8f} km | DELAY: {delay_time/86400:.12f} days | Iteration {iteration}")
            current_r_p += delta_r_p_angle
        else:
            # Angle is within tolerance, now adjust based on distance
            distance_error = crossing_distance - target_crossing_distance
            if abs(distance_error) <= tolerance_distance:
                # Build a string for all crossing distances up to target_crossing_number
                crossing_distances_str = " | ".join(
                    [f"C{i+1}: {(crossings_found[i][1] - R_titan):.8f} km" 
                     for i in range(target_crossing_number)]
                )

                print(f"ITERATION DONE FOR [{v_inf:.2f} km/s] | {crossing_distances_str} | FIRST APOAPSIS: {(first_apoapsis_value - R_titan):.8f} km | DELAY: {delay_time/86400:.12f} days")
                return current_r_p, delay_time, trajectory
            else:
                if crossing_distance - R_titan > atmo_height: # Outside Atmosphere
                    delta_r_p_distance = k_distance_base / 1.3 * distance_error
                else:                                         # Within Atmosphere
                    delta_r_p_distance = k_distance_base * distance_error
                print(f"[{v_inf:.2f} km/s] Error: {distance_error:.2f} km | CROSSING #{target_crossing_number} at {crossing_distance-R_titan:.8f} km| R_P {current_r_p - R_titan:.8f} -> {current_r_p + delta_r_p_distance - R_titan:.8f} km | DELAY: {delay_time/86400:.12f} days | Iteration {iteration}")
                current_r_p += delta_r_p_distance

        iteration += 1

    print(f"[{v_inf:.2f} km/s] Did not converge within {max_iterations-1} iterations for crossing #{target_crossing_number}.")
    return current_r_p, delay_time, trajectory

#endregion

r_p_configurations = {
    #region Unused Configs
    0.10: [
        {"first_crossing_rp":  R_titan + 1950.122826, "second_crossing_rp": R_titan + 2627.824652, "initial_delay": 86400 * 5.451417269225}, 
        {"first_crossing_rp":  R_titan +  491.385979, "second_crossing_rp": R_titan + 2353.814730, "initial_delay": 86400 * 5.454396422168} 
    ],
    0.15: [
        {"first_crossing_rp": R_titan + 1250.620257, "second_crossing_rp": R_titan + 750.139487, "initial_delay": 86400 * 11.867917191177}
    ],
    0.20: [
        {"first_crossing_rp":  R_titan + 2176.936029, "second_crossing_rp": R_titan + 3000.000000, "initial_delay": 86400 * -0.504537215623}, #REDO CROSSING 3 BECAUSE IT GOES NEGATIVE FOR CROSSING 6
        {"first_crossing_rp":  R_titan +  826.195800, "second_crossing_rp": R_titan + 559.915618, "initial_delay": 86400 * -0.501791489291}
    ],
    0.30: [
        {"first_crossing_rp":  R_titan + 2120.414171, "second_crossing_rp": R_titan +  324.866064, "initial_delay": 86400 * 6.238732764069},
        {"first_crossing_rp":  R_titan + 1325.540949, "second_crossing_rp": R_titan +  780.706233, "initial_delay": 86400 * 6.240348184963},
#        {"first_crossing_rp":  R_titan +  643.894230, "second_crossing_rp": R_titan + 1856.697498, "initial_delay": 86400 * 8.064835974913}
    ],
    #endregion
    0.42: [
#        {"first_crossing_rp":  R_titan + 2569.289913, "second_crossing_rp": R_titan + 2863.831865, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 9.439745602090},
#        {"first_crossing_rp":  R_titan + 1693.622893, "second_crossing_rp": R_titan + 4246.844781, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 9.441527453064},
#        {"first_crossing_rp":  R_titan +  770.330283, "second_crossing_rp": R_titan + 8170.509343, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 9.443422828826}
    ],
    0.45: [
#        {"first_crossing_rp":  R_titan + 2652.746800, "second_crossing_rp": R_titan + 2974.838282, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 8.061092166875},
#        {"first_crossing_rp":  R_titan + 1762.260716, "second_crossing_rp": R_titan + 4387.813762, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 8.062905845410},
 #       {"first_crossing_rp":  R_titan +  823.131133, "second_crossing_rp": R_titan + 8388.799456, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 8.064837600599}
    ],
    0.55: [
        {"first_crossing_rp":  R_titan + 2863.767912, "second_crossing_rp": R_titan +  608.678103, "initial_delay": 86400 * 3.962741393774},
        {"first_crossing_rp":  R_titan + 1936.737545, "second_crossing_rp": R_titan + 1097.933611, "initial_delay": 86400 * 3.964635511108},
        {"first_crossing_rp":  R_titan +  956.716337, "second_crossing_rp": R_titan + 2136.137782, "initial_delay": 86400 * 3.966662314205}
    ],
    0.65: [
        {"first_crossing_rp":  R_titan + 3007.226839, "second_crossing_rp": R_titan +  682.616274, "initial_delay": 86400 * 9.678909190530},
        {"first_crossing_rp":  R_titan + 2056.345702, "second_crossing_rp": R_titan + 1183.324407, "initial_delay": 86400 * 9.680860982702},
    #    {"first_crossing_rp":  R_titan + 1047.906087, "second_crossing_rp": R_titan + 2199.776902, "initial_delay": 86400 * 9.682960447280} # CROSSING 6???  
    ],
    0.75: [
        {"first_crossing_rp":  R_titan + 3093.578631, "second_crossing_rp": R_titan +  744.080283, "initial_delay": 86400 * 10.559585065391},
        {"first_crossing_rp":  R_titan + 2130.211897, "second_crossing_rp": R_titan + 1254.781567, "initial_delay": 86400 * 10.561573806661},
    #    {"first_crossing_rp":  R_titan + 1104.271503, "second_crossing_rp": R_titan + 2300.964561, "initial_delay": 86400 * 10.563725906816}  #REDO CROSSING 3
    ],
    0.85: [
        {"first_crossing_rp":  R_titan + 3138.491985, "second_crossing_rp": R_titan +  798.716911, "initial_delay": 86400 * 7.501786145134},
        {"first_crossing_rp":  R_titan + 2170.559506, "second_crossing_rp": R_titan + 1319.503274, "initial_delay": 86400 * 7.503799421187},
        {"first_crossing_rp":  R_titan + 1137.796645, "second_crossing_rp": R_titan + 2405.675419, "initial_delay": 86400 * 7.505990870882}
    ],
    0.88: [
#        {"first_crossing_rp":  R_titan + 3144.275066, "second_crossing_rp": R_titan + 4036.096690, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * -0.454353656021},
#        {"first_crossing_rp":  R_titan + 2176.030078, "second_crossing_rp": R_titan + 5745.520458, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * -0.452335143869},
        {"first_crossing_rp":  R_titan + 1142.37874100, "second_crossing_rp": R_titan +  702.820843, "third_crossing_rp": R_titan +  801.25941442, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * -0.450135767589}
    ],
    0.90: [
#        {"first_crossing_rp":  R_titan + 3139.809448, "second_crossing_rp": R_titan + 4064.343400, "third_crossing_rp": R_titan +  800.460742, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 4.726928376066},
#        {"first_crossing_rp":  R_titan + 2173.111149, "second_crossing_rp": R_titan + 5782.181390, "third_crossing_rp": R_titan +  800.460742,"fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 4.728945944399},
#        {"first_crossing_rp":  R_titan + 1141.50835132, "second_crossing_rp": R_titan +  715.641293, "third_crossing_rp": R_titan +  745.919723, "fourth_crossing_rp": R_titan + 15800, "initial_delay": 86400 * 4.731146858320}
    ],
    1.00: [
        {"first_crossing_rp":  R_titan + 3117.033205, "second_crossing_rp": R_titan +  863.014531, "initial_delay": 86400 * -2.988617961564},
        {"first_crossing_rp":  R_titan + 2159.975197, "second_crossing_rp": R_titan + 1395.616912, "initial_delay": 86400 * -2.986600522068},
        {"first_crossing_rp":  R_titan + 1133.472625, "second_crossing_rp": R_titan + 2502.216865, "initial_delay": 86400 * -2.984382959336}
    ],
}

def process_v_inf(v_inf_value):
    # Check if this v_inf is in our dictionary
    if v_inf_value not in r_p_configurations:
        print(f"No configuration found for v_inf={v_inf_value}")
        return []

    config_list = r_p_configurations[v_inf_value]
    results = []

    for cfg in config_list:
        r_p_initial = cfg["first_crossing_rp"]
        r_p_second_crossing = cfg["second_crossing_rp"]
        r_p_third_crossing = cfg ["third_crossing_rp"]
        r_p_fourth_crossing = cfg ["fourth_crossing_rp"]
        r_thrust_descend = R_saturn + r_thrust_descending

        # Step 2: Adjust the spacecraft orbit for the second crossing
#        r_p_after_second, delay_time_after_second, full_trajectory = adjust_r_p_to_match_crossing(
 #           v_inf=v_inf_value,
  #          target_crossing_number=3,  # or 2, etc.
   #         target_crossing_distance=r_p_second_crossing,
    #        r_p_initial=r_p_initial,
     #       tolerance_angle=2.0,
      #      tolerance_distance = 50.0
       # )

        # Adjust for the third crossing
        r_p_after_third, delay_time_after_third, full_trajectory = adjust_r_p_to_match_crossing(
            v_inf=v_inf_value,
            target_crossing_number=5,
            target_crossing_distance=r_p_third_crossing,
            r_p_initial=r_p_initial,
            tolerance_angle=2.0,
            tolerance_distance= 10.0
        )

#        r_p_after_third = r_p_after_second
 #       delay_time_after_third = delay_time_after_second

        # If delay_time_after_third is None, set some default
        final_delay = delay_time_after_third if delay_time_after_third is not None else 0

        # Build a result dictionary from the data you already have
        result = {
            'v_inf': v_inf_value,
            'r_p': r_p_after_third,
            'r_p_third_crossing_target': r_p_third_crossing,
            'r_p_slow': r_p_after_third,
            'full_trajectory': full_trajectory,
            'launch_time': final_delay
        }

        results.append(result)
    return results

#region Multithreading & Plots
    #region Multithread
if __name__ == "__main__":
    program_start_time = tm.time()
    full_trajectory_results = []

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_v_inf, v_inf): v_inf for v_inf in v_inf_values}
        for future in as_completed(futures):
            v_inf_completed = futures[future]
            result_list = future.result()  # These are now the dictionaries from process_v_inf
            full_trajectory_results.extend(result_list)

    # Organize and sort results by v_inf, periapsis height (r_p), and thrust_duration
    results_by_v_inf = {}
    for result in full_trajectory_results:
        v_inf = result['v_inf']
        if v_inf not in results_by_v_inf:
            results_by_v_inf[v_inf] = []
        results_by_v_inf[v_inf].append(result)
    sorted_v_inf_values = sorted(results_by_v_inf.keys())

    # Track the end time of the entire program
    program_end_time = tm.time()
    program_duration = (program_end_time - program_start_time) / 60

    # Print the total duration
    print(f"\nTotal program duration: {program_duration:.2f} minutes")
    #endregion

    #region Plot_Trajectory
    if Plot_Trajectory:

        plt.rcParams["figure.figsize"] = (8, 8)

        # Create the figure and main axes
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.15, right=0.85)  # Adjust to make space for widgets

        # Ensure we have some valid trajectory data
        if full_trajectory_results:
            # Organize trajectory data by r_p, but also store v_inf
            trajectory_data = {}
            for result in full_trajectory_results:
                r_p = result['r_p']
                v_inf = result['v_inf']
                full_trajectory = result['full_trajectory']
                trajectory_data[r_p] = (v_inf, full_trajectory)

            rp_values = sorted(trajectory_data.keys())
            rp_labels = [f"v_inf={trajectory_data[rp][0]:.2f} km/s, r_p={rp - R_titan:.2f} km" for rp in rp_values]

            # ===== CREATE SCROLLABLE CHECKBOXES =====
            # Set up a scrollable widget
            rp_checkbox_ax = plt.axes([0.01, 0.15, 0.15, 0.7], frameon=False)
            rp_checkbox = CheckButtons(rp_checkbox_ax, rp_labels, [False] * len(rp_labels))
            
            # Enable scrolling for the checkboxes
            def on_scroll(event):
                # Dynamically scroll the checkboxes based on the scroll event
                current_position = rp_checkbox_ax.get_position()
                delta = 0.03 if event.button == 'up' else -0.03
                new_position = [
                    current_position.x0,
                    max(current_position.y0 + delta, 0),  # Prevent scrolling out of bounds
                    current_position.width,
                    current_position.height,
                ]
                rp_checkbox_ax.set_position(new_position)
                plt.draw()

            fig.canvas.mpl_connect('scroll_event', on_scroll)

            # Initialize variables
            trajectory_lines = {}

            # ===== SATURN ======
            saturn = plt.Circle((0, 0), R_saturn / 1000, color='yellow', fill=True, label='Saturn')
            ax.add_patch(saturn)
            Sphere_of_Influence_Saturn = plt.Circle((0, 0), R_SOI_saturn / 1000, color='black', fill=False, alpha=0.6, linestyle='--', label='Saturn SOI')
            ax.add_patch(Sphere_of_Influence_Saturn)

            # ===== TITAN & ORBIT ======
            num_points = 360*6
            titan_orbit_x = []
            titan_orbit_y = []
            time_for_orbit_titan = np.linspace(0, T_titan, num_points)
            for t_it in time_for_orbit_titan:
                x_titan, y_titan, _ = get_titan_position(t_it)
                titan_orbit_x.append(x_titan / 1000)
                titan_orbit_y.append(y_titan / 1000)
            ax.plot(titan_orbit_x, titan_orbit_y, color='salmon', linestyle='--', label='Titan Orbit') 
            titan_circle = plt.Circle((0, 0), R_titan / 1000, color='red', fill=True, label='Titan')
            ax.add_patch(titan_circle)
            titan_atmosphere = plt.Circle((0, 0), (R_titan + atmo_height) / 1000, color='salmon', fill=True, alpha=0.6, label='Atmosphere Boundary')
            titan_soi = plt.Circle((0, 0), R_SOI_titan / 1000, color='grey', fill=False, alpha=0.6, linestyle='--', label='Titan SOI')
            ax.add_patch(titan_atmosphere)
            ax.add_patch(titan_soi)

            # ===== ENCELADUS & ORBIT ======
            num_points_enc = 360*3
            enceladus_orbit_x = []
            enceladus_orbit_y = []
            time_for_orbit_enceladus = np.linspace(0, T_enceladus, num_points_enc)
            for t_enc in time_for_orbit_enceladus:
                x_enceladus, y_enceladus, _ = get_enceladus_position(t_enc)
                enceladus_orbit_x.append(x_enceladus / 1000)
                enceladus_orbit_y.append(y_enceladus / 1000)
            ax.plot(enceladus_orbit_x, enceladus_orbit_y, color='lightblue', linestyle='--', label='Enceladus Orbit')
            enceladus_circle = plt.Circle((0, 0), R_enceladus / 1000, color='lightblue', fill=True, label='Enceladus')
            ax.add_patch(enceladus_circle)
            enceladus_soi = plt.Circle((0, 0), R_SOI_enceladus / 1000, color='darkblue', fill=False, alpha=0.6, linestyle='--', label='Enceladus SOI')
            ax.add_patch(enceladus_soi)

            # ===== SPACECRAFT ======
            spacecraft_plot, = ax.plot([], [], 'bo', markersize=8, label='Spacecraft Position')

            # ===== PLOT DESIGN ======
            plot_radius = R_SOI_saturn / 1000
            ax.set_xlim(-plot_radius, plot_radius)
            ax.set_ylim(-plot_radius, plot_radius)
            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x [$10^3$ km]')
            ax.set_ylabel('y [$10^3$ km]')
            ax.legend(loc='upper right')

            max_time = max([(data[1].t[-1] / 86400) for data in trajectory_data.values()])
            time_step = 0.001
            ax_slider = plt.axes([0.25, 0.05, 0.65, 0.02], facecolor='lightgoldenrodyellow')
            time_slider = mwidgets.Slider(ax_slider, 'Time (days)', 0, max_time, valinit=0, valstep=time_step, valfmt="%.4f")

            # Update plot when time slider changes
            def update_plot(val):
                current_time = time_slider.val
                x_titan, y_titan, _ = get_titan_position(current_time * 86400)
                x_titan /= 1000
                y_titan /= 1000
                titan_circle.center = (x_titan, y_titan)
                titan_atmosphere.center = (x_titan, y_titan)
                titan_soi.center = (x_titan, y_titan)

                # --- ENCELADUS UPDATE (add this) ---
                x_enc, y_enc, _ = get_enceladus_position(current_time * 86400)
                x_enc /= 1000
                y_enc /= 1000
                enceladus_circle.center = (x_enc, y_enc)
                enceladus_soi.center = (x_enc, y_enc)

                visible_lines = [line for line in trajectory_lines.values() if line.get_visible()]
                if len(visible_lines) == 1:
                    line = visible_lines[0]
                    key = [k for k, v in trajectory_lines.items() if v == line][0]
                    v_inf_line, full_trajectory = trajectory_data[key]

                    traj_time = full_trajectory.t / 86400
                    index = np.searchsorted(traj_time, current_time)
                    if index >= len(traj_time):
                        index = -1
                    r = full_trajectory.y[0, index]
                    phi = full_trajectory.y[1, index]
                    x_spacecraft = r * np.cos(phi) / 1000
                    y_spacecraft = r * np.sin(phi) / 1000
                    spacecraft_plot.set_data([x_spacecraft], [y_spacecraft])
                else:
                    spacecraft_plot.set_data([], [])

                fig.canvas.draw_idle()

            time_slider.on_changed(update_plot)

            # Function to handle keypress events and move the slider with keys
            def on_key(event):
                if event.key == 'd':  # d key to increase slider value
                    current_val = time_slider.val
                    time_slider.set_val(min(time_slider.valmax, current_val + time_slider.valstep))
                elif event.key == 'a':  # a key to decrease slider value
                    current_val = time_slider.val
                    time_slider.set_val(max(time_slider.valmin, current_val - time_slider.valstep))
                elif event.key == 'w':  # w key to increase slider value a lot
                    current_val = time_slider.val
                    time_slider.set_val(min(time_slider.valmax, current_val + time_slider.valstep*30))
                elif event.key == 'x':  # x key to decrease slider value a lot
                    current_val = time_slider.val
                    time_slider.set_val(max(time_slider.valmin, current_val - time_slider.valstep*30))

            # Connect the keypress event to the on_key function
            fig.canvas.mpl_connect('key_press_event', on_key)

            # Update trajectory visibility when a checkbox is clicked
            def update_rp_visibility(label):
                for line in trajectory_lines.values():
                    line.set_visible(False)

                # Find which r_p corresponds to this label
                idx = rp_labels.index(label)
                selected_rp = rp_values[idx]
                v_inf_line, full_trajectory = trajectory_data[selected_rp]

                if selected_rp not in trajectory_lines:
                    r = full_trajectory.y[0, :]
                    phi = full_trajectory.y[1, :]
                    subsample_factor = 1
                    subsampled_indices = np.arange(0, len(r), subsample_factor)
                    x = (r * np.cos(phi))[subsampled_indices]
                    y = (r * np.sin(phi))[subsampled_indices]
                    line, = ax.plot(
                        x / 1000, y / 1000, linewidth=0.4,
                        label=f"v_inf={v_inf_line} km/s, r_p={(selected_rp - R_titan):.8f} km",
                        visible=True
                    )
                    trajectory_lines[selected_rp] = line
                else:
                    trajectory_lines[selected_rp].set_visible(True)

                # Update the legend
                handles = [saturn, titan_circle, enceladus_circle, Sphere_of_Influence_Saturn, titan_soi, enceladus_soi, titan_atmosphere, spacecraft_plot]
                labels = ['Saturn', 'Titan', 'Enceladus', 'Saturn SOI', 'Titan SOI', 'Enceladus SOI', ' Titan Atmosphere Boundary', 'Spacecraft Position']
                for line in trajectory_lines.values():
                    if line.get_visible():
                        handles.append(line)
                        labels.append(line.get_label())
                ax.legend(handles=handles, labels=labels, loc='upper right')
                plt.draw()

            rp_checkbox.on_clicked(update_rp_visibility)
            plt.show()
        else:
            print("No valid trajectory data available.")

    #endregion
   
    #region Plot_Values
    if Plot_Values:
        # Instead of storing results by r_p and thrust, we store them by (v_inf, r_p)
        # First, extract unique (v_inf, r_p) pairs and organize results
        config_map = {}
        for result in full_trajectory_results:
            v_inf_val = result['v_inf']
            r_p_val = result['r_p']
            config_map[(v_inf_val, r_p_val)] = result

        # Sort configurations by v_inf and then by r_p
        sorted_configs = sorted(config_map.keys(), key=lambda x: (x[0], x[1]))
        config_labels = [f"v_inf={v:.2f} km/s \nr_p={r_p - R_titan:.2f} km" for v, r_p in sorted_configs]

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Position axes for the checkboxes
        config_check_ax = plt.axes([0.01, 0.1, 0.1, 0.8], frameon=False)
        parameter_check_ax = plt.axes([0.91, 0.3, 0.08, 0.6], frameon=False)

        parameter_labels = ['Orbital Energy', 'Distance to Saturn', 'Distance to Titan', 'Velocity']
        parameter_check = CheckButtons(parameter_check_ax, parameter_labels, [False] * len(parameter_labels))

        selected_params = set()
        current_config = None

        # Create a checkbox for configurations (v_inf, r_p)
        config_check = CheckButtons(config_check_ax, config_labels, [False] * len(config_labels))

        def plot_selected_configuration(v_inf, r_p):
            global current_config
            current_config = (v_inf, r_p)

            selected_result = config_map[(v_inf, r_p)]
            full_trajectory = selected_result['full_trajectory']

            time = full_trajectory.t / 86400  # Convert time to days
            r = full_trajectory.y[0, :]
            phi = full_trajectory.y[1, :]
            rhor = full_trajectory.y[2, :]
            rhophi = full_trajectory.y[3, :]

            # Compute Titan's position for each time step
            titan_positions = [get_titan_position(t * 86400)[:2] for t in time]
            x_titan = np.array([pos[0] for pos in titan_positions])
            y_titan = np.array([pos[1] for pos in titan_positions])

            # Spacecraft position in Cartesian coordinates
            x_spacecraft = r * np.cos(phi)
            y_spacecraft = r * np.sin(phi)

            # Compute distances
            v = np.sqrt(rhor ** 2 + (r * rhophi) ** 2)  # Velocity in [km/s]
            E = v**2 / 2 - mu_saturn / r  # Orbital Energy
            distance_to_saturn = r
            r_titan_spacecraft = np.sqrt((x_spacecraft - x_titan)**2 + (y_spacecraft - y_titan)**2)
            distance_to_titan = r_titan_spacecraft - R_titan

            # Subsample data for plotting
            subsample_factor = 1
            subsampled_indices = np.arange(0, len(time), subsample_factor)
            time_sub = time[subsampled_indices]
            distance_to_titan_sub = distance_to_titan[subsampled_indices]
            distance_to_saturn_sub = distance_to_saturn[subsampled_indices]
            v_sub = v[subsampled_indices]
            E_sub = E[subsampled_indices]

            # Functions to shade atmosphere regions
            def shade_atmosphere_regions(ax, time_sub, distance_to_titan_sub, atmo_height):
                in_atmosphere = distance_to_titan_sub <= atmo_height
                start = None
                for i in range(len(time_sub)):
                    if in_atmosphere[i] and start is None:
                        start = time_sub[i]
                    elif not in_atmosphere[i] and start is not None:
                        end = time_sub[i]
                        ax.axvspan(start, end, color='salmon', alpha=0.4)
                        start = None
                if start is not None:
                    ax.axvspan(start, time_sub[-1], color='salmon', alpha=0.4)



            def update_plot(label):
                if label in selected_params:
                    selected_params.remove(label)
                else:
                    selected_params.add(label)

                if current_config is None:
                    return

                ax.clear()

                # Plot each selected parameter
                if 'Orbital Energy' in selected_params:
                    ax.plot(time_sub, E_sub, label='Orbital Energy')
                    ax.set_ylabel('Energy (MJ/kg)')
                    ax.set_title('Orbital Energy vs Time')
                    shade_atmosphere_regions(ax, time_sub, distance_to_titan_sub, atmo_height)

                if 'Distance to Saturn' in selected_params:
                    ax.plot(time_sub, distance_to_saturn_sub, color="purple", label='Distance to Saturn')
                    ax.set_ylabel('Distance to Saturn (km)')
                    ax.set_title('Distance to Saturn vs Time')
                    ax.set_yscale('log')
                    ax.axhline(y=a_titan, color='red', linestyle='--', label='Titan')
                    ax.axhline(y=236245, color='black', linestyle='--', label='12.6km/s')
                    ax.axhline(y=329146, color='black', linestyle='--', label='13.6km/s')
                    ax.axhline(y=a_enceladus, color='blue', linestyle='--', label='Enceladus')
    
                    shade_atmosphere_regions(ax, time_sub, distance_to_titan_sub, atmo_height)

                if 'Distance to Titan' in selected_params:
                    ax.plot(time_sub, distance_to_titan_sub, color="darkcyan", label='Distance to Titan')
                    ax.set_ylabel('Distance to Titan (km)')
                    ax.set_title('Distance to Titan vs Time')
                    ax.axhline(y=atmo_height, color='gray', linestyle='--', label='Atmosphere Boundary')
                    shade_atmosphere_regions(ax, time_sub, distance_to_titan_sub, atmo_height)

                if 'Velocity' in selected_params:
                    ax.plot(time_sub, v_sub, color="green", label='Velocity')
                    ax.set_ylabel('Velocity (km/s)')
                    ax.set_title('Velocity vs Time')
                    shade_atmosphere_regions(ax, time_sub, distance_to_titan_sub, atmo_height)

                ax.set_xlabel('Time (days)')
                ax.legend(loc='upper right')
                ax.grid(True)
                plt.draw()

            parameter_check.on_clicked(update_plot)

        def update_config_selection(label):
            idx = config_labels.index(label)
            v_inf_val, r_p_val = sorted_configs[idx]
            plot_selected_configuration(v_inf_val, r_p_val)

        config_check.on_clicked(update_config_selection)
        plt.show()
    #endregion

    #region Plot Atmosphere
    if Plot_Atmosphere:
        heights = np.linspace(0, atmo_height, 500)
        densities_km3 = [atmospheric_density(height + R_titan) for height in heights]
        densities_m3 = [density * 1e-9 for density in densities_km3]  # Convert from kg/km^3 to kg/m^3

        plt.figure(figsize=(10, 6))
        plt.plot(densities_m3, heights, label="Atmospheric Density")

        # Configure x-axis (logarithmic) and y-axis
        plt.xscale('log')
        plt.yscale('linear')
        plt.xlabel('Atmospheric Density (kg/m³)')
        plt.ylabel('Height (km)')
        plt.title('Atmospheric Density vs Height')

        # Set x-axis limits and ticks for every exponential step
        plt.xlim(1e-12, 1e1)
        plt.xticks([10**i for i in range(-12, 2)])  # Create ticks from 10^-12 to 10^1
        plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5)

        # Set y-axis limits and ticks for every 100 km step
        plt.ylim(0, 1400)
        plt.yticks(range(0, 1401, 100))  # Create ticks every 100 km
        plt.grid(which='major', axis='y', linestyle='--', linewidth=0.5)

    #endregion

    #region Plot v_inf vs. r_p required with apoapsis height
    if Plot_Comparison:
        # Organize data by v_inf
        grouped_data = defaultdict(list)
        for result in full_trajectory_results:
            v_inf = result['v_inf']
            r_p = result['r_p'] - R_titan  # Subtract Titan's radius for correct periapsis height
            if 'full_trajectory' in result and result['full_trajectory'] is not None:
                trajectory = result['full_trajectory']
                r = trajectory.y[0, :]
                rhor = trajectory.y[2, :]
                # Detect the first apoapsis: where rhor switches from positive to negative
                for i in range(1, len(rhor)):
                    if rhor[i - 1] > 0 and rhor[i] <= 0:
                        first_apoapsis = r[i] - R_titan  # Subtract Titan's radius
                        break
                else:
                    first_apoapsis = None  # If no apoapsis is found
            else:
                first_apoapsis = None

            grouped_data[v_inf].append((r_p, first_apoapsis))

        # Sort v_inf groups
        sorted_v_inf = sorted(grouped_data.keys())

        # Create a plot
        plt.figure(figsize=(10, 6))
        cmap = plt.cm.plasma
        all_apoapsis_values = [apo for group in grouped_data.values() for _, apo in group if apo is not None]
        norm = plt.Normalize(min(all_apoapsis_values), max(all_apoapsis_values))

        # Iterate through sorted v_inf and connect points between adjacent groups
        for i in range(len(sorted_v_inf) - 1):
            v_inf = sorted_v_inf[i]
            next_v_inf = sorted_v_inf[i + 1]
            current_group = grouped_data[v_inf]
            next_group = grouped_data[next_v_inf]

            # Check all combinations of data points between the two groups
            for r_p, first_apoapsis in current_group:
                for next_r_p, next_first_apoapsis in next_group:
                    if (
                        first_apoapsis is not None
                        and next_first_apoapsis is not None
                        and abs(first_apoapsis - next_first_apoapsis) <= 500000  # Connect if apoapsis difference <= 500,000 km
                    ):
                        color = cmap(norm(first_apoapsis))
                        plt.plot([v_inf, next_v_inf], [r_p, next_r_p], color=color, linewidth=1.5)

        # Plot all points
        for v_inf, group in grouped_data.items():
            for r_p, first_apoapsis in group:
                if first_apoapsis is not None:
                    color = cmap(norm(first_apoapsis))
                    plt.scatter(v_inf, r_p, color=color, edgecolor='black', s=50)

        # Configure the plot
        plt.title('Required 1st Crossing Altitude vs $v_{inf}$')
        plt.xlabel('$v_{inf}$ [km/s]')
        plt.ylabel('Required 1st Crossing Altitude [km]')
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xlim(0, max(sorted_v_inf) * 1.1)
        plt.ylim(0, max([r_p for group in grouped_data.values() for r_p, _ in group]) * 1.1)

        # Add a color bar to represent the first apoapsis
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(all_apoapsis_values)
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Apoapsis of 1st Orbit [km]')

        # Show the plot
        plt.show()


    #endregion

    #region Plot Heating Rate
    if Plot_HeatingRate:
        # Compute data for heating rate plot
        def compute_heating_rate_data(results):
            global atmospheric_phases, color_map
            atmospheric_phases = []  # Reset atmospheric phases list
            color_map = {}  # Store mapping of r_p to color

            # Generate a color palette for configurations based on r_p
            unique_r_p_values = sorted({result['r_p'] for result in results})
            colors = plt.cm.plasma(np.linspace(0, 1, len(unique_r_p_values)))
            config_color_dict = {r_p: color for r_p, color in zip(unique_r_p_values, colors)}

            # Loop through the results to collect heating rate and duration data
            for result in results:
                full_trajectory = result['full_trajectory']
                time = full_trajectory.t / 86400  # Convert time to days
                r = full_trajectory.y[0, :]
                phi = full_trajectory.y[1, :]
                rhor = full_trajectory.y[2, :]
                rhophi = full_trajectory.y[3, :]
                v = np.sqrt(rhor ** 2 + (r * rhophi) ** 2)  # Velocity in [km/s]

                # Compute Titan's position at each time step in the trajectory
                titan_positions = [get_titan_position(t * 86400)[:2] for t in time]

                # Calculate atmospheric density and heating rate based on altitude above Titan
                atmo_density = np.zeros_like(r)
                Q_dot = np.zeros_like(r)
                for i, rad in enumerate(r):
                    x_spacecraft = rad * np.cos(phi[i])
                    y_spacecraft = rad * np.sin(phi[i])

                    # Distance from the spacecraft to Titan at this time step
                    titan_x, titan_y = titan_positions[i]
                    r_titan_spacecraft = np.sqrt((x_spacecraft - titan_x)**2 + (y_spacecraft - titan_y)**2)
                    altitude = r_titan_spacecraft - R_titan  # Altitude above Titan’s surface

                    # Compute atmo_density and Q_dot if within atmosphere
                    if altitude <= atmo_height:
                        atmo_density[i] = atmospheric_density(altitude + R_titan)
                        Q_dot[i] = c_q * np.sqrt(atmo_density[i]) * v[i] ** 3
                    else:
                        atmo_density[i] = 0
                        Q_dot[i] = 0

                # Track the max Q_dot, average Q_dot, and duration of each atmospheric phase
                in_atmosphere = (atmo_density > 0)
                start = None
                Q_dot_max = 0.0
                Q_dot_sum = 0.0
                duration_total = 0.0
                duration_above_threshold = 0.0

                for i, in_atmo in enumerate(in_atmosphere):
                    if in_atmo:
                        if start is None:
                            start = time[i]
                        Q_dot_max = max(Q_dot_max, Q_dot[i])
                        Q_dot_sum += Q_dot[i]
                    elif start is not None:
                        # Atmospheric phase ends
                        end = time[i]
                        duration_total = end - start

                        # Calculate average heating rate over the phase
                        Q_dot_avg = Q_dot_sum / (i - np.argmax(in_atmosphere))

                        # Calculate duration above 90% of Q_dot_max
                        for j in range(np.argmax(in_atmosphere), i):
                            if Q_dot[j] >= 0.9 * Q_dot_max:
                                duration_above_threshold += time[j + 1] - time[j]

                        # Save phase data
                        config = result['r_p']
                        color = config_color_dict[config]
                        atmospheric_phases.append((Q_dot_max, Q_dot_avg, duration_total, duration_above_threshold, color, config))

                        # Reset for the next atmospheric phase
                        start = None
                        Q_dot_max = 0.0
                        Q_dot_sum = 0.0
                        duration_above_threshold = 0.0

                # Handle the case where the spacecraft is still in the atmosphere at the end
                if start is not None:
                    end = time[-1]
                    duration_total = end - start
                    Q_dot_avg = Q_dot_sum / (len(time) - np.argmax(in_atmosphere))

                    for j in range(np.argmax(in_atmosphere), len(time) - 1):
                        if Q_dot[j] >= 0.9 * Q_dot_max:
                            duration_above_threshold += time[j + 1] - time[j]

                    config = result['r_p']
                    color = config_color_dict[config]
                    atmospheric_phases.append((Q_dot_max, Q_dot_avg, duration_total, duration_above_threshold, color, config))

            color_map = config_color_dict

        # Generate the data for heating rate plot
        compute_heating_rate_data(full_trajectory_results)

        # Plot the heating rate vs duration if data is available
        def plot_heating_rate_vs_duration():
            plt.figure(figsize=(7, 6))
            plt.subplots_adjust(right=0.8)

            # Extract the data for plotting
            Q_dot_max_values = np.array([phase[0] for phase in atmospheric_phases])
            Q_dot_avg_values = np.array([phase[1] for phase in atmospheric_phases])
            durations = np.array([phase[2] * 1440 for phase in atmospheric_phases])  # Convert durations from days to minutes
            durations_max = np.array([phase[3] * 1440 for phase in atmospheric_phases])  # Peak durations
            colors = [phase[4] for phase in atmospheric_phases]
            r_p_values = [phase[5] for phase in atmospheric_phases]  # Extract r_p values

            # Group atmospheric phases by r_p
            grouped_by_r_p = defaultdict(list)
            for i, r_p in enumerate(r_p_values):
                grouped_by_r_p[r_p].append((durations[i], Q_dot_max_values[i], durations_max[i], Q_dot_avg_values[i]))

            # Plot the data points and reset the counter for each r_p group
            for r_p, phases in grouped_by_r_p.items():
                crossing_counter = 1  # Reset counter for this r_p
                for durations, Q_dot_max, durations_max, Q_dot_avg in phases:
                    # Plot max heating rate points
                    plt.scatter(durations_max, Q_dot_max, color=colors[r_p_values.index(r_p)], marker='x', label='Max Heating Rate')
                    plt.text(durations_max, Q_dot_max, str(crossing_counter), fontsize=8, color='black', ha='left', va='bottom')

                    # Plot average heating rate points
                    plt.scatter(durations, Q_dot_avg, color=colors[r_p_values.index(r_p)], marker='o', label='Average Heating Rate')
                    plt.text(durations, Q_dot_avg, str(crossing_counter), fontsize=8, color='black', ha='right', va='top')

                    crossing_counter += 1  # Increment counter for the next crossing

            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Atmospheric Phase Duration (minutes)')
            plt.ylabel('Heating Rate (W/m²)')
            plt.title('Heating Rate vs Atmospheric Phase Duration')

            # Existing horizontal lines
            plt.axhline(28000, color='black', linestyle='--', linewidth=1, label='Space Shuttle Radiation Limit')

            # Adjust duration range for TPS curves
            duration_range = np.linspace(0.1, max(durations.max(), 200), 500)  # Ensure duration_range covers your data
            Q_total_ablative = 187.5e6  # J/m²
            Q_dot_ablative = Q_total_ablative / (duration_range * 60)
            plt.plot(duration_range, Q_dot_ablative, color='green', linestyle='-', label='Ablative Limit (PICA)')

            Q_total_heat_sink = 36.45e6  # J/m²
            Q_dot_heat_sink = Q_total_heat_sink / (duration_range * 60)
            plt.plot(duration_range, Q_dot_heat_sink, color='orange', linestyle='-', label='Heat Sink Limit (Aluminum)')

            # Combine all x-values and y-values for setting axis limits
            all_x_values = np.concatenate([np.atleast_1d(durations), np.atleast_1d(durations_max), np.atleast_1d(duration_range)])
            all_y_values = np.concatenate([np.atleast_1d(Q_dot_max_values), np.atleast_1d(Q_dot_avg_values),np.atleast_1d(Q_dot_ablative), np.atleast_1d(Q_dot_heat_sink)])

            all_y_values = all_y_values[all_y_values > 0]  # Exclude non-positive values for log scale

            # Include horizontal lines in y-values
            horizontal_lines = np.array([28000, 5000, 2800, 1120])
            all_y_values = np.concatenate([all_y_values, horizontal_lines])

            # Set x-axis limits
            x_max = all_x_values.max() * 1.1  # Add 10% margin
            plt.xlim(0.1, x_max)

            # Set y-axis limits
            y_max = all_y_values.max() * 1.2  # Add 20% margin
            plt.ylim(100, y_max)

            # Custom legend entries
            color_legend = [
                Line2D([0], [0], marker='o', color=color, lw=0, label=f'r_p = {config - R_titan:.0f} km')
                for config, color in color_map.items()
            ]

            # Add entries for max and average heating rates with specific markers
            marker_legend = [
                Line2D([0], [0], marker='x', color='black', lw=0, label='Max Heating Rate', markersize=6),
                Line2D([0], [0], marker='o', color='black', lw=0, label='Average Heating Rate', markersize=6)
            ]

            # Combine legends for configurations, max/avg heating rates, and limits
            plt.legend(handles=color_legend + marker_legend + [
                Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Space Shuttle Radiation Limit'),
                Line2D([0], [0], color='green', linestyle='-', linewidth=1, label='Ablative Limit (PICA)'),
                Line2D([0], [0], color='orange', linestyle='-', linewidth=1, label='Heat Sink Limit (Aluminum)')
            ], loc='center left', bbox_to_anchor=(1, 0.5))
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        plot_heating_rate_vs_duration()
    #endregion

    plt.show()
#endregion