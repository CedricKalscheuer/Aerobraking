import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.widgets as mwidgets
import time as tm
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.colors import PowerNorm  # Use PowerNorm for nonlinear scaling
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from matplotlib.widgets import CheckButtons, Slider
from scipy.integrate import solve_ivp
from scipy.interpolate import make_interp_spline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import PchipInterpolator
from scipy.signal import argrelextrema

#region User input
# === Rocket parameters ============================
m_0                         = 5712            # Initial spacecraft mass in [kg]
m_p                         = 2978            # Propellant mass in [kg]
v_inf_values                = [1]  # Hyperbolic excess speeds at infinity in [km/s] 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3.00, 3.25, 3.50, 3.75, 4.00, 4.25, 4.50, 4.75, 5.00
thrust                      = 0.235                 # Thrust in [N]
Isp                         = 3100              # Specific impulse
c_d                         = 2.2               # Drag coefficient
A                           = 11.04466              # Cross-sectional area of the spacecraft in [m^2]
# === Orbit parameters ============================
tint                        = 1500       # Integration time in days
tint_short                  = tint               # Integration time for short runs
max_crossings               = 3
tolerance                   = 0.5     # Tolerance for calculations
# === Output parameters ============================
Plot_Trajectory             = True              # Plot spacecraft trajectory?
Plot_Atmosphere             = False             # Plot atmospheric density over height?
Plot_Values                 = False             # Plot panel of values?
Plot_Comparison             = False             # Plot v_inf vs. r_p required?
Plot_HeatingRate            = False             # Plot heating rate maximum and average over time?
#endregion

#region Prepare the program
# Planetary and Orbital constants
g0_earth            = 9.80665               # Earth standard gravitational acceleration in [m/s^2]
# === Saturn ============================
mu_saturn           = 37931206.234            # Gravitational parameter of Saturn in [km^3/s^2]
R_saturn            = 60268.0               # Radius of Saturn in [km]
dist_saturn_titan   = 1221870.0             # Approximate distance between Saturn and Titan in [km]
R_SOI_saturn        = 54.5 * 1e6            # Saturn's Radius of Sphere of Influence in [km]
# === Titan ============================
mu_titan            = 8978.14               # Gravitational parameter of Titan in [km^3/s^2]
R_titan             = 2575.5               # Titan equatorial radius in [km]
R_SOI_titan         = 43321.241                 # Titan's radius of standard Sphere of Influence in [km]
a_titan             = 1221870               # Semi-major axis in km
e_titan             = 0.0288                # Eccentricity
T_titan             = 2 * np.pi * np.sqrt(a_titan**3 / mu_saturn)
# === Enceladus ============================
mu_enceladus        = 7.210367               # Gravitational parameter of Enceladus in [km^3/s^2]
R_enceladus         = 252.3                 # Enceladus equatorial radius in [km]
R_SOI_enceladus     = 488                   # Enceladus radius of standard Sphere of Influence in [km]
a_enceladus         = 238040                # Semi-major axis of Enceladus' orbit in [km]
e_enceladus         = 0.0047                # Eccentricity of Enceladus' orbit
T_enceladus         = 2 * np.pi * np.sqrt(a_enceladus**3 / mu_saturn)  # Orbital period of Enceladus in seconds
# === Accuracy ============================

max_step_critical           = 0.50            # Maximum step size during simulation for critical areas
max_step_non_critical       = 500             # Maximum step size during simulation for non-critical areas
critical_distance_titan_sim = R_titan + 10000
critical_distance_saturn_sim= R_saturn
# === Unit Conversion ============================
tmax                = tint * 86400                  # Convert integration time in [s]
tmax_short          = tint_short * 86400            # Convert integration time in [s]
A                   = A / 1e6                       # Convert cross-sectional area to [km^2]
thrust              = thrust / 1000.0               # Convert thrust to [kg*km/s^2]
ceff                = Isp * g0_earth / 1000.0       # Effective exhaust velocity in [km/s]

# Initialize mission phase variables
orbit_insertion_maneuver    = True      # Set to False after the first thruster burn
thrust_active               = False     # Track when thrust is active for plots
synchronized                = False
#endregion

#region Atmosphere
atmo_height = 1375  # Height of the atmosphere boundary in km

# Densities in kg/km^3 (or your chosen units)
density_surface = 1 * 1e8       # Density at the surface (r = R_titan)
density_boundary = 1e-2            # Density at the top of the atmosphere (r = R_titan + atmo_height)

# Intermediate altitude and density (user-defined)
h_int = 600                        # Intermediate altitude in km above the surface
density_int = 5.0e2        #5.0e2          # Density at h_int (in kg/km^3)

def atmospheric_density(r_titan_spacecraft):
    # Calculate altitude above the surface
    height = r_titan_spacecraft - R_titan
    # Below the intermediate altitude
    if height <= h_int:
        # Calculate first scale height H1
        H1 = h_int / np.log(density_surface / density_int)
        return density_surface * np.exp(-height / H1)  
    elif height <= atmo_height: # Between the intermediate altitude and the atmosphere boundary
        # Calculate second scale height H2
        H2 = (atmo_height - h_int) / np.log(density_int / density_boundary)
        # Continue from density_int at h_int, decaying with H2 for the extra height above h_int
        return density_int * np.exp(-(height - h_int) / H2)
    # Above the atmosphere, density is zero
    else:
        return 0
#endregion

#region Mission Phases
def handle_orbit_insertion_maneuver(r, m, mdry, t):
    global thrust_active, orbit_insertion_maneuver, crossing_count
    # If the crossing count has reached 7, disable the orbit insertion maneuver.
    if crossing_count >= 7:
        orbit_insertion_maneuver = False
        #print(f"Orbit insertion finished")
        return 0.0  # Throttle fully closed
    else:
        return 0.0  # No thrust

def synchronize_orbits(state, mdry, apoapsis, periapsis, t, r_p_slow):
    global thrust_active, synchronized
    r, phi, rhor, rhophi, m = state
    # If thrust is active, check if we should stop thrusting
    if thrust_active:
        v = np.sqrt(rhor**2 + (r * rhophi)**2)
        E = v**2 / 2 - mu_saturn / r
        a = -mu_saturn / (2 * E)
        T_sc = 2 * np.pi * np.sqrt(a**3 / mu_saturn) # Compute the current orbital period of the spacecraft:
        ratio = T_sc / T_enceladus # Compute ratio with respect to Enceladus’s orbital period:
        tolerance_ratio = 0.001  # Define a tolerance (0.1% difference)
        # Check if the ratio is nearly an integer:
        if abs(ratio - round(ratio)) < tolerance_ratio:
            thrust_active = False  # Stop thrusting
            #print("Thrust stopped: orbital period synchronized (T_sc ≈ {}×T_enceladus)".format(round(ratio)))
            synchronized = True
            return 0.0
        # Stop thrusting if mass reaches dry mass
        elif m <= mdry or abs(r-a_enceladus) > 400000:
            thrust_active = False
            return 0.0  # Throttle fully closed
        else:
            return 1.0  # Continue thrusting

    # If thrust is not active, check if we should start thrusting
    elif abs(r-a_enceladus) <= 400000 and m > mdry and not synchronized:
        thrust_active = True
        #print(f"Thrust initiated")
        return 1.0  # Throttle fully open

    else:
        return 0.0  # No thrust
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
    if v_inf <= 0.10:
        b_i = 13610e3 - dist_saturn_titan
    elif v_inf <= 0.50:
        b_i = 13547e3 - dist_saturn_titan
    elif v_inf <= 0.75:
        b_i = 9577e3 - dist_saturn_titan
    elif v_inf <= 1.00:
        b_i = 7455e3 - dist_saturn_titan
    elif v_inf <= 1.25:
        b_i = 6167e3 - dist_saturn_titan
    elif v_inf <= 1.50:
        b_i = 5298e3 - dist_saturn_titan
    elif v_inf <= 1.75:
        b_i = 4643e3 - dist_saturn_titan
    elif v_inf <= 2.00:
        b_i = 4158e3 - dist_saturn_titan
    elif v_inf <= 2.25:
        b_i = 3794e3 - dist_saturn_titan
    elif v_inf <= 2.50:
        b_i = 3479e3 - dist_saturn_titan
    elif v_inf <= 2.75:
        b_i = 3238e3 - dist_saturn_titan
    elif v_inf <= 3.00:
        b_i = 3028.3e3 - dist_saturn_titan
    elif v_inf <= 3.25:
        b_i = 2856e3 - dist_saturn_titan
    elif v_inf <= 3.50:
        b_i = 2705e3 - dist_saturn_titan
    elif v_inf <= 3.75:
        b_i = 2577.5e3 - dist_saturn_titan
    elif v_inf <= 4.00:
        b_i = 2467e3 - dist_saturn_titan
    elif v_inf <= 4.25:
        b_i = 2370e3 - dist_saturn_titan
    elif v_inf <= 4.50:
        b_i = 2282.5e3 - dist_saturn_titan
    elif v_inf <= 4.75:
        b_i = 2203e3 - dist_saturn_titan
    elif v_inf <= 5.00:
        b_i = 2135e3 - dist_saturn_titan
    elif v_inf <= 5.25:
        b_i = 2070e3 - dist_saturn_titan
    elif v_inf <= 5.50:
        b_i = 2017e3 - dist_saturn_titan
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

def eom(t, state, thrust, ceff, mdry, r_p_slow):
    global thrust_active, orbit_insertion_maneuver

    r, phi, rhor, rhophi, m = state

    # Get Titan's position (assumed to be in Saturn-centered coordinates)
    x_titan, y_titan, v_titan = get_titan_position(t)

    # Spacecraft's position in Saturn-centered coordinates
    x_spacecraft = r * np.cos(phi)
    y_spacecraft = r * np.sin(phi)

    # Compute distances from spacecraft to Titan and Saturn
    dx = x_spacecraft - x_titan
    dy = y_spacecraft - y_titan
    r_titan_spacecraft = np.sqrt(dx**2 + dy**2)
    r_saturn_spacecraft = r  # since r is the radial distance from Saturn

    # Calculate spacecraft velocity and flight path angle
    v = np.sqrt(rhor**2 + (r * rhophi)**2)
    alpha = np.arctan2(r * rhophi, rhor)
    E = v**2 / 2 - mu_saturn / r  
    h = r**2 * rhophi
    a = -mu_saturn / (2 * E)
    e = np.sqrt(1 + (2 * E * h**2) / (mu_saturn**2))
    apoapsis = a * (1 + e)
    periapsis = a * (1 - e)

    throttle = 0.0
    atmo_density = atmospheric_density(r_titan_spacecraft)
    D = max(0.5 * c_d * atmo_density * A * v**2, 0) if atmo_density > 0 else 0

    # === Titan's Gravitational Acceleration (always computed) ===
    accel_titan = -mu_titan / r_titan_spacecraft**2
    titan_dx = dx / r_titan_spacecraft
    titan_dy = dy / r_titan_spacecraft
    gravitational_acceleration_r_titan = accel_titan * (titan_dx * np.cos(phi) + titan_dy * np.sin(phi))
    gravitational_acceleration_phi_titan = accel_titan * (-titan_dx * np.sin(phi) + titan_dy * np.cos(phi))

    # === Saturn's Gravitational Acceleration (always computed) ===
    accel_saturn = -mu_saturn / r_saturn_spacecraft**2
    # The spacecraft's unit vector in Saturn-centered coordinates is (cos(phi), sin(phi))
    dx_norm = x_spacecraft / r_saturn_spacecraft
    dy_norm = y_spacecraft / r_saturn_spacecraft
    gravitational_acceleration_r_saturn = accel_saturn * (dx_norm * np.cos(phi) + dy_norm * np.sin(phi))
    gravitational_acceleration_phi_saturn = accel_saturn * (-dx_norm * np.sin(phi) + dy_norm * np.cos(phi))

    # === Combine both gravitational contributions ===
    gravitational_acceleration_r = gravitational_acceleration_r_titan + gravitational_acceleration_r_saturn
    gravitational_acceleration_phi = gravitational_acceleration_phi_titan + gravitational_acceleration_phi_saturn

        # Aerobraking thrust logic
    if orbit_insertion_maneuver:
        beta = alpha + np.pi
        throttle = handle_orbit_insertion_maneuver(r, m, mdry, t)
    else:
        beta = alpha + np.pi
        throttle = synchronize_orbits(state, mdry, apoapsis, periapsis, t, r_p_slow)

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

#endregion

#region Simulate Trajectory
class Trajectory:
    pass

def simulate_trajectory(v_inf, r_p_slow, short_run=True,
                        max_step_large=max_step_non_critical, max_step_small=max_step_critical,
                        launch_time=0, critical_distance=critical_distance_titan_sim,
                        stop_after_first_crossing=False, radial_tolerance_km=5):

    global orbit_insertion_maneuver, thrust_active
    global crossing_count
    inside_critical_zone = False

    # Reset mission phase variables
    orbit_insertion_maneuver = True
    thrust_active = False
    crossing_in_progress = False
    crossing_count = 0  # local variable for counting

    # Initial conditions
    x_0, y_0, r_0, phi_0, rho_r_0, rho_phi_0 = get_initial_spacecraft_state(v_inf, dist_saturn_titan, R_SOI_saturn, R_titan, mu_titan)
    init_val = [r_0, phi_0, rho_r_0, rho_phi_0, m_0]
    p = (thrust, ceff, m_0 - m_p, r_p_slow)

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
        nonlocal  crossing_in_progress
        global crossing_count

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
                event_list = [planet_crash, enter_critical_zone, first_crossing_event]
            else:
                event_list = [planet_crash, enter_critical_zone, multiple_crossings_event]
        else:
            if stop_after_first_crossing:
                event_list = [planet_crash, exit_critical_zone, first_crossing_event]
            else:
                event_list = [planet_crash, exit_critical_zone, multiple_crossings_event]

        # Perform integration
        sol = solve_ivp(eom, t_span, init_val, args=p, method='RK45', rtol=1e-8, atol=1e-15,max_step=max_step_current, events=event_list)

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
            elif event_occurred in [planet_crash, first_crossing_event, multiple_crossings_event]:
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
    #region Temporary
    rp_matching_tol = 3
    if initial_delay_time is None: # Check if a delay time for a similar configuration is in the database
        for cfg in r_p_configurations.get(v_inf, []):
            if abs(cfg["first_crossing_rp"] - r_p) <= rp_matching_tol:
                initial_delay_time = cfg.get("initial_delay", 0)
                break
        else:
            # Fallback if none matched
            initial_delay_time = 0 
    delay_time = initial_delay_time

    iteration = 1
    max_iterations = 21

    while iteration < max_iterations:
        # "Stop after first crossing" only
        _, _, trajectory, _, _ = simulate_trajectory(v_inf, r_p, short_run=True, launch_time=delay_time, 
        critical_distance=critical_distance_titan_sim, stop_after_first_crossing=True, radial_tolerance_km=radial_tolerance_km)

        crossings_in_iteration = []
        crossing_found = False
        within_tolerance = False

        for i in range(1, len(trajectory.y[0, :])):
            # === Position Spacecraft/Titan ===
            x_titan, y_titan, v_titan   = get_titan_position(trajectory.t[i])   # Titan's position
            r_spacecraft                = trajectory.y[0, i]                    # Spacecraft radial position
            phi_spacecraft              = trajectory.y[1, i]                    # Spacecraft angular position
            x_spacecraft                = r_spacecraft * np.cos(phi_spacecraft) # Spacecraft's position in Cartesian coordinates
            y_spacecraft                = r_spacecraft * np.sin(phi_spacecraft)

            # === Distance Calculations ===
            distance_to_titan           = np.sqrt((x_spacecraft - x_titan)**2 + (y_spacecraft - y_titan)**2)    # Distance between spacecraft and Titan
            distance_to_titan_surface   = distance_to_titan - R_titan                                           # Altitude above Titan's surface
            r_titan_at_phi              = np.sqrt(x_titan**2 + y_titan**2)                                      # Titan's radial distance from Saturn
            radial_difference           = abs(r_spacecraft - r_titan_at_phi)                                    # Distance between spacecraft and Titans orbit

            # Check if the spacecraft is within radial tolerance of Titan's orbit
            if radial_difference <= radial_tolerance_km:
                if not within_tolerance:                    #Ensure a crossing only gets checked once until out of tolerance region again
                    within_tolerance = True
                    crossing_found = True
                    time_at_orbit = trajectory.t[i]
                    phi_spacecraft_at_orbit = phi_spacecraft
                    x_titan_at_orbit = x_titan
                    y_titan_at_orbit = y_titan
                    v_titan_at_orbit = v_titan
                    distance_to_titan_surface_at_orbit = distance_to_titan_surface
                    crossings_in_iteration.append((i, time_at_orbit, phi_spacecraft_at_orbit, x_titan_at_orbit,y_titan_at_orbit, v_titan_at_orbit, distance_to_titan_surface_at_orbit))
            else:
                if within_tolerance:
                    within_tolerance = False

        if not crossing_found:
            print(f"WARNING: The spacecraft did not cross Titan's orbit during iteration {iteration}.")
            return delay_time, None, None, None, None 

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

        # Adjust delay time based on the angular difference
        if abs(delta_phi_deg) > 1: 
            r_current = np.sqrt(x_titan_at_orbit**2 + y_titan_at_orbit**2)
            omega_avg = v_titan_at_orbit / r_current  
            delay_adjustment = delta_phi / omega_avg * 0.800 
            delay_time += delay_adjustment  
            print(f"                    [{v_inf:.2f} km/s] Delay-Iteration {iteration} for r_p = {r_p - R_titan:.6f} km: Spacecraft is {delta_phi_deg:.2f} degrees away from crossing {delay_time/86400:.14f} days")
            #print(f"  Adjusting delay by {delay_adjustment / 3600:.2f} hours to {delay_time/86400:.10f} days.")
        else:  # Fine adjustments for small angular differences
    #endregion
            if delta_phi_deg > 0:
                if abs(distance_error) >= 200:
                    delay_adjustment = abs(distance_error) * 0.1 * (1/v_inf)**0.03
                else:
                    if abs(distance_error) < 0.0000002:
                        delay_adjustment = abs(distance_error) *0.08 *0.020
                    else:
                        if v_inf <= 0.50:
                            delay_adjustment = abs(distance_error) * 0.18
                        elif v_inf <= 0.75:
                            delay_adjustment = abs(distance_error) * 0.184/1.5 
                        elif v_inf <= 1.0:
                            delay_adjustment = abs(distance_error) * 0.183*0.35   
                        elif v_inf <= 2.0:
                            delay_adjustment = abs(distance_error) * 0.133*0.4   
                        elif v_inf <= 2.5:
                            delay_adjustment = abs(distance_error) * 0.11*0.6  
                        elif v_inf <= 3.0:
                            delay_adjustment = abs(distance_error) * 0.095*0.6 
                        elif v_inf <= 3.5:
                            delay_adjustment = abs(distance_error) * 0.07*0.5 
                        elif v_inf <= 4.0:
                            delay_adjustment = abs(distance_error) * 0.07*0.5 
                        elif v_inf <= 5.0:
                            delay_adjustment = abs(distance_error) * 0.07*0.07
                        elif v_inf <= 6.0:
                            delay_adjustment = abs(distance_error) * 0.09*0.1
                        else:
                            delay_adjustment = abs(distance_error) * 0.177 * v_inf**0.0213
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
                    print(f"                    [{v_inf:.2f} km/s] Delay-Iteration {iteration} for r_p = {r_p - R_titan:.6f} km: Spacecraft is {abs(distance_error):.10f} km too close to Titan's surface. {delay_time/86400:.14f} days")
                    delay_time -= delay_adjustment
                else:
                    print(f"                    [{v_inf:.2f} km/s] Delay-Iteration {iteration} for r_p = {r_p - R_titan:.6f} km: Spacecraft is {abs(distance_error):.10f} km too far ahead of Titan. {delay_time/86400:.14f} days")
                    delay_time += delay_adjustment
            else:  # Spacecraft is behind Titan
                print(f"                    [{v_inf:.2f} km/s] Delay-Iteration {iteration} for r_p = {r_p - R_titan:.6f} km: Spacecraft is behind Titan ({distance_to_titan_surface_at_orbit:.8f} km), {delay_time/86400:.14f} days")
                delay_time -= delay_adjustment

        iteration += 1

    # If we reach here, we have either converged or hit the maximum iterations
    print(f"Final launch delay for [v_inf = {v_inf:.2f} km/s] r_p = {r_p - R_titan:.4f} km: {delay_time / 86400:.14f} days")
#endregion

#region Adjust Aerobrake Altitude
def adjust_r_p_to_match_crossing(v_inf, target_crossing_number, target_crossing_distance, r_p_initial, tolerance_angle, tolerance_distance):
    current_r_p = r_p_initial
    iteration = 1
    radial_tolerance_km = 5
    max_iterations = 21

    if target_crossing_number == 3:
        if v_inf < 1:
            k_angle_base = 1.15 * v_inf ** 0.21 *0.1
            k_distance_base = 0.00004 * v_inf ** 0.1 * 0.5
        elif v_inf < 2:
            k_angle_base = 1.1*0.1
            k_distance_base = 0.00007*0.1
        elif v_inf < 2.5:
            k_angle_base = 1.25*0.1
            k_distance_base = 0.000055*0.2
        elif v_inf < 3:
            k_angle_base = 0.2*0.6
            k_distance_base = 0.000003*2.5
        elif v_inf <= 3.5:
            k_angle_base = 0.3*0.2
            k_distance_base = 0.0000045*0.6
        elif v_inf <= 4.0:
            k_angle_base = 0.3*0.04
            k_distance_base = 0.0000045*0.4
        else:
            k_angle_base = 0.3*0.0005
            k_distance_base = 0.0000004
    elif target_crossing_number == 6:
        if v_inf < 1:
            k_angle_base    = -0.00055
            k_distance_base = -0.000000035
        if v_inf < 1.5:
            k_angle_base    = -0.00055
            k_distance_base = -0.000000045
        elif v_inf < 2:
            k_angle_base    = -0.00060
            k_distance_base = -0.000000045
        elif v_inf < 2.5:
            k_angle_base    = -0.00045
            k_distance_base = -0.000000040
        elif v_inf < 3:
            k_angle_base    = -0.000045
            k_distance_base = -0.000000004
        elif v_inf <= 3.5:
            k_angle_base    = -0.00003
            k_distance_base = -0.000000003
        else:
            k_angle_base    = -0.0000025*1
            k_distance_base = -0.00000000015
    elif target_crossing_number == 8:
        if v_inf <= 1:
            k_angle_base = 0.00000075
            k_distance_base = 0.000000000040*3   
        elif v_inf <= 1.5:
            k_angle_base = 0.00000100
            k_distance_base = 0.000000000047*3  
        elif v_inf <= 2:
            k_angle_base = 0.00000075
            k_distance_base = 0.000000000040*1             
        elif v_inf <= 2.5:
            k_angle_base = 0.00000068
            k_distance_base = 0.000000000035*0.9   
        else:
            k_angle_base = 0.00000022
            k_distance_base = 0.000000000010*5  
    else:
        k_angle_base = 10 * (1 / v_inf) ** 0.7
        k_distance_base = 0.00015 * (1 / v_inf) ** 1.4
    
    while iteration < max_iterations:
        delay_time, _, _, _, _ = calculate_launch_delay_iterative(0, current_r_p, v_inf)

        _, _, trajectory, _, _ = simulate_trajectory( v_inf, current_r_p, short_run=True, launch_time=delay_time,
            critical_distance=critical_distance_titan_sim,stop_after_first_crossing=False)

        within_tolerance = False
        crossings_found = []
        last_periapsis_value = None

        r = trajectory.y[0, :]
        phi = trajectory.y[1, :]
        rhor = trajectory.y[2, :]

        for i in range(len(trajectory.t)):
            # Detect periapsis by finding where rhor switches from negative to positive
            if i > 0 and rhor[i - 1] < 0 and rhor[i] > 0:
                last_periapsis_value = r[i]

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

        if len(crossings_found) < target_crossing_number:
            print(f"[{v_inf:.2f} km/s] NOT ENOUGH CROSSINGS. FOUND: {len(crossings_found)}, REQUIRED: {target_crossing_number}")
            break

        _, crossing_distance, crossing_angle_deg = crossings_found[target_crossing_number - 1]

        #Check angle and distance conditions
        if abs(crossing_angle_deg) > tolerance_angle or crossing_angle_deg < 0:
            # Adjust r_p based on angle
            if crossing_angle_deg < 0 and abs(crossing_angle_deg) < 1:
                delta_r_p_angle = 7 * k_angle_base * crossing_angle_deg
            else:
                delta_r_p_angle = k_angle_base * crossing_angle_deg
            print(f"[{v_inf:.2f} km/s] Error: {crossing_angle_deg:.2f} deg | CROSSING #{target_crossing_number} | R_P {current_r_p - R_titan:.10f} -> {current_r_p + delta_r_p_angle - R_titan:.10f} km | DELAY: {delay_time/86400:.14f} days | Iteration {iteration}")
            current_r_p += delta_r_p_angle
        else:
            # Angle is within tolerance, now adjust based on distance
            distance_error = crossing_distance - target_crossing_distance
            if abs(distance_error) <= tolerance_distance:
                crossings_to_display = [1, 3, 6, 8]
                crossing_distances_str = " | ".join(
                    [f"C{i+1}: {(crossings_found[i][1] - R_titan):.10f} km" 
                    for i in range(target_crossing_number) if i+1 in crossings_to_display]
                )

                print(f"ITERATION DONE FOR [{v_inf:.2f} km/s] | {crossing_distances_str} | LAST PERIAPSIS DISTANCE TO ENCELADUS: {(last_periapsis_value-a_enceladus):.0f} km | DELAY: {delay_time/86400:.14f} days")
                if target_crossing_number <= 3:
                    print(
                        f'{{"first_crossing_rp": R_titan + {crossings_found[0][1] - R_titan:.10f}, '
                        f'"second_crossing_rp": R_titan + {crossings_found[2][1] - R_titan:.10f}, '
                        f'"initial_delay": 86400 * {delay_time/86400:.14f}}},'
                    )
                elif target_crossing_number <=6:
                    print(
                        f'{{"first_crossing_rp": R_titan + {crossings_found[0][1] - R_titan:.10f}, '
                        f'"second_crossing_rp": R_titan + {crossings_found[2][1] - R_titan:.10f}, '
                        f'"third_crossing_rp": R_titan + {crossings_found[5][1] - R_titan:.10f}, '
                        f'"initial_delay": 86400 * {delay_time/86400:.14f}}},'
                    )
                else:
                    print(
                        f'{{"first_crossing_rp": R_titan + {crossings_found[0][1] - R_titan:.10f}, '
                        f'"second_crossing_rp": R_titan + {crossings_found[2][1] - R_titan:.10f}, '
                        f'"third_crossing_rp": R_titan + {crossings_found[5][1] - R_titan:.10f}, '
                        f'"fourth_crossing_rp": R_titan + {crossings_found[7][1] - R_titan:.10f}, '
                        f'"initial_delay": 86400 * {delay_time/86400:.14f}}},'
                    )        
                return current_r_p, delay_time, trajectory
            else:
                if current_r_p - R_titan <= 600: 
                    delta_r_p_distance = k_distance_base / 2.0 * distance_error
                elif current_r_p - R_titan <= atmo_height: 
                    delta_r_p_distance = k_distance_base / 1.5 * distance_error
                else:                                         
                    delta_r_p_distance = k_distance_base * distance_error
                print(f"[{v_inf:.2f} km/s] Error: {distance_error:.2f} km | CROSSING #{target_crossing_number} at {crossing_distance-R_titan:.10f} km|  R_P {current_r_p - R_titan:.10f} -> {current_r_p + delta_r_p_distance - R_titan:.10f} km | DELAY: {delay_time/86400:.14f} days | Iteration {iteration}")
                current_r_p += delta_r_p_distance

        iteration += 1

    print(f"[{v_inf:.2f} km/s] Did not converge within {max_iterations-1} iterations for crossing #{target_crossing_number}.")
    return current_r_p, delay_time, trajectory
#endregion

r_p_configurations = {
    0.50: [
        {"first_crossing_rp": R_titan + 1713.7799894077, "second_crossing_rp": R_titan + 3404.9700263399, "initial_delay": 86400 * 6.83129846973556},
        {"first_crossing_rp": R_titan + 1119.0238754141, "second_crossing_rp": R_titan + 3590.5441335887, "initial_delay": 86400 * 6.83250354925897},
        {"first_crossing_rp": R_titan + 595.6124182022, "second_crossing_rp": R_titan + 25277.9700108570, "initial_delay": 86400 * 6.83356476899504},
        {"first_crossing_rp": R_titan + 485.7766413515, "second_crossing_rp": R_titan + 25652.6346652750, "initial_delay": 86400 * 6.83378600862785},
        {"first_crossing_rp": R_titan + 439.3954803493, "second_crossing_rp": R_titan + 3167.0616461872, "initial_delay": 86400 * 6.83387672495941},
#RADI LIMIT       {"first_crossing_rp": R_titan + 432, "second_crossing_rp": R_titan + 1322.2315743601, "initial_delay": 86400 * 6.83388417983348}, #RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 389, "second_crossing_rp": R_titan + 1322.2315743601, "initial_delay": 86400 * 6.83396766932430},#TITAN CRASH
    ],
    0.75: [
        {"first_crossing_rp": R_titan + 1807.4267575483, "second_crossing_rp": R_titan + 3886.0780806325, "initial_delay": 86400 * -0.69028600379482},
        {"first_crossing_rp": R_titan + 1214.5342390835, "second_crossing_rp": R_titan + 4074.7392331124, "initial_delay": 86400 * -0.68908392690691},
        {"first_crossing_rp": R_titan + 639.7831714036, "second_crossing_rp": R_titan + 18904.9683206127, "initial_delay": 86400 * -0.68791688239107},
        {"first_crossing_rp": R_titan + 484.0836085095, "second_crossing_rp": R_titan + 29585.3027005094, "initial_delay": 86400 * -0.68760384752055},
        {"first_crossing_rp": R_titan + 432.7835128977, "second_crossing_rp": R_titan + 9573.8780207406, "initial_delay": 86400 * -0.68750538007065},
#RADI LIMIT          {"first_crossing_rp": R_titan + 419, "second_crossing_rp": R_titan + 1351.1752755678, "initial_delay": 86400 * -0.68747876926056}, #RADI LIMIT 
#        {"first_crossing_rp": R_titan + 417, "second_crossing_rp": R_titan + 1351.1752755678, "initial_delay": 86400 * -0.68747621419103},
#        {"first_crossing_rp": R_titan + 380, "second_crossing_rp": R_titan + 1351.1752755678, "initial_delay": 86400 * -0.68741345504280},
#Titancrash        {"first_crossing_rp": R_titan + 379, "second_crossing_rp": R_titan + 1351.1752755678, "initial_delay": 86400 * -0.68741186471215},#Titancrash
    ],
    1.00: [
        {"first_crossing_rp": R_titan + 1749.6537536955, "second_crossing_rp": R_titan + 3894.9185283103, "initial_delay": 86400 * 0.13142660716527},
        {"first_crossing_rp": R_titan + 1185.1902126877, "second_crossing_rp": R_titan + 6858.3080221450, "initial_delay": 86400 * 0.13257237620635},
        {"first_crossing_rp": R_titan + 635.3913235511, "second_crossing_rp": R_titan + 24348.3241884286, "initial_delay": 86400 * 0.13369027229860},
        {"first_crossing_rp": R_titan + 483.5621298501, "second_crossing_rp": R_titan + 13831.4365502735, "initial_delay": 86400 * 0.13399426906728},
        {"first_crossing_rp": R_titan + 430.6420931143, "second_crossing_rp": R_titan + 36609.0345899678, "initial_delay": 86400 * 0.13409278685466},
#RADI LIMIT         {"first_crossing_rp": R_titan + 414, "second_crossing_rp": R_titan + 1389.0596367778, "initial_delay": 86400 * 0.13414839375079},#RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 374, "second_crossing_rp": R_titan + 1389.0596367778, "initial_delay": 86400 * 0.13418139908368}, #TITAN CRASH
    ],
    1.25: [
        {"first_crossing_rp": R_titan + 1625.2287280127, "second_crossing_rp": R_titan + 4007.8034906710, "initial_delay": 86400 * 1.57449807356945},
        {"first_crossing_rp": R_titan + 1101.1188932105, "second_crossing_rp": R_titan + 3940.7797491708, "initial_delay": 86400 * 1.57556390962396},
        {"first_crossing_rp": R_titan + 609.2275280910, "second_crossing_rp": R_titan + 25105.9636947657, "initial_delay": 86400 * 1.57656558600902},
        {"first_crossing_rp": R_titan + 483.7710826641, "second_crossing_rp": R_titan + 21150.7658159071, "initial_delay": 86400 * 1.57681379927659},
        {"first_crossing_rp": R_titan + 431.2104476349, "second_crossing_rp": R_titan + 24121.3953783499, "initial_delay": 86400 * 1.57690744600111},
#RADI LIMIT        {"first_crossing_rp": R_titan + 412, "second_crossing_rp": R_titan + 1366.4622943408, "initial_delay": 86400 * 1.57694531869574},#RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 372, "second_crossing_rp": R_titan + 1366.4622943408, "initial_delay": 86400 * 1.57699231775723}, #TITAN CRASH
    ],
    1.50: [
        {"first_crossing_rp": R_titan + 1446.6971341340, "second_crossing_rp": R_titan + 4743.2322420021, "initial_delay": 86400 * 8.48237331394427},
        {"first_crossing_rp": R_titan + 970.4882235099, "second_crossing_rp": R_titan + 3287.1861531770, "initial_delay": 86400 * 8.48334478480190},
        {"first_crossing_rp": R_titan + 579.4594678774, "second_crossing_rp": R_titan + 21678.7031960333, "initial_delay": 86400 * 8.48414110932865},
        {"first_crossing_rp": R_titan + 485.4746370802, "second_crossing_rp": R_titan + 13777.2900515403, "initial_delay": 86400 * 8.48432222984586},
        {"first_crossing_rp": R_titan + 434.2153678125, "second_crossing_rp": R_titan + 8534.8071724448, "initial_delay": 86400 * 8.48440797659150},
#RADI LIMIT         {"first_crossing_rp": R_titan + 413, "second_crossing_rp": R_titan + 1329.1227424510, "initial_delay": 86400 * 8.48444026312037},#RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 370, "second_crossing_rp": R_titan + 1329.1227424510, "initial_delay": 86400 * 8.48448816863470},#TITAN CRASH
    ],
    1.75: [
        {"first_crossing_rp": R_titan + 1577.2290637327, "second_crossing_rp": R_titan + 3766.1243305006, "initial_delay": 86400 * 7.92629878657725},
        {"first_crossing_rp": R_titan + 1191.0195167939, "second_crossing_rp": R_titan + 3552.1369059400, "initial_delay": 86400 * 7.92708722009431},
        {"first_crossing_rp": R_titan + 778.8148397491, "second_crossing_rp": R_titan + 29252.5612023349, "initial_delay": 86400 * 7.92792985371700},
        {"first_crossing_rp": R_titan + 551.6959008528, "second_crossing_rp": R_titan + 25744.0820672361, "initial_delay": 86400 * 7.92838915147439},
        {"first_crossing_rp": R_titan + 484.5019168699, "second_crossing_rp": R_titan + 16175.0380035403, "initial_delay": 86400 * 7.92851227794428},
        {"first_crossing_rp": R_titan + 436.6619739801, "second_crossing_rp": R_titan + 393.6830157576, "initial_delay": 86400 * 7.92858653791670},
#RADI LIMIT         {"first_crossing_rp": R_titan + 417, "second_crossing_rp": R_titan + 1342.2518758381, "initial_delay": 86400 * 7.92861204198171},#RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 371, "second_crossing_rp": R_titan + 1342.2518758381, "initial_delay": 86400 * 7.92865970007782},#TITAN CRASH
    ],
    2.00: [
        {"first_crossing_rp": R_titan + 1259.7437387251, "second_crossing_rp": R_titan + 4821.5072019957, "initial_delay": 86400 * 1.80120443482009},
        {"first_crossing_rp": R_titan + 927.6367790131, "second_crossing_rp": R_titan + 31641.6547688457, "initial_delay": 86400 * 1.80188518945401},
        {"first_crossing_rp": R_titan + 639.6365527437, "second_crossing_rp": R_titan + 26446.9251529457, "initial_delay": 86400 * 1.80247388287408},
        {"first_crossing_rp": R_titan + 539.0368501752, "second_crossing_rp": R_titan + 9730.2559531902, "initial_delay": 86400 * 1.80266968730508},
        {"first_crossing_rp": R_titan + 486.9576251310, "second_crossing_rp": R_titan + 948.9865575386, "initial_delay": 86400 * 1.80275814965874},
        {"first_crossing_rp": R_titan + 441.3345273558, "second_crossing_rp": R_titan + 6851.6471672571, "initial_delay": 86400 * 1.80282205665151},
#RADI LIMIT        {"first_crossing_rp": R_titan + 423, "second_crossing_rp": R_titan + 1281.1894428357, "initial_delay": 86400 * 1.80284057024789}, #RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 372, "second_crossing_rp": R_titan + 1281.1894428357, "initial_delay": 86400 * 1.80288773956795},#TITAN CRASH
    ],
    2.25: [
        {"first_crossing_rp": R_titan + 1454.7668591684, "second_crossing_rp": R_titan + 4334.6425522479, "initial_delay": 86400 * 7.39186601562955},
        {"first_crossing_rp": R_titan + 961.7220569981, "second_crossing_rp": R_titan + 27562.7530591930, "initial_delay": 86400 * 7.39287794683798},
        {"first_crossing_rp": R_titan + 718.2202112260, "second_crossing_rp": R_titan + 24375.4989483561, "initial_delay": 86400 * 7.39337826027632},
        {"first_crossing_rp": R_titan + 595.4843914986, "second_crossing_rp": R_titan + 31830.8881008898, "initial_delay": 86400 * 7.39362270646564},
        {"first_crossing_rp": R_titan + 539.9475901321, "second_crossing_rp": R_titan + 19902.5853014662, "initial_delay": 86400 * 7.39372226153959},
        {"first_crossing_rp": R_titan + 496.0579800498, "second_crossing_rp": R_titan + 15274.7764694115, "initial_delay": 86400 * 7.39378888008285},
        {"first_crossing_rp": R_titan + 450.9779007077, "second_crossing_rp": R_titan + 26653.5038316860, "initial_delay": 86400 * 7.39384377085680},
#RADI LIMIT         {"first_crossing_rp": R_titan + 433, "second_crossing_rp": R_titan + 1229.0546320099, "initial_delay": 86400 * 7.39386119530458},#RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 372, "second_crossing_rp": R_titan + 1229.0546320099, "initial_delay": 86400 * 7.39390580573488},#TITAN CRASH
    ],
    2.50: [
#        {"first_crossing_rp": R_titan + 1366.4752039420, "second_crossing_rp": R_titan + 4494.2894850809, "initial_delay": 86400 * 9.73196702888179},
#        {"first_crossing_rp": R_titan + 1029.9954107799, "second_crossing_rp": R_titan + 29191.0758345630, "initial_delay": 86400 * 9.73265946135321},
#        {"first_crossing_rp": R_titan + 699.8807077641, "second_crossing_rp": R_titan + 16938.5862300525, "initial_delay": 86400 * 9.73333821071920},
#        {"first_crossing_rp": R_titan + 620.9656800527, "second_crossing_rp": R_titan + 22833.0113061032, "initial_delay": 86400 * 9.73349325802836},
#        {"first_crossing_rp": R_titan + 575.0700705262, "second_crossing_rp": R_titan + 2887.5927243643, "initial_delay": 86400 * 9.73357508333107},
#        {"first_crossing_rp": R_titan + 537.7404513322, "second_crossing_rp": R_titan + 7653.3957991050, "initial_delay": 86400 * 9.73363423425030},
        {"first_crossing_rp": R_titan + 500.5484726921, "second_crossing_rp": R_titan + 2011.6314626235, "third_crossing_rp": R_titan + 590.9230552122, "initial_delay": 86400 * 9.73368374127172},
#        {"first_crossing_rp": R_titan + 457.1217427143, "second_crossing_rp": R_titan + 20567.5764350106, "initial_delay": 86400 * 9.73372952549530},
#RADI LIMIT         {"first_crossing_rp": R_titan + 445, "second_crossing_rp": R_titan + 5112.7952800644, "initial_delay": 86400 * 9.73374288772710},#RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 373, "second_crossing_rp": R_titan + 5112.7952800644, "initial_delay": 86400 * 9.73378234068890},#TITAN CRASH
    ],
    2.75: [
        {"first_crossing_rp": R_titan + 1367.2750118219, "second_crossing_rp": R_titan + 7276.0960620669, "initial_delay": 86400 * 9.46460598647556},
        {"first_crossing_rp": R_titan + 946.0943751269, "second_crossing_rp": R_titan + 26994.5735040835, "initial_delay": 86400 * 9.46547689977355},
        {"first_crossing_rp": R_titan + 742.9341880821, "second_crossing_rp": R_titan + 24461.3425077654, "initial_delay": 86400 * 9.46589435363994},
        {"first_crossing_rp": R_titan + 641.4352974696, "second_crossing_rp": R_titan + 16577.6108031430, "initial_delay": 86400 * 9.46609035881207},
        {"first_crossing_rp": R_titan + 609.6078535576, "second_crossing_rp": R_titan + 372.4089260053, "initial_delay": 86400 * 9.46614495316338},
        {"first_crossing_rp": R_titan + 580.3181452652, "second_crossing_rp": R_titan + 11217.6117156348, "initial_delay": 86400 * 9.46619028035707},
        {"first_crossing_rp": R_titan + 549.8466408356, "second_crossing_rp": R_titan + 4031.4365141871, "initial_delay": 86400 * 9.46623073460923},
        {"first_crossing_rp": R_titan + 515.7928534624, "second_crossing_rp": R_titan + 2079.7462410097, "third_crossing_rp": R_titan + 570.4763521514, "initial_delay": 86400 * 9.46626926903396},
        {"first_crossing_rp": R_titan + 471.0687655726, "second_crossing_rp": R_titan + 2699.6526034727, "initial_delay": 86400 * 9.46630796078291},
#RADI LIMIT        {"first_crossing_rp": R_titan + 465, "second_crossing_rp": R_titan + 1336.9011366221, "initial_delay": 86400 * 9.46631035863471},#RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 377, "second_crossing_rp": R_titan + 1336.9011366221, "initial_delay": 86400 * 9.46635793111330}, #TITAN CRASH
    ],
    3.00: [
        {"first_crossing_rp": R_titan + 1476.5823291266, "second_crossing_rp": R_titan + 3377.1355259953, "initial_delay": 86400 * 7.11884894858709},
        {"first_crossing_rp": R_titan + 892.6161754707, "second_crossing_rp": R_titan + 37137.2074482566, "initial_delay": 86400 * 7.12006044249988},
        {"first_crossing_rp": R_titan + 719.6537407873, "second_crossing_rp": R_titan + 28391.1680101009, "initial_delay": 86400 * 7.12041029018749},
        {"first_crossing_rp": R_titan + 672.3511558829, "second_crossing_rp": R_titan + 9071.6118619204, "initial_delay": 86400 * 7.12049582696361},
        {"first_crossing_rp": R_titan + 631.5323671254, "second_crossing_rp": R_titan + 19497.3247321767, "initial_delay": 86400 * 7.12056158333569},
        {"first_crossing_rp": R_titan + 610.2437628382, "second_crossing_rp": R_titan + 33586.4844975449, "initial_delay": 86400 * 7.12059126235388},
        {"first_crossing_rp": R_titan + 588.0214072842, "second_crossing_rp": R_titan + 22487.6115417414, "initial_delay": 86400 * 7.12062009285308},
        {"first_crossing_rp": R_titan + 561.7571003204, "second_crossing_rp": R_titan + 9312.3468842065, "initial_delay": 86400 * 7.12064907262872},
        {"first_crossing_rp": R_titan + 529.0265837221, "second_crossing_rp": R_titan + 33665.3466918343, "initial_delay": 86400 * 7.12067932350926},
        {"first_crossing_rp": R_titan + 483.8778646980, "second_crossing_rp": R_titan + 26831.8345634937, "initial_delay": 86400 * 7.12071160156350},
#RADI LIMIT      {"first_crossing_rp": R_titan + 488, "second_crossing_rp": R_titan + 3619.8269875502, "initial_delay": 86400 * 7.12071735691466}, #RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 378, "second_crossing_rp": R_titan + 3619.8269875502, "initial_delay": 86400 * 7.12075508099522},
    ],
    3.25: [
#SOI LIMIT       {"first_crossing_rp": R_titan + 1705, "second_crossing_rp": R_titan + 7098.8903685469, "initial_delay": 86400 * 3.06945941759303},
        {"first_crossing_rp": R_titan + 919.6719589573, "second_crossing_rp": R_titan + 35878.3206513840, "initial_delay": 86400 * 3.07108021765544},
        {"first_crossing_rp": R_titan + 742.1270079024, "second_crossing_rp": R_titan + 11324.2709462893, "initial_delay": 86400 * 3.07143241495437},
        {"first_crossing_rp": R_titan + 696.8085169630, "second_crossing_rp": R_titan + 17497.9380035169, "initial_delay": 86400 * 3.07150781046926},
        {"first_crossing_rp": R_titan + 671.7198569782, "second_crossing_rp": R_titan + 1866.7473777996, "initial_delay": 86400 * 3.07154354042440},
        {"first_crossing_rp": R_titan + 643.3322403666, "second_crossing_rp": R_titan + 16968.0827977221, "initial_delay": 86400 * 3.07158017137278},
        {"first_crossing_rp": R_titan + 626.3959708085, "second_crossing_rp": R_titan + 15087.9366108144, "initial_delay": 86400 * 3.07159939382350},
        {"first_crossing_rp": R_titan + 606.7285498483, "second_crossing_rp": R_titan + 1170.9763008737, "initial_delay": 86400 * 3.07161965665499},
        {"first_crossing_rp": R_titan + 582.2109910142, "second_crossing_rp": R_titan + 24403.1044955613, "initial_delay": 86400 * 3.07164158972718},
        {"first_crossing_rp": R_titan + 550.2590974142, "second_crossing_rp": R_titan + 38364.1558968402, "initial_delay": 86400 * 3.07166575877676},
        {"first_crossing_rp": R_titan + 502.4431097718, "second_crossing_rp": R_titan + 25374.4570269394, "initial_delay": 86400 * 3.07169294808360},
#RADI LIMIT        {"first_crossing_rp": R_titan + 522, "second_crossing_rp": R_titan + 3382.5368659681, "initial_delay": 86400 * 3.07168648245964}, #RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 383, "second_crossing_rp": R_titan + 3382.5368659681, "initial_delay": 86400 * 3.07173184258699},
    ],
    3.50: [
#SOI LIMIT        {"first_crossing_rp": R_titan + 1045, "second_crossing_rp": R_titan + 17863.5445236075, "initial_delay": 86400 * -2.38190653046270},
        {"first_crossing_rp": R_titan + 784.1899790035, "second_crossing_rp": R_titan + 38042.1948772280, "initial_delay": 86400 * -2.38138969553969},
        {"first_crossing_rp": R_titan + 731.1793771353, "second_crossing_rp": R_titan + 28974.0314986809, "initial_delay": 86400 * -2.38130907721950},
        {"first_crossing_rp": R_titan + 703.0405928583, "second_crossing_rp": R_titan + 33143.6900076446, "initial_delay": 86400 * -2.38127293836228},
        {"first_crossing_rp": R_titan + 684.3910926960, "second_crossing_rp": R_titan + 23531.1379297059, "initial_delay": 86400 * -2.38125203642780},
        {"first_crossing_rp": R_titan + 661.0610549070, "second_crossing_rp": R_titan + 30247.0521509635, "initial_delay": 86400 * -2.38122810374134},
        {"first_crossing_rp": R_titan + 645.7120237337, "second_crossing_rp": R_titan + 20852.1594912271, "initial_delay": 86400 * -2.38121456864139},
        {"first_crossing_rp": R_titan + 627.7035177912, "second_crossing_rp": R_titan + 37157.7935014987, "initial_delay": 86400 * -2.38119953312241},
        {"first_crossing_rp": R_titan + 604.2253591650, "second_crossing_rp": R_titan + 38798.8335268972, "initial_delay": 86400 * -2.38118261229808},
        {"first_crossing_rp": R_titan + 571.6472451740, "second_crossing_rp": R_titan + 23487.9019437596, "initial_delay": 86400 * -2.38116315415265},
        {"first_crossing_rp": R_titan + 521.5240985698, "second_crossing_rp": R_titan + 21587.7414922222, "initial_delay": 86400 * -2.38114020482305},
#RADI LIMIT         {"first_crossing_rp": R_titan + 562, "second_crossing_rp": R_titan + 10476.5957770165, "initial_delay": 86400 * -2.38115610979585}, #RADI LIMIT  
#TITAN CRASH         {"first_crossing_rp": R_titan + 385, "second_crossing_rp": R_titan + 10476.5957770165, "initial_delay": 86400 * -2.38110647193592},     
    ],
    3.75: [
#SOI LIMIT        {"first_crossing_rp": R_titan + 860, "second_crossing_rp": R_titan + 9082.9770422901, "initial_delay": 86400 * 6.94004541027769},
        {"first_crossing_rp": R_titan + 785.4384342723, "second_crossing_rp": R_titan + 27850.6662689758, "initial_delay": 86400 * 6.94015340597653},
        {"first_crossing_rp": R_titan + 751.4657747329, "second_crossing_rp": R_titan + 21799.8384702975, "initial_delay": 86400 * 6.94019171969451},
        {"first_crossing_rp": R_titan + 728.7448222546, "second_crossing_rp": R_titan + 36805.9976198815, "initial_delay": 86400 * 6.94021424547617},
        {"first_crossing_rp": R_titan + 712.7591797827, "second_crossing_rp": R_titan + 16430.0072203307, "initial_delay": 86400 * 6.94022837062832},
        {"first_crossing_rp": R_titan + 690.8283224412, "second_crossing_rp": R_titan + 11902.2349070784, "initial_delay": 86400 * 6.94024549644634},
        {"first_crossing_rp": R_titan + 676.4834991151, "second_crossing_rp": R_titan + 24353.4579671037, "initial_delay": 86400 * 6.94025564914435},
        {"first_crossing_rp": R_titan + 657.7820496941, "second_crossing_rp": R_titan + 12353.4579671037, "initial_delay": 86400 * 6.94026723662850},
        {"first_crossing_rp": R_titan + 634.3757475761, "second_crossing_rp": R_titan + 21884.6182867755, "initial_delay": 86400 * 6.94028067627247},
        {"first_crossing_rp": R_titan + 600.4214262418, "second_crossing_rp": R_titan + 30311.8368843430, "initial_delay": 86400 * 6.94029672085612},
       
#RADI LIMIT         {"first_crossing_rp": R_titan + 615, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 6.94028880100885},#RADI LIMIT 
#TITAN CRASH        {"first_crossing_rp": R_titan + 390, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 6.94034604574189},
    ],
    4.00: [
#SOI LIMIT        {"first_crossing_rp": R_titan + 869, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * -0.67522608556601}, #SOI LIMIT
        {"first_crossing_rp": R_titan + 816.0564996124, "second_crossing_rp": R_titan + 13158.9942751456, "initial_delay": 86400 * -0.67516769176641},
        {"first_crossing_rp": R_titan + 786.9450596074, "second_crossing_rp": R_titan + 28903.4620714261, "initial_delay": 86400 * -0.67514297847242},
        {"first_crossing_rp": R_titan + 765.6155674248, "second_crossing_rp": R_titan + 37728.3616317796, "initial_delay": 86400 * -0.67512721987209},
        {"first_crossing_rp": R_titan + 750.0572834143, "second_crossing_rp": R_titan + 31585.4203352419, "initial_delay": 86400 * -0.67511688070687},
        {"first_crossing_rp": R_titan + 728.9291282696, "second_crossing_rp": R_titan + 26399.1857614335, "initial_delay": 86400 * -0.67510391450443},
        {"first_crossing_rp": R_titan + 714.2834752644, "second_crossing_rp": R_titan + 37291.0485714519, "initial_delay": 86400 * -0.67509600940611},
#RADI LIMIT        {"first_crossing_rp": R_titan + 685, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * -0.67508133191035},
#TITAN CRASH        {"first_crossing_rp": R_titan + 395, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * -0.67501954417108},
    ],
    4.25: [
#SOI LIMIT        {"first_crossing_rp": R_titan + 907, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 6.81623020984830}, #LIMIT TO STAY IN SOI
 #       {"first_crossing_rp": R_titan + 861.2044660553, "second_crossing_rp": R_titan + 30322.0294620604, "initial_delay": 86400 * 6.81626718924519},
 #       {"first_crossing_rp": R_titan + 832.9739871466, "second_crossing_rp": R_titan + 27000.0359393262, "initial_delay": 86400 * 6.81628487218152},
#RADI LIMIT       {"first_crossing_rp": R_titan + 770, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 6.81631586519228},#RADI LIMIT
       {"first_crossing_rp": R_titan + 440, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 6.81638697752264}
    ],
    4.50: [
#        {"first_crossing_rp": R_titan + 950, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * -2.36292813734977}, #SOI LIMIT
        {"first_crossing_rp": R_titan + 870, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * -2.36288639646971},
    ],
    4.75: [
        {"first_crossing_rp": R_titan + 400, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 3.78079646074548},
    ],
    5.00: [
        {"first_crossing_rp": R_titan + 400, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 9.38519012831494},
    ],
    5.25: [
        {"first_crossing_rp": R_titan + 400, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * -1.43783823883067},
    ],
    5.50: [
     {"first_crossing_rp": R_titan + 1247, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 3.27876162456497}, #SOI LIMIT 
     #{"first_crossing_rp": R_titan + 735, "second_crossing_rp": R_titan + 2705.7389250334, "initial_delay": 86400 * 3.27882462836744},  
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
        check_second_crossing   = True
        check_third_crossing    = False
        check_fourth_crossing   = False
        r_p_initial         = cfg ["first_crossing_rp"]
   #     r_p_second_crossing = R_titan + 1200 
        r_p_second_crossing = cfg ["second_crossing_rp"]
        r_p_third_crossing = R_titan + 650
   #     r_p_third_crossing  = cfg ["third_crossing_rp"]
    #    r_p_fourth_crossing =  cfg ["fourth_crossing_rp"] 
        r_p_fourth_crossing = R_titan + 1500 

        delay_time_after_second = None
        delay_time_after_third = None
        delay_time_after_fourth = None
        r_p_after_second = None
        r_p_after_third = None
        r_p_after_fourth = None

        #Adjust the spacecraft orbit for crossings
        if check_second_crossing:
            r_p_after_second, delay_time_after_second, full_trajectory = adjust_r_p_to_match_crossing(
                v_inf=v_inf_value,
                target_crossing_number=3, 
                target_crossing_distance=r_p_second_crossing,
                r_p_initial=r_p_initial,
                tolerance_angle=2.0,
                tolerance_distance = 25000.0
            )

        if check_third_crossing:
            if r_p_after_second is None:
                r_p_after_third, delay_time_after_third, full_trajectory = adjust_r_p_to_match_crossing(
                    v_inf=v_inf_value,
                    target_crossing_number=6,
                    target_crossing_distance=r_p_third_crossing,
                    r_p_initial=r_p_initial,
                    tolerance_angle=2.0,
                    tolerance_distance= 300.0
                )
            else:
                r_p_after_third, delay_time_after_third, full_trajectory = adjust_r_p_to_match_crossing(
                    v_inf=v_inf_value,
                    target_crossing_number=6,
                    target_crossing_distance=r_p_third_crossing,
                    r_p_initial=r_p_after_second,
                    tolerance_angle=2.0,
                    tolerance_distance= 300.0
                )

        if check_fourth_crossing:
            if r_p_after_third is None:
                r_p_after_fourth, delay_time_after_fourth, full_trajectory = adjust_r_p_to_match_crossing(
                    v_inf=v_inf_value,
                    target_crossing_number=8,
                    target_crossing_distance=r_p_fourth_crossing,
                    r_p_initial=r_p_initial,
                    tolerance_angle=2.0,
                    tolerance_distance= 1000.0
                )
            else:
                r_p_after_fourth, delay_time_after_fourth, full_trajectory = adjust_r_p_to_match_crossing(
                    v_inf=v_inf_value,
                    target_crossing_number=8,
                    target_crossing_distance=r_p_fourth_crossing,
                    r_p_initial=r_p_after_third,
                    tolerance_angle=2.0,
                    tolerance_distance= 1000.0
                )

        # If delay_time_after_third is None, set some default
        r_p_final = r_p_after_fourth if r_p_after_fourth is not None else r_p_after_third if r_p_after_third is not None else r_p_after_second 
        final_delay = delay_time_after_fourth if delay_time_after_fourth is not None else delay_time_after_third if delay_time_after_third is not None else delay_time_after_second if delay_time_after_second is not None else 0

        result = {
            'v_inf': v_inf_value,
            'r_p': r_p_final,
            'r_p_third_crossing_target': r_p_third_crossing,
            'r_p_slow': r_p_final,
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
            rp_labels = [
                f"v_inf={trajectory_data[rp][0]:.2f} km/s, Flyby altitude={(rp - R_titan):.2f}km"
                for rp in rp_values
            ]

            # ===== CREATE CHECKBOXES (no scrolling) =====
            # Make the checkbox area a bit larger vertically.
            # If the list is still cut off, increase the height fraction in y0/h arguments.
            rp_checkbox_ax = plt.axes([0.01, 0.15, 0.20, 0.7], frameon=False)
            rp_checkbox = CheckButtons(rp_checkbox_ax, rp_labels, [False] * len(rp_labels))

            # Initialize variables
            trajectory_lines = {}

            # ===== SATURN ======
            saturn = plt.Circle((0, 0), R_saturn / 1e6, color='yellow', fill=True, label='Saturn')
            ax.add_patch(saturn)
            Sphere_of_Influence_Saturn = plt.Circle((0, 0), R_SOI_saturn / 1e6, color='black',
                                                fill=False, alpha=0.6, linestyle='--', label='')
            ax.add_patch(Sphere_of_Influence_Saturn)

            # ===== TITAN & ORBIT ======
            num_points = 360 * 6
            titan_orbit_x = []
            titan_orbit_y = []
            time_for_orbit_titan = np.linspace(0, T_titan, num_points)
            for t_it in time_for_orbit_titan:
                x_titan, y_titan, _ = get_titan_position(t_it)
                titan_orbit_x.append(x_titan / 1e6)
                titan_orbit_y.append(y_titan / 1e6)
            titan_orbit_line, = ax.plot(titan_orbit_x, titan_orbit_y, color='salmon',
                                    linestyle='--', label='Titan orbit')
            titan_circle = plt.Circle((0, 0), R_titan / 1e6, color='red', fill=True, label='')
            ax.add_patch(titan_circle)
            titan_atmosphere = plt.Circle((0, 0), (R_titan + atmo_height) / 1e6, color='salmon',
                                        fill=True, alpha=0.6, label='')
            ax.add_patch(titan_atmosphere)
            titan_soi = plt.Circle((0, 0), R_SOI_titan / 1e6, color='grey', fill=False,
                                alpha=0.6, linestyle='--', label='')
            ax.add_patch(titan_soi)

            # ===== ENCELADUS & ORBIT ======
            num_points_enc = 360 * 3
            enceladus_orbit_x = []
            enceladus_orbit_y = []
            time_for_orbit_enceladus = np.linspace(0, T_enceladus, num_points_enc)
            for t_enc in time_for_orbit_enceladus:
                x_enceladus, y_enceladus, _ = get_enceladus_position(t_enc)
                enceladus_orbit_x.append(x_enceladus / 1e6)
                enceladus_orbit_y.append(y_enceladus / 1e6)
            enceladus_orbit_line, = ax.plot(enceladus_orbit_x, enceladus_orbit_y, color='lightblue',
                                        linestyle='--', label='Enceladus orbit')
            enceladus_circle = plt.Circle((0, 0), R_enceladus / 1e6, color='lightblue',
                                        fill=True, label='')
            ax.add_patch(enceladus_circle)
            enceladus_soi = plt.Circle((0, 0), R_SOI_enceladus / 1e6, color='darkblue',
                                    fill=False, alpha=0.6, linestyle='--', label='')
            ax.add_patch(enceladus_soi)

            # ===== SPACECRAFT ======
            spacecraft_plot, = ax.plot([], [], 'bo', markersize=8, label='')

            # ===== PLOT DESIGN ======
            plot_radius = R_SOI_saturn / 1e6
            ax.set_xlim(-plot_radius, plot_radius)
            ax.set_ylim(-plot_radius, plot_radius)
            ax.set_aspect('equal', 'box')
            ax.set_xlabel('x [$10^6$ km]')
            ax.set_ylabel('y [$10^6$ km]')

            legend_handles = [saturn, titan_orbit_line, enceladus_orbit_line]
            legend_labels = ['Saturn', 'Titan orbit', 'Enceladus orbit']
            ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right')

            max_time = max([(data[1].t[-1] / 86400) for data in trajectory_data.values()])
            time_step = 0.001
            ax_slider = plt.axes([0.25, 0.05, 0.65, 0.02], facecolor='lightgoldenrodyellow')
            time_slider = mwidgets.Slider(ax_slider, 'Time (days)', 0, max_time,
                                        valinit=0, valstep=time_step, valfmt="%.4f")

            # Update plot when time slider changes
            def update_plot(val):
                current_time = time_slider.val
                x_titan, y_titan, _ = get_titan_position(current_time * 86400)
                x_titan /= 1e6
                y_titan /= 1e6
                titan_circle.center = (x_titan, y_titan)
                titan_atmosphere.center = (x_titan, y_titan)
                titan_soi.center = (x_titan, y_titan)

                # --- ENCELADUS UPDATE ---
                x_enc, y_enc, _ = get_enceladus_position(current_time * 86400)
                x_enc /= 1e6
                y_enc /= 1e6
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
                    x_spacecraft = r * np.cos(phi) / 1e6
                    y_spacecraft = r * np.sin(phi) / 1e6
                    spacecraft_plot.set_data([x_spacecraft], [y_spacecraft])
                else:
                    spacecraft_plot.set_data([], [])

                fig.canvas.draw_idle()

            time_slider.on_changed(update_plot)

            # Keypress events (optional)
            def on_key(event):
                if event.key == 'd':
                    current_val = time_slider.val
                    time_slider.set_val(min(time_slider.valmax, current_val + time_slider.valstep))
                elif event.key == 'a':
                    current_val = time_slider.val
                    time_slider.set_val(max(time_slider.valmin, current_val - time_slider.valstep))
                elif event.key == 'w':
                    current_val = time_slider.val
                    time_slider.set_val(min(time_slider.valmax, current_val + time_slider.valstep * 30))
                elif event.key == 'x':
                    current_val = time_slider.val
                    time_slider.set_val(max(time_slider.valmin, current_val - time_slider.valstep * 30))

            fig.canvas.mpl_connect('key_press_event', on_key)

            # Update trajectory visibility when a checkbox is clicked
            def update_rp_visibility(label):
                status = rp_checkbox.get_status()  # Boolean list for each checkbox
                for i, checked in enumerate(status):
                    selected_rp = rp_values[i]
                    v_inf_line, full_trajectory = trajectory_data[selected_rp]
                    if selected_rp not in trajectory_lines:
                        if checked:
                            r = full_trajectory.y[0, :]
                            phi = full_trajectory.y[1, :]
                            subsample_factor = 1
                            subsampled_indices = np.arange(0, len(r), subsample_factor)
                            x = (r * np.cos(phi))[subsampled_indices]
                            y = (r * np.sin(phi))[subsampled_indices]
                            # Create the trajectory line
                            line, = ax.plot(
                                x / 1e6, y / 1e6, linewidth=0.4,
                                label=f"Trajectory (r_p: {(selected_rp - R_titan):.2f} km)",
                                visible=True
                            )
                            trajectory_lines[selected_rp] = line
                    else:
                        trajectory_lines[selected_rp].set_visible(checked)

                # Update legend to include Saturn, Titan Orbit, Enceladus Orbit,
                # and all visible trajectories
                legend_handles = [saturn, titan_orbit_line, enceladus_orbit_line]
                legend_labels = ['Saturn', 'Titan orbit', 'Enceladus orbit']
                for rp, line in trajectory_lines.items():
                    if line.get_visible():
                        v_inf, _ = trajectory_data[rp]
                        line.set_label(f"Spacecraft trajectory\nFlyby altitude: {(rp - R_titan):.2f} km\nv_inf: {v_inf:.2f} km/s")
                        legend_handles.append(line)
                        legend_labels.append(line.get_label())

                ax.legend(handles=legend_handles, labels=legend_labels, loc='upper right')
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
        config_labels = [f"v_inf: {v:.2f} km/s \n1st flyby altitude: {r_p - R_titan:.2f} km" for v, r_p in sorted_configs]

        # Create the figure and axis
        fig, ax = plt.subplots(figsize=(8, 6))

        # Position axes for the checkboxes
        config_check_ax = plt.axes([0.91, 0.05, 0.08, 0.2], frameon=False)
        parameter_labels = ['Orbital Energy', 'Distance to Saturn', 'Distance to Titan', 'Velocity', 'Orbital Period', 'Mass']
        parameter_check_ax = plt.axes([0.91, 0.3, 0.08, 0.6], frameon=False)
        parameter_check = CheckButtons(parameter_check_ax, parameter_labels, [False] * len(parameter_labels))
        config_check = CheckButtons(config_check_ax, config_labels, [False] * len(config_labels))


        # Global sets to store selected parameters and configs
        selected_params = set()          # e.g. {"Velocity"}
        selected_configs = []            # list of (v_inf, r_p) tuples

        # Helper function: compute and subsample trajectory data for a given configuration,
        # and truncate the arrays if distance to Titan hits 0 or below.
        def compute_config_data(config):
            v_inf, r_p = config
            selected_result = config_map[config]
            full_trajectory = selected_result['full_trajectory']

            time = full_trajectory.t / 86400  # Convert time to days
            r = full_trajectory.y[0, :]
            phi = full_trajectory.y[1, :]
            rhor = full_trajectory.y[2, :]
            rhophi = full_trajectory.y[3, :]
            mass = full_trajectory.y[4, :]

            # Compute Titan's position for each time step
            titan_positions = [get_titan_position(t * 86400)[:2] for t in time]
            x_titan = np.array([pos[0] for pos in titan_positions])
            y_titan = np.array([pos[1] for pos in titan_positions])

            # Spacecraft position in Cartesian coordinates
            x_spacecraft = r * np.cos(phi)
            y_spacecraft = r * np.sin(phi)

            # Compute quantities
            v = np.sqrt(rhor**2 + (r * rhophi)**2)  # Velocity [km/s]
            E = v**2 / 2 - mu_saturn / r            # Orbital Energy
            distance_to_saturn = r
            r_titan_spacecraft = np.sqrt((x_spacecraft - x_titan)**2 + (y_spacecraft - y_titan)**2)
            distance_to_titan = r_titan_spacecraft - R_titan

            # Subsample data for plotting
            subsample_factor = 5
            subsampled_indices = np.arange(0, len(time), subsample_factor)
            time_sub = time[subsampled_indices]
            distance_to_titan_sub = distance_to_titan[subsampled_indices]
            distance_to_saturn_sub = distance_to_saturn[subsampled_indices]
            v_sub = v[subsampled_indices]
            E_sub = E[subsampled_indices]
            mass_sub = mass[subsampled_indices]

            # Truncate data if distance to Titan reaches 0 or below.
            hit_zero_indices = np.where(distance_to_titan_sub <= 0)[0]
            if hit_zero_indices.size > 0:
                cut_idx = hit_zero_indices[0]
                time_sub = time_sub[:cut_idx]
                distance_to_titan_sub = distance_to_titan_sub[:cut_idx]
                distance_to_saturn_sub = distance_to_saturn_sub[:cut_idx]
                v_sub = v_sub[:cut_idx]
                E_sub = E_sub[:cut_idx]
                mass_sub = mass_sub[:cut_idx]

            return {
                'time_sub': time_sub,
                'distance_to_titan_sub': distance_to_titan_sub,
                'distance_to_saturn_sub': distance_to_saturn_sub,
                'v_sub': v_sub,
                'E_sub': E_sub,
                'mass_sub': mass_sub
            }

        # Function to shade regions when the spacecraft is within Titan's atmosphere
        def shade_atmosphere_regions(ax, time_sub, distance_to_titan_sub, atmo_height):
            in_atmosphere = distance_to_titan_sub <= atmo_height
            start = None
            for i in range(len(time_sub)):
                if in_atmosphere[i] and start is None:
                    start = time_sub[i]
                elif (not in_atmosphere[i]) and (start is not None):
                    end = time_sub[i]
                    ax.axvspan(start, end, color='salmon', alpha=0.4)
                    start = None
            if start is not None:
                ax.axvspan(start, time_sub[-1], color='salmon', alpha=0.4)

        # Update the plot using the current set of selected configs and parameters.
        def update_plot():
            ax.clear()
            # This flag ensures the atmosphere boundary is labeled only once.
            atmo_boundary_plotted = False

            # For each selected configuration, compute data and plot each selected parameter.
            for config in selected_configs:
                data = compute_config_data(config)
                time_sub = data['time_sub']
                # Get a label for this config for use in the legend.
                idx = sorted_configs.index(config)
                base_label = config_labels[idx]
                if 'Orbital Energy' in selected_params:
                    ax.plot(time_sub, data['E_sub'], label=f"Orbital energy\n{base_label}")
                    ax.set_ylabel('Energy (MJ/kg)')
                    shade_atmosphere_regions(ax, time_sub, data['distance_to_titan_sub'], atmo_height)
                if 'Distance to Saturn' in selected_params:
                    ax.plot(time_sub, data['distance_to_saturn_sub'], label=f"Distance to Saturn\n{base_label}")
                    ax.set_ylabel('Distance to Saturn (km)')
                    ax.set_yscale('log')
                    ax.axhline(y=a_titan, color='red', linestyle='--', label='Titan')
                    ax.axhline(y=a_enceladus, color='lightblue', linestyle='--', label='Enceladus')
                    shade_atmosphere_regions(ax, time_sub, data['distance_to_titan_sub'], atmo_height)
                if 'Distance to Titan' in selected_params:
                    ax.plot(time_sub, data['distance_to_titan_sub'], label=f"Distance to Titan\n{base_label}")
                    ax.set_ylabel('Distance to Titan (km)')
                    # Only add the atmosphere boundary once.
                    if not atmo_boundary_plotted:
                        ax.axhline(y=atmo_height, color='gray', linestyle='--', label='Atmospheric interface')
                        atmo_boundary_plotted = True
                    else:
                        ax.axhline(y=atmo_height, color='gray', linestyle='--')
                    shade_atmosphere_regions(ax, time_sub, data['distance_to_titan_sub'], atmo_height)
                if 'Velocity' in selected_params:
                    ax.plot(time_sub, data['v_sub'], label=f"Velocity\n{base_label}")
                    ax.set_ylabel('Velocity (km/s)')
                    shade_atmosphere_regions(ax, time_sub, data['distance_to_titan_sub'], atmo_height)
                if 'Orbital Period' in selected_params:
                    # Compute the semi-major axis from orbital energy:
                    # (Ensure that the energy is negative so that a is positive.)
                    a_sub = -mu_saturn / (2 * data['E_sub'])
                    # Calculate the orbital period using the semi-major axis:
                    T_orbit = 2 * np.pi * np.sqrt(a_sub**3 / mu_saturn)
                    # Convert the orbital period from seconds to days
                    T_orbit_days = T_orbit / 86400
                    # Filter out values greater than 50 days:
                    valid_mask = T_orbit_days <= 100
                    if np.any(valid_mask):
                        ax.plot(time_sub[valid_mask], T_orbit_days[valid_mask], label=f"Orbital period\n{base_label}")
                    # Plot horizontal reference lines only if they fall below 50 days
                    for multiple in range(1, 11):
                        ref_val = multiple * T_enceladus/86400
                        if ref_val <= 50:
                            # Use a different style for the first two lines
                            linestyle = '--' if multiple <= 2 else '--'
                            color = 'gray' if multiple == 1 else 'grey'
                            label = f'x times Enceladus orbital period' if multiple == 1 else None
                            ax.axhline(y=ref_val, color=color, linestyle=linestyle, label=label)
                    ax.set_ylabel('Orbital Period (days)')
                    shade_atmosphere_regions(ax, time_sub, data['distance_to_titan_sub'], atmo_height)
                if 'Mass' in selected_params:
                    ax.plot(time_sub, data['mass_sub'], label=f"Mass\n{base_label}")
                    ax.set_ylabel('Mass (kg)')
            ax.set_xlabel('Time (days)')
            ax.legend(loc='upper right')
            ax.grid(True)
            plt.draw()

        # Callback for parameter checkboxes: update the selected parameters then re-plot.
        def on_parameter_clicked(label):
            if label in selected_params:
                selected_params.remove(label)
            else:
                selected_params.add(label)
            update_plot()

        parameter_check.on_clicked(on_parameter_clicked)

        # Callback for config checkboxes: update the list of selected configs then re-plot.
        def on_config_clicked(label):
            status = config_check.get_status()  # Boolean list matching config_labels order.
            selected_configs.clear()
            for idx, checked in enumerate(status):
                if checked:
                    selected_configs.append(sorted_configs[idx])
            update_plot()

        config_check.on_clicked(on_config_clicked)

        plt.show()
    #endregion

    #region Plot Atmosphere
    if Plot_Atmosphere:
        heights = np.linspace(0, atmo_height, 500)
        densities_km3 = [atmospheric_density(height + R_titan) for height in heights]
        densities_m3 = [density * 1e-9 for density in densities_km3]  # Convert from kg/km^3 to kg/m^3

        plt.figure(figsize=(8, 6))
        plt.plot(densities_m3, heights, label="Atmospheric density")

        # Configure x-axis (logarithmic) and y-axis
        plt.xscale('log')
        plt.yscale('linear')
        plt.xlabel('Atmospheric Density (kg/m³)')
        plt.ylabel('Altitude (km)')

        # Set x-axis limits and ticks for every exponential step
        plt.xlim(1e-12, 1e1)
        plt.xticks([10**i for i in range(-12, 2)])  # Create ticks from 10^-12 to 10^1
        plt.grid(which='both', axis='x', linestyle='--', linewidth=0.5)

        # Set y-axis limits and ticks for every 100 km step
        plt.ylim(0, 1400)
        plt.yticks(range(0, 1401, 100))  # Create ticks every 100 km
        plt.grid(which='major', axis='y', linestyle='--', linewidth=0.5)

    #endregion

    #region Plot Comparison
    if Plot_Comparison:
        #region Temporary
        # ---------------------------
        # 1) GATHER AND ORGANIZE DATA
        # ---------------------------
        grouped_data = defaultdict(list)

        all_v_inf = []
        all_r_p   = []
        all_energy = []  # orbital energy values

        for result in full_trajectory_results:
            v_inf = result['v_inf']
            # Periapsis altitude above Titan:
            r_p_aboveTitan = result['r_p'] - R_titan

            # Compute orbital energy after the first flyby.
            # We choose the point where the radial velocity changes from positive to negative.
            trajectory = result.get('full_trajectory', None)
            orbital_energy = None
            if trajectory is not None:
                r    = trajectory.y[0, :]
                rhor = trajectory.y[2, :]
                # Ensure we have the tangential velocity component
                if trajectory.y.shape[0] > 3:
                    rhophi = trajectory.y[3, :]
                else:
                    rhophi = np.zeros_like(r)
                for i in range(1, len(rhor)):
                    if rhor[i-1] > 0 and rhor[i] <= 0:
                        v_val = np.sqrt(rhor[i]**2 + (r[i]*rhophi[i])**2)
                        orbital_energy = v_val**2/2 - mu_saturn/r[i]
                        break

            grouped_data[v_inf].append((r_p_aboveTitan, orbital_energy))

            all_v_inf.append(v_inf)
            all_r_p.append(r_p_aboveTitan)
            all_energy.append(orbital_energy if orbital_energy is not None else np.nan)

        all_v_inf = np.array(all_v_inf)
        all_r_p   = np.array(all_r_p)
        all_energy = np.array(all_energy)

        sorted_v_infs = sorted(grouped_data.keys())

        # ------------------------------------
        # 2) SCATTER-PLOT ALL DATA
        # ------------------------------------
        plt.figure(figsize=(10, 6))
        plt.xlabel('$v_{inf}$ [km/s]')
        plt.ylabel('1st Flyby Altitude [km]')
        plt.grid(True, linestyle='--', linewidth=0.5)

        valid_mask = ~np.isnan(all_energy)
        if np.any(valid_mask):
            energy_min = np.nanmin(all_energy[valid_mask])
            energy_max = np.nanmax(all_energy[valid_mask])
        else:
            energy_min = 0
            energy_max = 1

        # Use PowerNorm for nonlinear color scaling (adjust gamma as needed)
        norm = PowerNorm(gamma=1.2, vmin=energy_min, vmax=energy_max)
        cmap = plt.cm.plasma

        sc = plt.scatter(
            all_v_inf[valid_mask],
            all_r_p[valid_mask],
            c=all_energy[valid_mask],
            cmap=cmap,
            norm=norm,
            edgecolor='black',
            s=60,
        )
        cbar = plt.colorbar(sc, ax=plt.gca())
        cbar.set_label('Orbital Energy [MJ/kg]')
    #endregion
        upper_bound_v = np.array([3.25, 3.5, 3.75, 4.0, 4.25, 4.50])  # velocities in km/s
        upper_bound_alt = np.array([1705, 1045, 860, 868, 907, 972])         # corresponding altitudes in km
        v_dense_upper = np.linspace(upper_bound_v.min(), upper_bound_v.max(), 200)
        spline_upper = make_interp_spline(upper_bound_v, upper_bound_alt, k=3)
        alt_dense_upper = spline_upper(v_dense_upper)
        plt.plot(v_dense_upper, alt_dense_upper, linestyle='--', color='black', linewidth=2, label='Boundary to stay in Saturnian system')
        plt.axhline(y=atmo_height, color='red', linestyle='--', label='Atmospheric interface')
        lower_bound_v   = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.50])         # velocities in km/s
        lower_bound_alt = np.array([432,  419,  414, 412, 413,  417, 423,  433, 445,  465, 488,  522, 562,  615, 685,  770,  870])              # corresponding altitudes in km
        # Generate a dense velocity range and interpolate the lower boundary curve
        v_dense = np.linspace(lower_bound_v.min(), lower_bound_v.max(), 200)
        spline = make_interp_spline(lower_bound_v, lower_bound_alt, k=3)
        alt_dense = spline(v_dense)
        plt.plot(v_dense, alt_dense, linestyle='--', color='gray', linewidth=2, label='Radiative TPS limit')
        safety_bound_v   = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.50])         # velocities in km/s
        safety_bound_alt = np.array([428,  417, 412,  409, 407,  408, 409,  409, 410,  415, 416,  421,  424, 429, 435,  440,  445])              # corresponding altitudes in km
        v_dense_safety = np.linspace(safety_bound_v.min(), safety_bound_v.max(), 200)
        spline = make_interp_spline(safety_bound_v, safety_bound_alt, k=3)
        alt_dense_safety = spline(v_dense_safety)
        plt.plot(v_dense_safety, alt_dense_safety, linestyle='--', color='black', linewidth=2, label='Lowest altitude (incl. safety margin)')
        # ---------------------------

        # ---------------------------
        # 3) BUILD MULTI-VELOCITY FAMILIES
        # ---------------------------
        # Set a tolerance for energy differences (adjust as needed)
        tolerance_energy = 0.15 #0.1 working good, trying to enhance accuracy further  

        # For each velocity, store a list of [r_p, energy, used=False]
        data_dict = {}
        for v in sorted_v_infs:
            # Sort each velocity's data by energy for consistent grouping
            sorted_points = sorted(grouped_data[v], key=lambda x: x[1] if x[1] is not None else float('inf'))
            data_dict[v] = [[p[0], p[1], False] for p in sorted_points]

        families = []

        def find_family_starting_at(v_idx, pt_idx):
            """
            Build a chain (family) starting at data_dict[sorted_v_infs[v_idx]][pt_idx].
            We chain forward in v_inf if the difference in orbital energy is within tolerance.
            Returns a list of (v_inf, r_p, energy).
            """
            chain = []
            cur_v_inf = sorted_v_infs[v_idx]
            r_p_cur, energy_cur, _ = data_dict[cur_v_inf][pt_idx]
            data_dict[cur_v_inf][pt_idx][2] = True
            chain.append((cur_v_inf, r_p_cur, energy_cur))

            for next_v_idx in range(v_idx+1, len(sorted_v_infs)):
                v_next = sorted_v_infs[next_v_idx]
                best_idx = -1
                best_diff = float('inf')
                for j, (r_p_nxt, energy_nxt, used_nxt) in enumerate(data_dict[v_next]):
                    if used_nxt or energy_nxt is None:
                        continue
                    diff = abs(energy_cur - energy_nxt)
                    if diff < best_diff:
                        best_diff = diff
                        best_idx = j
                if best_idx >= 0 and best_diff <= tolerance_energy:
                    r_p_match, energy_match, _ = data_dict[v_next][best_idx]
                    data_dict[v_next][best_idx][2] = True
                    chain.append((v_next, r_p_match, energy_match))
                    energy_cur = energy_match
                else:
                    break
            return chain

        for v_idx, v_inf in enumerate(sorted_v_infs):
            for pt_idx, (r_p_val, energy_val, used_flag) in enumerate(data_dict[v_inf]):
                if used_flag or energy_val is None:
                    continue
                new_family = find_family_starting_at(v_idx, pt_idx)
                if len(new_family) > 0:
                    families.append(new_family)

        # ---------------------------
        # 4) PLOT EACH FAMILY WITH A SPLINE
        # ---------------------------
        def plot_family_spline(family):
            # Sort the chain by v_inf
            fam_sorted = sorted(family, key=lambda x: x[0])
            v_array  = np.array([pt[0] for pt in fam_sorted])
            rp_array = np.array([pt[1] for pt in fam_sorted])
            energy_arr = np.array([pt[2] for pt in fam_sorted])
            mean_energy = np.nanmean(energy_arr)
            color = cmap(norm(mean_energy)) if not np.isnan(mean_energy) else 'gray'
            if len(fam_sorted) >= 3:
                try:
                    sp = PchipInterpolator(v_array, rp_array)
                    v_dense = np.linspace(v_array.min(), v_array.max(), 100)
                    rp_dense = sp(v_dense)
                    plt.plot(v_dense, rp_dense, color=color, linewidth=2)
                except ValueError:
                    plt.plot(v_array, rp_array, color=color, linewidth=2)
            else:
                if len(fam_sorted) == 2:
                    plt.plot(v_array, rp_array, color=color, linewidth=2)

        for fam in families:
            plot_family_spline(fam)

        plt.xlim(0, max(sorted_v_infs)*1.1 if sorted_v_infs else 1)
        if len(all_r_p) > 0:
            plt.ylim(0, max(all_r_p)*1.1)

        legend = plt.legend(loc='upper right')
        current_size = legend.get_texts()[0].get_fontsize()  # get current font size
        new_size = current_size * 1.2  # scale by 20%
        for text in legend.get_texts():
            text.set_fontsize(new_size)

        plt.show()
    #endregion

    #region Plot Heating Rate (Instantaneous Values, W/cm² using Sutton-Graves)
    if Plot_HeatingRate:
        plt.figure(figsize=(7, 6))
        
        # Assume a nose radius (R_n) in meters; adjust as needed
        nose_radius = np.sqrt((A * 1e6) / np.pi)  # For example, ~1.875 m

        # Define parameters for radiative heating correlation:
        k_rad = 1e-1         # Empirical constant for radiative heating (W/m² units before conversion)
        kappa = 2            # Atmosphere-specific constant; κ = 2 for Titan

        def shade_high_heating(ax, times, heating, threshold):
            """
            Shades the time intervals on the given axes where the heating rate exceeds the threshold.
            
            Parameters:
                ax       : The matplotlib axes object.
                times    : 1D numpy array of time values (seconds).
                heating  : 1D numpy array of heating rate values (W/cm²).
                threshold: Threshold value to check (e.g. 50.4267).
            """
            above_threshold = heating > threshold
            start = None
            for i in range(len(times)):
                if above_threshold[i] and start is None:
                    start = times[i]
                elif (not above_threshold[i]) and start is not None:
                    end = times[i]
                    ax.axvspan(start, end, color='orange', alpha=0.3, label='Thermal load above limit')
                    start = None
            if start is not None:
                ax.axvspan(start, times[-1], color='orange', alpha=0.3, label='Thermal load above limit')

        # Loop through each trajectory result (assuming only one configuration is active at a time)
        for result in full_trajectory_results:
            full_trajectory = result['full_trajectory']
            time_days = full_trajectory.t / 86400.0  # Time in days
            r = full_trajectory.y[0, :]
            phi = full_trajectory.y[1, :]
            rhor = full_trajectory.y[2, :]
            rhophi = full_trajectory.y[3, :]

            # Compute velocity in km/s then later convert to m/s
            v = np.sqrt(rhor**2 + (r * rhophi)**2)  # in km/s

            # Get Titan's position for each time step (only the x,y positions)
            titan_positions = [get_titan_position(t * 86400)[:2] for t in time_days]

            # Initialize arrays for convective and radiative heating (W/cm²)
            q_conv = np.zeros_like(r)
            q_rad = np.zeros_like(r)

            for i in range(len(r)):
                x_spacecraft = r[i] * np.cos(phi[i])
                y_spacecraft = r[i] * np.sin(phi[i])
                titan_x, titan_y = titan_positions[i]
                r_titan_spacecraft = np.sqrt((x_spacecraft - titan_x)**2 + (y_spacecraft - titan_y)**2)
                altitude = r_titan_spacecraft - R_titan  # Altitude above Titan’s surface in km

                if altitude <= atmo_height:
                    # Convert density from kg/km³ to kg/m³
                    density = atmospheric_density(r_titan_spacecraft) * 1e-9
                    # Convert velocity from km/s to m/s
                    v_m = v[i] * 1000
                    q_conv[i] = 1.7415e-8 * np.sqrt(density / nose_radius) * v_m**3  # W/cm²
                    q_rad_unc = (k_rad * density * v_m**3) / 1e4  # Uncoupled radiative heating in W/cm²
                    Gamma = 4 * k_rad  # Simplified non-dimensional parameter
                    q_rad[i] = q_rad_unc / (1 + kappa * (Gamma)**0.7)
                else:
                    q_conv[i] = np.nan
                    q_rad[i] = np.nan

            # Identify valid indices (inside atmosphere)
            valid_indices = np.where(~np.isnan(q_conv) & ~np.isnan(q_rad))[0]
            if valid_indices.size > 0:
                # Group contiguous valid indices into atmospheric phases
                groups = []
                current_group = [valid_indices[0]]
                for idx in valid_indices[1:]:
                    if idx == current_group[-1] + 1:
                        current_group.append(idx)
                    else:
                        groups.append(current_group)
                        current_group = [idx]
                groups.append(current_group)

                # Use a qualitative colormap for phase-specific colors
                phase_colors = plt.cm.Set1(np.linspace(0, 1, len(groups)))
                
                # Add a header entry for this trajectory result to appear in the legend
                header_label = f"1st flyby altitude: {result['r_p'] - R_titan:.0f} km, v_inf: {result['v_inf']:.2f} km/s"
                plt.plot([], [], color='white', label=header_label)
                
                # Process each atmospheric phase group
                for phase_num, group in enumerate(groups, start=1):
                    # Convert group times to seconds and rebaseline to start at 0 sec for the phase
                    group_time = time_days[group] * 86400  
                    group_time = group_time - group_time[0]
                    group_q_conv = q_conv[group]
                    group_q_rad = q_rad[group]
                    phase_color = phase_colors[phase_num - 1]

                    # Plot convective (solid) and radiative (dashed) heating curves
                    plt.plot(group_time, group_q_conv, color=phase_color,
                            linestyle='-', linewidth=1.5, label=f'Convective, Phase {phase_num}')
                    plt.plot(group_time, group_q_rad, color=phase_color,
                            linestyle='--', linewidth=1.5, label=f'Radiative, Phase {phase_num}')

                    # Combine the heating rates (element-wise maximum) so that any high value is captured
                    group_heating = np.maximum(group_q_conv, group_q_rad)
                    # Shade the time intervals where the heating rate exceeds 50.4267 W/cm²
                    shade_high_heating(plt.gca(), group_time, group_heating, 50.4267)
                    
                    # Annotate the phase number near the midpoint of the phase
                    mid_idx = group[len(group) // 2]
                    mid_time = (time_days[mid_idx] * 86400) - group_time[0]
                    mid_value = q_conv[mid_idx]  # Could also use q_rad or an average value
                    plt.text(mid_time, mid_value, f'{phase_num}', fontsize=10, fontweight='bold',
                            color=phase_color, verticalalignment='bottom', horizontalalignment='right')

        # Plot a horizontal line for the radiation limit (modern coatings)
        radiation_limit_cm2 = 504267 * 1e-4  
        plt.axhline(radiation_limit_cm2, color='black', linestyle='--', linewidth=1,
                    label='Radiation limit modern coatings')

        plt.xlabel('Time (seconds)')
        plt.ylabel('Heating Rate (W/cm²)')
        plt.legend(loc='upper right', fontsize=8)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()
    #endregion

    plt.show()
#endregion