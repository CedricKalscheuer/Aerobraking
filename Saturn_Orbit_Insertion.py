import numpy as np
import matplotlib.pyplot as plt
import time as tm
from concurrent.futures import ProcessPoolExecutor, as_completed
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter 
from matplotlib.widgets import CheckButtons, Slider
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema

#Ideas:
#Plot for Time in atmosphere for Titan Flyby vs. b-plane-offset (b_i+200k: TiA=11.8-12.6min, 250k: TiA=12.2-13min, 300k: TiA=12.4-13.2min depending on r_p)
#Plot for launch time window based on current Saturn to Titan constellation
#Have thrust_duration set so that when crossing Titans Orbit on the way back to Saturn the S/C will aerobrake again to reduce the velocity

#region User input
# === Rocketparameters ============================
m_0                         = 3725.0            # Initial spacecraft mass in [kg]
m_p                         = 2725.0            # Propellant mass in [kg]
thrust                      = 424.0             # Thrust in [N]
c_3                         = 30                # Characteristic energy in [km^2/s^2]
Isp                         = 321.0             # Specific impulse
c_d                         = 2.2               # Drag coefficient  
A                           = 29.3              # Cross-sectional area of the spacecraft in [m^2]
c_q                         = 0.5               # 0.5 for highest Heat Load 
# === Orbitparameters ============================
tint                        = 200               # Integration time in days
r_p_lowest                  = 50                # Lowest Periapsis tested
r_p_highest                 = 151               # Highest Periapsis tested  
r_p_step_size               = 50                # Step size between r_p_lowest and r_p_highest
r_p_orbit                   = 1500              # Periapsis of desired Orbit in [km]
r_a                         = 2000              # Apoapsis of desired Orbit in [km]
# === Outputparameters ============================
Plot_Trajectory             = True              # Plot spacecraft trajectory?
Plot_Atmosphere             = False             # Plot atmospheric density over height?
Plot_Values                 = True              # Plot panel of values?
Optimize_Thrust_Duration    = True              # Optimize Thrust duration?
Thrust_Only_Run             = False             # Calculate Thrust Only Run? (NOT CURRENTLY WORKING)
Plot_Comparison             = False             # Plot propellant consumed vs time to stable orbit? (NOT CURRENTLY WORKING)
Plot_HeatingRate            = True             # Plot Heating Rate Maximum and Average over time?
#endregion

#region Prepare the program
# General constants for the problem
g0_earth            = 9.80665               # Earth standard gravitational acceleration in [m/s^2]
v_inf               = np.sqrt(c_3)          # Hyperbolic excess speed at infinity in [km/s]
# === Saturn ============================
mu_saturn           = 37931207.8            # Gravitational parameter of Saturn in [km^3/s^2]
R_saturn            = 60268.0               # Radius of Saturn in [km]
dist_saturn_titan   = 1221870.0             # Approximate distance between Saturn and Titan in [km]
R_SOI_saturn        = 54.5 * 1e6            # Saturn's Radius of Sphere of Influence in [km]
# === Titan ============================
mu_titan            = 8978.14               # Gravitational parameter of Titan in [km^3/s^2]
R_titan             = 2574.73               # Titan Equatorial radius in [km]
R_SOI_titan         = 1200000               # Titan's Radius of standard Sphere of Influence in [km]
a_titan             = 1221870               # Semi-major axis in km
e_titan             = 0.0288                # Eccentricity
T_titan             = 2*np.pi*np.sqrt(a_titan**3 / mu_saturn)

# Start values for later Iteration
tint_short                  = 120.0         # Integration time for short runs
r_thrust_descending         = 550000        # Altitude where thrust begins for Orbit Insertion Maneuver
r_thrust_ascending          = 490000        # Altitude where thrust stops for Orbit Insertion Maneuver
thrust_duration             = 4 * 3600      # Thrust duration in seconds
thrust_duration_reduction   = 0.5 * 3600    # Reduction of thrust_duration in [s]
thrust_duration_limit       = 3 * 3600      # Lower Limit of thrust_duration reduction
tolerance                   = 5             # 5 for testing functions, 0.2 für fertige Arbeit
max_step_critical           = 5             # maximum step size during simulation for critical areas
max_step_non_critical       = 200           # maximum step size during simulation for non-critical areas                    
b_i                         = R_titan*np.sqrt(2*mu_titan/(R_titan*v_inf**2)+1)+200000 
critical_distance_titan_sim = R_titan + 75000
critical_distance_saturn_sim= R_saturn + r_thrust_descending + 50000

# === Unit-Conversion ============================
tmax                = tint * 86400                  # Convert Integration time in [s]
tmax_short          = tint_short * 86400            # Convert Integration time in [s]
A                   = A / 1e6                       # Convert Cross-Sectional area in [km^2]
thrust              = thrust / 1000.0               # Convert thrust to [kg*km/s^2]
ceff                = Isp * g0_earth / 1000.0       # Effective exhaust velocity in [km/s]
r_p_orbit           = R_titan + r_p_orbit           # Periapsis of desired Orbit including Planetradius 
r_a                 = R_titan + r_a                 # Apoapsis of desired Orbit including Planetradius 
r_p_values          = np.arange(R_titan + r_p_lowest, R_titan + r_p_highest, r_p_step_size)

# Initialize mission phase variables
orbit_insertion_maneuver    = True      # Set to false after the first Thruster burn
full_aerobraking            = True      # Set to false when r_p is raised to r_p_slow to prevent apoapsis from dropping lower than r_a
is_descending               = True      # Track if spacecraft is ascending or descending
thrust_active               = False     # Track when thrust is active for plots
stable_orbit_time           = None      # Track the time when stable orbit is reached
slowdown_start_time         = None
thrust_start_time           = None
stabilization_start_time    = None
aerobrake                   = True      # Use aerobraking for the main simulations
#endregion

#region Atmosphere
atmo_height             = 1200                                              # Height of the atmosphere boundary in [km]
def atmospheric_density(r_titan_spacecraft):
    height              = r_titan_spacecraft - R_titan                      # Altitude above Titan's surface in km
    atmo_density        = 5.38 * 1e9                                        # Titan's atmospheric density in [kg/km^3]
    atmo_density_1200   = 1.00                                              # Density at 1200 km altitude
    H                   = 1200 / np.log(atmo_density / atmo_density_1200)   # Scale height based on the model
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
            #print(f"  Orbit Insertion Ended at t: {t/86400:.2f} days after thrusting for {(t - thrust_start_time)/60:.1f} minutes.")
            return 0.0  # Throttle fully closed
        # Stop thrusting if mass reaches dry mass
        elif m <= mdry:
            orbit_insertion_maneuver = False
            thrust_active = False
            #print(f"  Thrust stopped due to fuel depletion at t: {t/86400:.2f} days after thrusting for {(t - thrust_start_time)/60:.1f} minutes.")
            return 0.0  # Throttle fully closed
        else:
            return 1.0  # Continue thrusting

    # If thrust is not active, check if we should start thrusting
    elif is_descending and r < r_thrust_descend and m > mdry:
        thrust_start_time = t
        thrust_active = True
        #print(f"    Thrust start time set to t = {thrust_start_time/86400:.2f} days")
        return 1.0  # Throttle fully open

    else:
        return 0.0  # No thrust

def handle_slowdown_aerobraking(r, m, mdry, apoapsis, periapsis, t, r_p_slow):
    global thrust_active, full_aerobraking, slowdown_start_time

    if r_p_slow - R_titan <= 950:
        dynamic_range = 2700            # r_p = 900 (1. Mal periapsis kleiner als 4200) 
    elif 951 < r_p_slow - R_titan <= 960:
        dynamic_range = 1800            # r_p = 910 (1. Mal periapsis kleiner als 3300) 
    elif 961 < r_p_slow - R_titan <= 970:       
        dynamic_range = 1200            # r_p = 920, 85 (1. Mal periapsis kleiner als 2700)
    elif 971 < r_p_slow - R_titan <= 980:       
        dynamic_range = 700             # r_p = 930, 95 (1. Mal periapsis kleiner als 2200)
    else:                       
        dynamic_range = 500             # r_p = 100 (1. Mal periapsis kleiner als 2000)

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
        return 1.0  # Throttle fully open at apoapsis once the desired apoapsis is reached to increase periapsis
    elif periapsis >= (r_p_orbit-5):
        if stable_orbit_time is None:
            stable_orbit_time = t 
        return 0.0  # Throttle fully closed once the desired periapsis (and with that the final orbit) is reached
    return 1.0 if thrust_active else 0.0  # Maintain current state

def handle_thrust_only(r, m, mdry, apoapsis, periapsis, t):
    global stable_orbit_time, full_aerobraking
    if apoapsis >= r_a and abs(r - periapsis) <= 10 and m > mdry:
        return 1.0  # Throttle fully open to reach the desired orbit
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

def get_initial_spacecraft_state(b=None, v_inf=None, dist_saturn_titan=None, R_SOI_saturn=None):
    if b is None or b <= 0:
        b = b_i
    else:
        b = b
    #print(f"b: {b:.0f} km")

    # Rest of the function as before
    y_0 = -np.sqrt(R_SOI_saturn**2 - (b + dist_saturn_titan)**2)  # Initial y-coordinate relative to Saturn
    x_0 = b + dist_saturn_titan  # Initial x-coordinate relative to Saturn

    # Calculate the radial distance and angle in polar coordinates
    r_0 = np.sqrt(x_0**2 + y_0**2)  # Radial distance from Saturn
    phi_0 = np.arctan2(y_0, x_0)  # Angle between spacecraft and Saturn

    # Calculate initial velocities
    rho_r_0 = v_inf * np.cos(phi_0 - np.pi / 2)  # Radial velocity relative to Saturn
    rho_phi_0 = -v_inf * np.sin(phi_0 - np.pi / 2) / r_0  # Angular velocity relative to Saturn

    return x_0, y_0, r_0, phi_0, rho_r_0, rho_phi_0

def eom(t, state, thrust, ceff, mdry, r_thrust_descend, r_thrust_ascend, thrust_duration, r_p_slow):
    global is_descending, thrust_active, orbit_insertion_maneuver, full_aerobraking
    
    r, phi, rhor, rhophi, m = state

    # Get Titan's position based on the actual simulation time t (not affected by launch_time)
    x_titan, y_titan, v_titan = get_titan_position(t)  # Titan moves from t=0, no launch_time adjustment

    # Spacecraft's position, affected by launch_time for its delayed launch
    x_spacecraft = r * np.cos(phi)
    y_spacecraft = r * np.sin(phi)
    dx = x_spacecraft - x_titan
    dy = y_spacecraft - y_titan
    r_titan_spacecraft = np.sqrt(dx**2 + dy**2)

    # Calculate spacecraft velocity and flight path angle
    v = np.sqrt(rhor**2 + (r * rhophi)**2)  # Total velocity
    alpha = np.arctan2(r * rhophi, rhor)    # Flight path angle
    E = v**2 / 2 - mu_titan / r             # Orbital Energy (for Titan)
    h = r**2 * rhophi                       # Specific Angular Momentum
    a = -mu_titan / (2 * E)                 # Semi-major axis
    e = np.sqrt(1 + (2 * E * h**2) / (mu_titan**2))  # Eccentricity
    apoapsis = a * (1 + e)                  # Current apoapsis
    periapsis = a * (1 - e)                 # Current periapsis
    is_descending = rhor < 0                # Check if the spacecraft is descending
    throttle = 0.0                          # Default to no thrust
    atmo_density = atmospheric_density(r)  # Density in [kg/km^3]
    D = max(0.5 * c_d * atmo_density * A * v ** 2, 0) if atmo_density > 0 else 0  # Drag Force

    # Check if the spacecraft is inside Titan's SOI
    if r_titan_spacecraft <= R_SOI_titan:
        # Inside Titan's SOI: Gravitational force towards Titan
        accel_titan = -mu_titan / r_titan_spacecraft**2  # Magnitude of Titan's gravitational force

        # Vector components of the gravitational force towards Titan
        titan_dx = (x_spacecraft - x_titan) / r_titan_spacecraft  # X-component direction to Titan
        titan_dy = (y_spacecraft - y_titan) / r_titan_spacecraft  # Y-component direction to Titan

        # Gravitational acceleration in polar coordinates (radial and angular components)
        gravitational_acceleration_r_titan = accel_titan * (titan_dx * np.cos(phi) + titan_dy * np.sin(phi))  # Radial component
        gravitational_acceleration_phi_titan = accel_titan * (-titan_dx * np.sin(phi) + titan_dy * np.cos(phi))  # Angular component

        # Calculate Saturn's gravitational influence, even inside Titan's SOI
        x_saturn = 0  # Saturn is at the origin in the current setup
        y_saturn = 0  # Saturn is at the origin in the current setup
        r_saturn_spacecraft = np.sqrt(x_spacecraft**2 + y_spacecraft**2)  # Distance between spacecraft and Saturn

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
        x = r * np.cos(phi)  # Spacecraft x-position
        y = r * np.sin(phi)  # Spacecraft y-position

        # Compute the vector from the spacecraft to Saturn
        r_saturn = np.sqrt(x**2 + y**2)  # Distance between spacecraft and Saturn

        # Saturn's gravitational force magnitude
        accel_saturn = -mu_saturn / r_saturn**2

        # Normalize the vector pointing from the spacecraft to Saturn
        dx_norm = x / r_saturn
        dy_norm = y / r_saturn

        # Decompose Saturn's gravitational force into radial and angular components
        gravitational_acceleration_r = accel_saturn * (dx_norm * np.cos(phi) + dy_norm * np.sin(phi))  # Radial component
        gravitational_acceleration_phi = accel_saturn * (-dx_norm * np.sin(phi) + dy_norm * np.cos(phi))  # Angular component

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
        # Choose appropriate thrust logic based on the mission phase
        if orbit_insertion_maneuver:
            beta = alpha + np.pi
            throttle = handle_orbit_insertion_maneuver(r, m, mdry, t, r_thrust_descend, thrust_duration)
        elif not orbit_insertion_maneuver and full_aerobraking:
            beta = alpha
            throttle = handle_slowdown_aerobraking(r, m, mdry, apoapsis, periapsis, t, r_p_slow)
        elif not orbit_insertion_maneuver and not full_aerobraking:
            beta = alpha
            throttle = handle_orbit_stabilization(r, m, mdry, apoapsis, periapsis, t)

    # Update equations of motion including gravitational acceleration and thrust
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
    altitude = r - R_titan  # Altitude above the Titan surface
    return altitude - 20  # Trigger event when altitude is close to 0
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

def simulate_trajectory(b, r_thrust_descend, r_thrust_ascend, r_p_slow, thrust_duration, short_run=True, max_step_large=max_step_non_critical, max_step_small=max_step_critical, launch_time=0, critical_distance=critical_distance_titan_sim):
    global orbit_insertion_maneuver, is_descending, thrust_active, full_aerobraking
    global stable_orbit_time, slowdown_start_time, stabilization_start_time, thrust_start_time

    # Reset mission phase variables
    orbit_insertion_maneuver = True
    full_aerobraking = True
    is_descending = True
    thrust_active = False
    stable_orbit_time = None
    slowdown_start_time = None
    stabilization_start_time = None
    thrust_start_time = None
    inside_critical_zone = False

    # Get initial spacecraft state
    x_0, y_0, r_0, phi_0, rho_r_0, rho_phi_0 = get_initial_spacecraft_state(b, v_inf, dist_saturn_titan, R_SOI_saturn)
    init_val = [r_0, phi_0, rho_r_0, rho_phi_0, m_0]

    # Parameters for the spacecraft's thrust and orbital mechanics
    p = (thrust, ceff, m_0 - m_p, r_thrust_descend, r_thrust_ascend, thrust_duration, r_p_slow)

    # Define event functions with access to the inside_critical_zone flag
    def enter_critical_zone(t, state, *args):
        global thrust_active  # Ensure we can access the global thrust_active variable
        r, phi, _, _, _ = state
        x_spacecraft = r * np.cos(phi)
        y_spacecraft = r * np.sin(phi)
        x_titan, y_titan, _ = get_titan_position(t)
        dx = x_spacecraft - x_titan
        dy = y_spacecraft - y_titan
        r_titan_spacecraft = np.sqrt(dx**2 + dy**2)
        distance_to_titan_critical = r_titan_spacecraft - critical_distance
        distance_to_saturn_critical = r - critical_distance_saturn_sim

        if inside_critical_zone:
            # Already inside the critical zone; prevent event from triggering
            return 1  # Positive value; event will not trigger
        else:
            # Trigger event when entering critical zone
            if distance_to_titan_critical <= 0:
                # Entering critical zone
                return -1  # Negative value; event triggers
            elif distance_to_saturn_critical <=0 and orbit_insertion_maneuver:
                return -1
            else:
                return 1  # Positive value; no event

    enter_critical_zone.terminal = True
    enter_critical_zone.direction = -1  # Trigger when value crosses zero from positive to negative

    def exit_critical_zone(t, state, *args):
        global thrust_active  # Ensure we can access the global thrust_active variable
        r, phi, _, _, _ = state
        x_spacecraft = r * np.cos(phi)
        y_spacecraft = r * np.sin(phi)
        x_titan, y_titan, _ = get_titan_position(t)
        dx = x_spacecraft - x_titan
        dy = y_spacecraft - y_titan
        r_titan_spacecraft = np.sqrt(dx**2 + dy**2)
        distance_to_titan_critical = r_titan_spacecraft - critical_distance
        distance_to_saturn_critical = r - critical_distance_saturn_sim

        if not inside_critical_zone:
            # Not inside the critical zone; prevent event from triggering
            return -1  # Negative value; event will not trigger
        else:
            # Trigger event when exiting critical zone
            if distance_to_titan_critical > 0 and distance_to_saturn_critical > 0:
                # Exiting critical zone
                return 1  # Positive value; event triggers
            else:
                return -1  # Negative value; no event

    exit_critical_zone.terminal = True
    exit_critical_zone.direction = 1  # Trigger when value crosses zero from negative to positive


    # Initialize variables
    t_start = launch_time
    t_end = (tmax_short if short_run else tmax + 86400) + launch_time
    trajectory_segments = []
    max_step_current = max_step_large

    while t_start < t_end:
        t_span = (t_start, t_end)
        # Choose events based on the current state
        if not inside_critical_zone:
            event_list = [planet_crash, stable_orbit_reached, enter_critical_zone]
        else:
            event_list = [planet_crash, stable_orbit_reached, exit_critical_zone]

        # Perform integration
        sol = solve_ivp(eom, t_span, init_val, args=p, method='RK45', rtol=1e-8, atol=1e10,max_step=max_step_current, events=event_list)

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
            elif event_occurred in [planet_crash, stable_orbit_reached]:
                # Terminate integration
                break
        else:
            # No event occurred; end integration
            break

        # Update initial conditions for next segment
        init_val = sol.y[:, -1]
        t_start = sol.t[-1] + 1e-6  # Add a small time increment

    # Combine trajectory segments
    t_combined = np.hstack([seg.t for seg in trajectory_segments])
    y_combined = np.hstack([seg.y for seg in trajectory_segments])

    # Create combined trajectory object
    trajectory = Trajectory()
    trajectory.t = t_combined
    trajectory.y = y_combined
    trajectory.status = trajectory_segments[-1].status
    trajectory.t_events = [event for seg in trajectory_segments for event in seg.t_events]

    # Continue with your analysis as before
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
def calculate_launch_delay_iterative(initial_launch_time, b, r_p, radial_tolerance_km=500):
    global tolerance
    # Get the initial spacecraft state using the helper function
    x_0, y_0, r_0, phi_0, rho_r_0, rho_phi_0 = get_initial_spacecraft_state(b, v_inf, dist_saturn_titan, R_SOI_saturn)

    r_thrust_descend    = R_saturn + r_thrust_descending  # Total radius where thrust begins
    r_thrust_ascend     = R_saturn + r_thrust_ascending    # Total radius where thrust stops

    delay_time          = initial_launch_time  # Initialize delay time
    iteration           = 0  # Iteration counter
    max_iterations      =10

    while iteration < max_iterations:
        # Simulate trajectory with the current delay time
        _, _, trajectory, _, _ = simulate_trajectory(b, r_thrust_descend, r_thrust_ascend, r_p, thrust_duration, short_run=True, launch_time=delay_time, critical_distance=critical_distance_titan_sim)
        crossing_found = False # Find the orbit crossing point

        # Loop through trajectory to find orbit crossing
        for i in range(1, len(trajectory.y[0, :])):  # Start from index 1 to allow for interpolation
            # === Position Spacecraft/Titan ============================
            x_titan, y_titan, v_titan = get_titan_position(trajectory.t[i]) # Titan's position
            r_spacecraft = trajectory.y[0, i]                               # Spacecraft radial position
            phi_spacecraft = trajectory.y[1, i]                             # Spacecraft angular position
            x_spacecraft = r_spacecraft * np.cos(phi_spacecraft)            # Spacecraft's position in Cartesian coordinates
            y_spacecraft = r_spacecraft * np.sin(phi_spacecraft)
            # === Distance Calculations ============================
            distance_to_titan = np.sqrt((x_spacecraft - x_titan)**2 + (y_spacecraft - y_titan)**2) # Distance between the spacecraft and Titan
            distance_to_titan_surface = distance_to_titan - R_titan # Spacecraft altitude above Titan's surface
            r_titan_at_phi = np.sqrt(x_titan**2 + y_titan**2) # Titan's radial distance from Saturn

            # Check if the spacecraft is outside Titan and within radial tolerance
            if abs(r_spacecraft - r_titan_at_phi) <= radial_tolerance_km:
                crossing_found = True
                time_at_orbit = trajectory.t[i]
                phi_spacecraft_at_orbit = phi_spacecraft
                x_titan_at_orbit = x_titan
                y_titan_at_orbit = y_titan
                v_titan_at_orbit = v_titan
                distance_to_titan_surface_at_orbit = distance_to_titan_surface
                break

        if not crossing_found:
            print(f"  Warning: The spacecraft did not cross Titan's orbit during iteration {iteration}.")
            return delay_time  # Return current delay time if no crossing is found

        # Calculate Titan's angular position (phi) at the orbit crossing time
        phi_titan_at_orbit = np.arctan2(y_titan_at_orbit, x_titan_at_orbit)

        # Calculate the angular difference between Titan and the spacecraft (in radians)
        delta_phi = (phi_spacecraft_at_orbit - phi_titan_at_orbit + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-π, π]
        delta_phi_deg = np.degrees(delta_phi) # Convert delta_phi to degrees

        print(f"Iteration {iteration} for r_p = {r_p - R_titan} km:")

        # Check if spacecraft is ahead of Titan and at the correct distance
        if delta_phi_deg > 0 and abs(distance_to_titan_surface_at_orbit + R_titan - (r_p)) <= tolerance:
            print(f"  Spacecraft is {distance_to_titan_surface_at_orbit:.2f} km away from Titan at {time_at_orbit/86400:.4f} days.")
            return delay_time  # Stop iteration if spacecraft is ahead of Titan and at the desired distance

        # First, handle the rough adjustments for large angular differences
        if abs(delta_phi_deg) > 1:  # Rough adjustments for large delta_phi_deg
            print(f"  Angular difference between Spacecraft and Titan: {delta_phi_deg:.2f} degrees at orbit crossing. Making coarse adjustments.")
            r_current = np.sqrt(x_titan_at_orbit**2 + y_titan_at_orbit**2)
            omega_avg = v_titan_at_orbit / r_current  # Angular velocity of Titan

            delay_adjustment = delta_phi / omega_avg * 0.9757  # Adjust based on angular velocity
            delay_time += delay_adjustment  # Apply delay adjustment

        else:  # Fine adjustments for small angular differences
            distance_error = distance_to_titan_surface_at_orbit + R_titan - r_p
            if delta_phi_deg > 0:
                if abs(distance_error) >= 1000:
                    delay_adjustment = abs(distance_error) / 5 * 1.05
                elif 1000 > abs(distance_error) >= 400:
                    delay_adjustment = abs(distance_error) / 5 * 0.915
                elif 400 > abs(distance_error) > tolerance:
                    delay_adjustment = abs(distance_error) / 5 * 0.87
            else:
                if abs(distance_error) >= R_titan:
                    delay_adjustment = abs(distance_error) / 5 * 2
                else: 
                    delay_adjustment = abs(distance_error+R_titan) / 5 * 2.1
            # Adjust delay based on angular difference and distance to Titan
            if delta_phi_deg > 0:  # Spacecraft is ahead of Titan
                if distance_to_titan_surface_at_orbit + R_titan < r_p:
                    print(f"  Spacecraft is {abs(distance_error):.1f} km too close to Titans surface. Decreasing delay by {delay_adjustment/60:.2f} minutes.")
                    delay_time -= delay_adjustment
                else:
                    print(f"  Spacecraft is {abs(distance_error):.1f} km too far ahead of Titan. Increasing delay by {delay_adjustment/60:.2f} minutes.")
                    delay_time += delay_adjustment
            else:  # Spacecraft is behind Titan
                print(f"  Spacecraft is behind Titan ({distance_to_titan_surface_at_orbit:.1f} km) error: {distance_error:.1f}. Decreasing delay by {delay_adjustment/60:.2f} minutes.")
                delay_time -= delay_adjustment

        iteration += 1

    print(f"Reached maximum iterations. Final launch delay: {delay_time / 86400:.4f} days")
    return delay_time

def calculate_delay_time_for_r_p(r_p):
    delay_time = calculate_launch_delay_iterative(0, b_i, r_p)
    print(f"  Optimal launch delay for r_p = {r_p - R_titan:.0f} km: {delay_time / 86400:.4f} days")
    return r_p, delay_time

def adjust_launch_time():
    delay_times = {}  # Dictionary to store delay time for each r_p

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(calculate_delay_time_for_r_p, r_p) for r_p in r_p_values]
        for future in as_completed(futures):
            r_p, delay_time = future.result()
            delay_times[r_p] = delay_time  # Store the launch time for this r_p

    return delay_times
#endregion

#region Optimize Thrustduration
def simulate_for_r_p(r_p):
    global r_thrust_descend, r_thrust_ascend, best_fuel_usage, best_b, best_r_thrust_descend, best_r_thrust_ascend, tested_combinations, best_configs, r_p_slow, best_trajectory
    
    if r_p - R_titan <= 74:
        r_p_slow = r_p + 20
    elif r_p - R_titan <= 79:
        r_p_slow = r_p + 16
    elif r_p - R_titan <= 89:
        r_p_slow = r_p + 12
    elif r_p - R_titan <= 99:
        r_p_slow = r_p + 8
    else:
        r_p_slow = r_p + 4

    r_thrust_descend = R_saturn + r_thrust_descending  # Reset to initial value
    r_thrust_ascend = R_saturn + r_thrust_ascending    # Reset to initial value

    best_fuel_usage = float('inf')
    best_b = None
    best_r_thrust_descend = r_thrust_descend
    best_r_thrust_ascend = r_thrust_ascend
    tested_combinations = set()
    best_configs = []
    best_trajectory = None  # Initialize to store the best trajectory
    
    # Hardcode a value for b (e.g., initial_best_b) instead of calling find_optimal_b
    initial_best_b = b_i  # You can adjust this value as necessary
    best_b = initial_best_b  # Use the initial guess directly

    return {
        'r_p': r_p,
        'best_configs': best_configs,
        'r_p_slow': r_p_slow  # Include r_p_slow for the full trajectory simulation
    }

#endregion

#region Final Simulation of best configurations
def run_full_trajectory(config):
    global stable_orbit_time, aerobrake, slowdown_start_time, stabilization_start_time, max_step_critical, max_step_non_critical

    r_p = config['r_p']
    b = config['b']
    r_thrust_descend = config['r_thrust_descend']
    r_thrust_ascend = config['r_thrust_ascend']
    r_p_slow = config['r_p_slow']
    thrust_only = config['thrust_only']
    launch_time = config['launch_time']  # Use the launch time from config
    thrust_duration = config['thrust_duration']  # Get thrust_duration from config

    # Set aerobrake to false if it's a thrust_only run
    if thrust_only:
        original_aerobrake = aerobrake
        aerobrake = False

    final_sim_start_time = tm.time()

    # Call simulate_trajectory with the specified parameters and launch_time
    if r_p - R_titan <= 80:
        _, _, full_trajectory, _, final_mass = simulate_trajectory(
            b, r_thrust_descend, r_thrust_ascend, r_p_slow, thrust_duration=thrust_duration, short_run=False,
            max_step_large=max_step_non_critical, max_step_small=max_step_critical * 0.5, launch_time=launch_time, critical_distance=critical_distance_titan_sim)
    elif r_p - R_titan <= 90:
        _, _, full_trajectory, _, final_mass = simulate_trajectory(
            b, r_thrust_descend, r_thrust_ascend, r_p_slow, thrust_duration=thrust_duration, short_run=False,
            max_step_large=max_step_non_critical, max_step_small=max_step_critical * 0.5, launch_time=launch_time, critical_distance=critical_distance_titan_sim)
    elif r_p - R_titan <= 95:
        _, _, full_trajectory, _, final_mass = simulate_trajectory(
            b, r_thrust_descend, r_thrust_ascend, r_p_slow, thrust_duration=thrust_duration, short_run=False,
            max_step_large=max_step_non_critical, max_step_small=max_step_critical * 0.5, launch_time=launch_time, critical_distance=critical_distance_titan_sim)
    else:
        _, _, full_trajectory, _, final_mass = simulate_trajectory(
            b, r_thrust_descend, r_thrust_ascend, r_p_slow, thrust_duration=thrust_duration, short_run=False,
            max_step_large=max_step_non_critical, max_step_small=max_step_critical * 0.5, launch_time=launch_time, critical_distance=critical_distance_titan_sim)

    # Check initial position after simulation starts
    r = full_trajectory.y[0, :]
    phi = full_trajectory.y[1, :]
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    final_sim_end_time = tm.time()
    final_sim_duration = final_sim_end_time - final_sim_start_time

    if stable_orbit_time is not None:
        stable_orbit_days = stable_orbit_time / 86400
    else:
        stable_orbit_days = None

    fuel_used = m_0 - final_mass

    if stable_orbit_time is not None:
        print(f"TTSO: {stable_orbit_days:.1f} days | r_p: {r_p-R_titan:.1f} km | Thrust Duration: {thrust_duration/3600:.1f} h | Fuel: {fuel_used:.1f} kg | Computation time: {final_sim_duration/60:.2f} min")
    else:
        print(f"Stable orbit not reached for r_p: {r_p-R_titan:.1f} km | Thrust Duration: {thrust_duration/3600:.1f} h | Fuel used: {fuel_used:.1f} kg | Computation time: {final_sim_duration/60:.2f} min")

    # Reset aerobrake to its original value if it was modified
    if thrust_only:
        aerobrake = original_aerobrake

    return {
        'r_p': r_p,
        'b': b,
        'r_thrust_descend': r_thrust_descend,
        'r_thrust_ascend': r_thrust_ascend,
        'thrust_duration': thrust_duration,
        'fuel_used': fuel_used,
        'time_to_stable_orbit': stable_orbit_days,
        'full_trajectory': full_trajectory,
        'slowdown_start_time': slowdown_start_time,  # Store slowdown start time
        'stabilization_start_time': stabilization_start_time  # Store stabilization start time
    }

#endregion

#region Multithreading & Plots
    #region Multithread
if __name__ == "__main__":
    program_start_time  = tm.time()     # Track the start time of the entire program
    results             = []            # Store results for each r_p
    all_configs         = []            # Collect all valid configurations first

    # Adjust launch times for each r_p
    calculated_launch_times = adjust_launch_time()  # Returns a dict mapping r_p to launch_time

    # Create configs for each r_p
    for r_p in r_p_values:
        launch_time = calculated_launch_times[r_p]
        r_p_slow = r_p + 5  # Or compute based on r_p if necessary
        config = {
            'r_p': r_p,
            'b': b_i,  # Fixed b
            'r_thrust_descend': R_titan + r_thrust_descending,
            'r_thrust_ascend': R_titan + r_thrust_ascending,
            'thrust_duration': thrust_duration,  # Include thrust_duration
            'fuel_used': None,
            'time_to_stable_orbit': None,
            'full_trajectory': None,
            'r_p_slow': r_p_slow,
            'thrust_only': False,
            'launch_time': launch_time,  # Include the launch time
        }
        all_configs.append(config)

    # If Optimize_Thrust_Duration is True, create multiple configs with reduced thrust_duration
    if Optimize_Thrust_Duration:
        optimized_configs = []
        for config in all_configs:
            initial_thrust_duration = config['thrust_duration']
            thrust_duration = initial_thrust_duration
            while thrust_duration >= thrust_duration_limit:
                new_config = config.copy()
                new_config['thrust_duration'] = thrust_duration
                optimized_configs.append(new_config)
                thrust_duration -= thrust_duration_reduction
        all_configs = optimized_configs

    # Simulate full Aerobraking Trajectory until final orbit is reached
    full_trajectory_results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_full_trajectory, config)
            for config in all_configs
        ]
        for future in as_completed(futures):
            full_trajectory_results.append(future.result())

    # Organize and sort results by Periapsis height (r_p) and thrust_duration
    results_by_r_p = {}
    for result in full_trajectory_results:
        r_p = result['r_p']
        if r_p not in results_by_r_p:
            results_by_r_p[r_p] = []
        results_by_r_p[r_p].append(result)
    sorted_r_p_values = sorted(results_by_r_p.keys())

    # Track the end time of the entire program
    program_end_time = tm.time()
    program_duration = (program_end_time - program_start_time)/60

    # Print the total duration
    print(f"Total program duration: {program_duration:.2f} minutes")
    #endregion

    #region Plot Propellant consumed vs Time to reach stable Orbit
    if Plot_Comparison:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_r_p_values) + 1))

        for i, r_p in enumerate(sorted_r_p_values):
            results_for_r_p = results_by_r_p[r_p]
            fuel_used = [res['fuel_used'] for res in results_for_r_p]
            time_to_stable_orbit = [res['time_to_stable_orbit'] for res in results_for_r_p]
            labels = [f"{res['r_thrust_ascend'] - R_titan:.0f} km" for res in results_for_r_p]  # Subtract R_titan for the plot

            # Filter out None values
            valid_indices = [j for j, t in enumerate(time_to_stable_orbit) if t is not None]
            filtered_times = [time_to_stable_orbit[j] for j in valid_indices]
            filtered_fuel = [fuel_used[j] for j in valid_indices]
            filtered_labels = [labels[j] for j in valid_indices]

            if len(filtered_times) > 0:
                sorted_indices = np.argsort(filtered_times)
                sorted_times = np.array(filtered_times)[sorted_indices]
                sorted_fuel = np.array(filtered_fuel)[sorted_indices]
                sorted_labels = np.array(filtered_labels)[sorted_indices]

                plt.plot(sorted_times, sorted_fuel, linestyle='-', color=colors[i], alpha=0.6, label=f'r_p = {r_p - R_titan:.0f} km')
                plt.scatter(sorted_times, sorted_fuel, color=colors[i])

                # Add annotations for every second point with dynamic text positioning
                max_time = max(sorted_times)
                for j, (time, fuel, label) in enumerate(zip(sorted_times[::2], sorted_fuel[::2], sorted_labels[::2])):
                    x_offset = 10 * (time / max_time)  # Dynamic adjustment based on the position of the time
                    plt.annotate(label, (time, fuel), textcoords="offset points", xytext=(x_offset, 5), ha='right', fontsize=8, rotation=45)

        plt.xlabel('Time to Reach Stable Orbit (days)')
        plt.ylabel('Propellant Consumed (kg)')
        plt.title('Propellant Consumed vs Time to Reach Stable Orbit for Different r_p')
        plt.legend(title='Pericenter Heights', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
    #endregion

    #region Plot_Values
    if Plot_Values:
        # Organize results by r_p and thrust_duration
        from collections import defaultdict

        # Create a nested dictionary {r_p: {thrust_duration: [results]}}
        results_by_r_p_and_thrust = defaultdict(dict)
        for result in full_trajectory_results:
            r_p = result['r_p']
            thrust_duration = result['thrust_duration']
            if thrust_duration not in results_by_r_p_and_thrust[r_p]:
                results_by_r_p_and_thrust[r_p][thrust_duration] = []
            results_by_r_p_and_thrust[r_p][thrust_duration].append(result)

        sorted_r_p_values = sorted(results_by_r_p_and_thrust.keys())

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Position axes for the checkboxes
        config_check_ax = plt.axes([0.01, 0.5, 0.15, 0.4], frameon=False)
        thrust_check_ax = plt.axes([0.01, 0.1, 0.15, 0.35], frameon=False)
        parameter_check_ax = plt.axes([0.91, 0.3, 0.08, 0.6], frameon=False)
        parameter_labels = ['Orbital Energy', 'Distance to Saturn', 'Mass', 'Velocity', 'Heating Rate', 'Atmospheric Density']
        parameter_check = CheckButtons(parameter_check_ax, parameter_labels, [False] * len(parameter_labels))

        selected_params = set()
        current_config = None

        # Get unique r_p values and corresponding configs
        rp_labels = [f"r_p = {r_p - R_titan:.0f} km" for r_p in sorted_r_p_values]
        config_check = CheckButtons(config_check_ax, rp_labels, [False] * len(rp_labels))

        thrust_check = None  # Initialize thrust_check globally

        def plot_selected_configuration(r_p, thrust_duration):
            global current_config
            current_config = (r_p, thrust_duration)

            selected_results = results_by_r_p_and_thrust[r_p][thrust_duration]
            selected_result = selected_results[0]  # Assuming only one result per config

            full_trajectory = selected_result['full_trajectory']
            time = full_trajectory.t / 86400  # Convert time to days
            r = full_trajectory.y[0, :]
            phi = full_trajectory.y[1, :]
            rhor = full_trajectory.y[2, :]
            rhophi = full_trajectory.y[3, :]
            mass = full_trajectory.y[4, :]
            v = np.sqrt(rhor ** 2 + (r * rhophi) ** 2)  # Velocity in [km/s]
            E = v ** 2 / 2 - mu_titan / r  # Orbital Energy
            distance_to_saturn = np.sqrt((r * np.cos(phi))**2 + (r * np.sin(phi))**2) - R_saturn

            # Compute Titan's position at each time step in the trajectory
            titan_positions = [get_titan_position(t * 86400)[:2] for t in time]  # Titan position (x, y) in km

            # Calculate atmospheric density and heating rate based on altitude above Titan
            atmo_density = np.zeros_like(r)
            Q_dot = np.zeros_like(r)
            altitude = np.zeros_like(r)
            for i, rad in enumerate(r):
                x_spacecraft = rad * np.cos(phi[i])  # Spacecraft x-position in Titan-centered frame
                y_spacecraft = rad * np.sin(phi[i])  # Spacecraft y-position in Titan-centered frame

                # Calculate the distance from the spacecraft to Titan at this time step
                titan_x, titan_y = titan_positions[i]
                r_titan_spacecraft = np.sqrt((x_spacecraft - titan_x)**2 + (y_spacecraft - titan_y)**2)
                altitude[i] = r_titan_spacecraft - R_titan  # Altitude above Titan’s surface

                # Only compute atmo_density and Q_dot if within atmosphere
                if altitude[i] <= atmo_height:
                    atmo_density[i] = atmospheric_density(altitude[i] + R_titan)
                    Q_dot[i] = c_q * np.sqrt(atmo_density[i]) * v[i] ** 3
                else:
                    atmo_density[i] = 0
                    Q_dot[i] = 0

            # Subsampled versions for efficient plotting
            subsample_factor = 50
            subsampled_indices = np.arange(0, len(time), subsample_factor)
            time_sub = time[subsampled_indices]
            altitude_sub = altitude[subsampled_indices]
            distance_to_saturn_sub = distance_to_saturn[subsampled_indices]  # Distance to Saturn's surface
            v_sub = v[subsampled_indices]
            E_sub = E[subsampled_indices]
            mass_sub = mass[subsampled_indices]
            Q_dot_sub = Q_dot[subsampled_indices]
            atmo_density_sub = atmo_density[subsampled_indices]  # Subsampled atmospheric density

            def shade_atmosphere_regions(ax, time_sub, altitude_sub, atmo_height):
                in_atmosphere = altitude_sub <= atmo_height
                regions = []
                start = None

                for i in range(len(time_sub)):
                    if in_atmosphere[i] and start is None:
                        start = time_sub[i]
                    elif not in_atmosphere[i] and start is not None:
                        end = time_sub[i]
                        regions.append((start, end))
                        start = None

                if start is not None:
                    regions.append((start, time_sub[-1]))

                for start, end in regions:
                    ax.axvspan(start, end, color='salmon', alpha=0.4)

            def shade_thrust_active(ax, time_sub, mass_sub):
                thrust_active_regions = []
                thrust_active = False
                for i in range(1, len(mass_sub)):
                    if mass_sub[i] < mass_sub[i - 1]:
                        if not thrust_active:
                            thrust_start = time_sub[i - 1]
                            thrust_active = True
                    elif thrust_active:
                        thrust_end = time_sub[i - 1]
                        thrust_active_regions.append((thrust_start, thrust_end))
                        thrust_active = False
                if thrust_active:
                    thrust_end = time_sub[-1]
                    thrust_active_regions.append((thrust_start, thrust_end))

                for start, end in thrust_active_regions:
                    ax.axvspan(start, end, color='lightskyblue', alpha=0.8)

            def update_plot(label):
                global slowdown_start_time, stabilization_start_time
                if label in selected_params:
                    selected_params.remove(label)
                else:
                    selected_params.add(label)

                if current_config is None:
                    return

                ax.clear()

                if 'Orbital Energy' in selected_params:
                    ax.plot(time_sub, E_sub, label='Orbital Energy')
                    ax.set_ylabel('Energy (MJ/kg)')
                    ax.set_title('Orbital Energy vs Time')
                    shade_thrust_active(ax, time_sub, mass_sub)
                    shade_atmosphere_regions(ax, time_sub, altitude_sub, atmo_height)

                if 'Distance to Saturn' in selected_params:
                    ax.plot(time_sub, distance_to_saturn_sub, color="purple", label='Distance to Saturn')
                    ax.set_ylabel('Distance to Saturn (km)')
                    ax.set_title('Distance to Saturn vs Time')
                    ax.set_yscale('log')
                    ax.axhline(y=R_saturn, color='gray', linestyle='--', label='Saturn Surface')
                    shade_thrust_active(ax, time_sub, mass_sub)
                    shade_atmosphere_regions(ax, time_sub, altitude_sub, atmo_height)

                if 'Mass' in selected_params:
                    ax.plot(time_sub, mass_sub, color="blue", label='Mass')
                    ax.set_ylabel('Mass (kg)')
                    ax.set_title('Mass vs Time')
                    shade_thrust_active(ax, time_sub, mass_sub)
                    shade_atmosphere_regions(ax, time_sub, altitude_sub, atmo_height)

                if 'Velocity' in selected_params:
                    ax.plot(time_sub, v_sub, color="green", label='Velocity')
                    ax.set_ylabel('Velocity (km/s)')
                    ax.set_title('Velocity vs Time')
                    shade_thrust_active(ax, time_sub, mass_sub)
                    shade_atmosphere_regions(ax, time_sub, altitude_sub, atmo_height)

                if 'Heating Rate' in selected_params:
                    ax.plot(time_sub, Q_dot_sub, color="red", label='Heating Rate')
                    ax.set_ylabel('Heating Rate (W/m^2)')
                    ax.set_title('Heating Rate vs Time')
                    ax.set_yscale('log')
                    ax.yaxis.set_major_formatter(ScalarFormatter())
                    shade_thrust_active(ax, time_sub, mass_sub)
                    shade_atmosphere_regions(ax, time_sub, altitude_sub, atmo_height)

                if 'Atmospheric Density' in selected_params:
                    ax.plot(time_sub, atmo_density_sub, color="blue", label='Atmospheric Density')
                    ax.set_ylabel('Atmospheric Density (kg/km³)')
                    ax.set_yscale('log')
                    ax.set_title('Atmospheric Density vs Time')
                    shade_thrust_active(ax, time_sub, mass_sub)
                    shade_atmosphere_regions(ax, time_sub, altitude_sub, atmo_height)

                ax.set_xlabel('Time (days)')
                ax.legend(loc='upper right')
                ax.grid(True)
                plt.draw()

            parameter_check.on_clicked(update_plot)

        def update_thrust_selection(label):
            global thrust_check
            if current_rp_index is None:
                return
            thrust_duration_hours = float(label.split(' = ')[1].split(' ')[0])
            thrust_duration = thrust_duration_hours * 3600  # Convert hours back to seconds

            r_p = sorted_r_p_values[current_rp_index]
            plot_selected_configuration(r_p, thrust_duration)


        def update_config_selection(label):
            global thrust_check, current_rp_index
            current_rp_index = rp_labels.index(label)
            r_p = sorted_r_p_values[current_rp_index]
            thrust_durations = sorted(results_by_r_p_and_thrust[r_p].keys())

            # Remove previous thrust checkboxes and recreate them with new labels
            thrust_check_ax.clear()
            thrust_labels = [f"Duration = {td / 3600:.1f} h" for td in thrust_durations]
            thrust_check = CheckButtons(thrust_check_ax, thrust_labels, [False] * len(thrust_labels))
            thrust_check.on_clicked(update_thrust_selection)

        current_rp_index = None
        config_check.on_clicked(update_config_selection)
        plt.show()
    #endregion

    #region Plot Atmosphere
    if Plot_Atmosphere:
        heights = np.linspace(0, atmo_height, 500)
        densities = [atmospheric_density(height + R_titan) for height in heights]

        plt.figure(figsize=(10, 6))
        plt.plot(densities, heights)
        plt.xscale('log')
        plt.ylabel('Height (km)')
        plt.xlabel('Atmospheric Density (kg/km^3)')
        plt.title('Atmospheric Density vs Height')
        plt.grid(True)
    #endregion

    #region Plot_Trajectory
    if Plot_Trajectory:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import CheckButtons, Slider

        plt.rcParams["figure.figsize"] = (8, 8)

        # Create the figure and main axes
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.25, bottom=0.15)  # Adjust to make space for the slider and checkboxes

        # Ensure we have some valid trajectory data
        if full_trajectory_results:
            # Organize trajectory data by r_p and thrust_duration
            trajectory_data = {}
            for result in full_trajectory_results:
                r_p = result['r_p']
                thrust_duration = result['thrust_duration']
                key = (r_p, thrust_duration)
                trajectory_data[key] = result['full_trajectory']

            # Create axes for the checkboxes
            rp_checkbox_ax = plt.axes([0.01, 0.5, 0.15, 0.4], frameon=False)
            thrust_checkbox_ax = plt.axes([0.01, 0.1, 0.15, 0.35], frameon=False)

            # Create labels for the rp checkboxes
            rp_values = sorted(set(rp for rp, _ in trajectory_data.keys()))
            rp_labels = [f"r_p = {rp - R_titan:.0f} km" for rp in rp_values]
            rp_checkbox = CheckButtons(rp_checkbox_ax, rp_labels, [False]*len(rp_labels))

            # Initialize variables
            trajectory_lines = {}
            selected_rp = None
            thrust_labels = []  # Declare thrust_labels as a global variable
            thrust_checkbox = None  # Declare thrust_checkbox as a global variable

            # Saturn plot at the center
            saturn = plt.Circle((0, 0), R_saturn / 1000, color='yellow', fill=True, label='Saturn')  # Saturn is at the origin
            ax.add_patch(saturn)
            Sphere_of_Influence_Saturn = plt.Circle((0, 0), R_SOI_saturn / 1000, color='black', fill=False, alpha=0.6, linestyle='--', label='Saturn SOI')
            ax.add_patch(Sphere_of_Influence_Saturn)

            # Calculate Titan's elliptical orbit using get_titan_position
            num_points = 1000  # Number of points to plot for the orbit
            titan_orbit_x = []
            titan_orbit_y = []
            time_for_orbit = np.linspace(0, T_titan, num_points)  # Time range over one orbital period of Titan

            for t in time_for_orbit:
                x_titan, y_titan, _ = get_titan_position(t)
                titan_orbit_x.append(x_titan / 1000)  # Convert to 1000 km for plotting
                titan_orbit_y.append(y_titan / 1000)

            # Plot Titan's elliptical orbit
            ax.plot(titan_orbit_x, titan_orbit_y, color='salmon', linestyle='--', label='Titan Orbit')

            # Initial plot of spacecraft (updated by the slider)
            spacecraft_plot, = ax.plot([], [], 'bo', markersize=8, label='Spacecraft Position')

            # Initialize Titan as a Circle
            titan_circle = plt.Circle((0, 0), R_titan / 1000, color='red', fill=True, label='Titan')
            ax.add_patch(titan_circle)

            # Titan atmosphere and SOI boundaries
            titan_atmosphere = plt.Circle((0, 0), (R_titan + atmo_height) / 1000, color='salmon', fill=True, alpha=0.6, label='Atmosphere Boundary')
            titan_soi = plt.Circle((0, 0), R_SOI_titan / 1000, color='grey', fill=False, alpha=0.6, linestyle='--', label='Titan SOI')
            ax.add_patch(titan_atmosphere)
            ax.add_patch(titan_soi)

            # Set fixed plot limits and aspect ratio to prevent stretching
            plot_radius = R_SOI_saturn / 1000  # Define the range based on the Sphere of Influence
            ax.set_xlim(-plot_radius, plot_radius)
            ax.set_ylim(-plot_radius, plot_radius)
            ax.set_aspect('equal', 'box')  # Fix the aspect ratio to keep shapes consistent

            # Set axis labels
            ax.set_xlabel('x [$10^3$ km]')
            ax.set_ylabel('y [$10^3$ km]')

            # Add a legend
            ax.legend(loc='upper right')

            # Create a smaller slider for controlling the time in days
            max_time = max([traj.t[-1]/86400 for traj in trajectory_data.values()])
            time_step = 0.001  # Adjust the time step as needed
            ax_slider = plt.axes([0.25, 0.05, 0.65, 0.02], facecolor='lightgoldenrodyellow')  # Slider position
            time_slider = Slider(ax_slider, 'Time (days)', 0, max_time, valinit=0, valstep=time_step, valfmt="%.3f")

            # Function to update the spacecraft and Titan positions
            def update_plot(val):
                current_time = time_slider.val  # This is in days

                # Titan position (moves from t=0)
                x_titan, y_titan, _ = get_titan_position(current_time * 86400)
                x_titan /= 1000
                y_titan /= 1000
                titan_circle.center = (x_titan, y_titan)
                titan_atmosphere.center = (x_titan, y_titan)
                titan_soi.center = (x_titan, y_titan)

                # Update spacecraft position if only one trajectory is visible
                visible_lines = [line for line in trajectory_lines.values() if line.get_visible()]
                if len(visible_lines) == 1:
                    line = visible_lines[0]
                    key = [k for k, v in trajectory_lines.items() if v == line][0]
                    full_trajectory = trajectory_data[key]

                    traj_time = full_trajectory.t / 86400  # Convert to days
                    index = np.searchsorted(traj_time, current_time)
                    if index >= len(traj_time):
                        index = -1  # Use the last point
                    r = full_trajectory.y[0, index]
                    phi = full_trajectory.y[1, index]
                    x_spacecraft = r * np.cos(phi) / 1000  # Convert to 1000 km
                    y_spacecraft = r * np.sin(phi) / 1000
                    # Update spacecraft position
                    spacecraft_plot.set_data([x_spacecraft], [y_spacecraft])
                else:
                    # No trajectory selected or multiple trajectories selected, hide spacecraft
                    spacecraft_plot.set_data([], [])

                fig.canvas.draw_idle()  # Redraw the canvas

            # Connect the slider to the update function
            time_slider.on_changed(update_plot)
            time_slider.valtext.set_text(f'{time_slider.val:.3f}')

            # Function to handle keypress events and move the slider with keys
            def on_key(event):
                if event.key == 'd':  # d key to increase slider value
                    current_val = time_slider.val
                    time_slider.set_val(min(time_slider.valmax, current_val + time_slider.valstep*0.1))
                elif event.key == 'a':  # a key to decrease slider value
                    current_val = time_slider.val
                    time_slider.set_val(max(time_slider.valmin, current_val - time_slider.valstep*0.1))
                elif event.key == 'w':  # w key to increase slider value a lot
                    current_val = time_slider.val
                    time_slider.set_val(min(time_slider.valmax, current_val + time_slider.valstep))
                elif event.key == 'x':  # x key to decrease slider value a lot
                    current_val = time_slider.val
                    time_slider.set_val(max(time_slider.valmin, current_val - time_slider.valstep))

            # Connect the keypress event to the on_key function
            fig.canvas.mpl_connect('key_press_event', on_key)

            # Function to update thrust checkboxes based on selected r_p
            def update_thrust_checkboxes(label):
                global selected_rp, thrust_checkbox, thrust_labels
                selected_rp = rp_values[rp_labels.index(label)]

                # Get thrust durations for the selected r_p
                thrust_durations = sorted(td for rp, td in trajectory_data.keys() if rp == selected_rp)
                thrust_labels = [f"Duration = {td / 3600:.1f} h" for td in thrust_durations]

                # Remove previous thrust checkboxes and recreate them with new labels
                thrust_checkbox_ax.clear()
                thrust_checkbox_ax.set_position([0.01, 0.1, 0.15, 0.35])  # Ensure the position remains consistent
                thrust_checkbox = CheckButtons(thrust_checkbox_ax, thrust_labels, [False]*len(thrust_labels))
                thrust_checkbox.on_clicked(update_trajectory_visibility)
                plt.draw()

            # Function to update trajectory visibility based on selected thrust durations
            def update_trajectory_visibility(label):
                global selected_rp, thrust_labels
                if selected_rp is None:
                    return

                idx = thrust_labels.index(label)
                thrust_duration_hours = float(label.split(' = ')[1].split(' ')[0])
                thrust_duration = thrust_duration_hours * 3600  # Convert hours back to seconds
                key = (selected_rp, thrust_duration)
                if key not in trajectory_lines:
                    # Create trajectory line if it doesn't exist
                    full_trajectory = trajectory_data[key]

                    r = full_trajectory.y[0, :]
                    phi = full_trajectory.y[1, :]
                    time = full_trajectory.t / 86400  # Convert time to days for the slider

                    # Sample the data for plotting
                    subsample_factor = 10  # Adjust this factor to control the amount of data plotted
                    subsampled_indices = np.arange(0, len(r), subsample_factor)

                    # Adjust the positions to make Saturn the central body
                    x = (r * np.cos(phi))[subsampled_indices]
                    y = (r * np.sin(phi))[subsampled_indices]

                    # Plot the trajectory, but set visibility based on checkbox
                    visible = thrust_checkbox.get_status()[idx]
                    line, = ax.plot(x / 1000, y / 1000, linewidth=0.4, label=f"{rp_labels[rp_values.index(selected_rp)]}, {label}", visible=visible)
                    trajectory_lines[key] = line
                else:
                    # Update visibility
                    line = trajectory_lines[key]
                    line.set_visible(not line.get_visible())

                # Update the legend
                handles = [saturn, titan_circle, Sphere_of_Influence_Saturn, titan_soi, titan_atmosphere, spacecraft_plot]
                labels = ['Saturn', 'Titan', 'Saturn SOI', 'Titan SOI', 'Atmosphere Boundary', 'Spacecraft Position']
                for line in trajectory_lines.values():
                    if line.get_visible():
                        handles.append(line)
                        labels.append(line.get_label())
                ax.legend(handles=handles, labels=labels, loc='upper right')
                plt.draw()

            # Connect the rp checkbox event
            rp_checkbox.on_clicked(update_thrust_checkboxes)

            # Display the plot
            plt.show()

        else:
            print("No valid trajectory data available.")
    #endregion

    #region Plot Heating Rate
    if Plot_HeatingRate:
        # Compute data for heating rate plot
        def compute_heating_rate_data(results):
            global atmospheric_phases, color_map
            atmospheric_phases = []  # Reset atmospheric phases list
            color_map = {}  # Store mapping of (r_thrust, r_p) to color

            # Generate a color palette for configurations
            unique_configs = sorted({(result['r_thrust_ascend'], result['r_p']) for result in results})
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_configs)))
            config_color_dict = {config: color for config, color in zip(unique_configs, colors)}

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
                duration_above_threshold = 0.0
                total_duration = 0.0

                for i, in_atmo in enumerate(in_atmosphere):
                    if in_atmo:
                        if start is None:
                            start = time[i]
                        Q_dot_max = max(Q_dot_max, Q_dot[i])
                        Q_dot_sum += Q_dot[i]
                    elif start is not None:
                        end = time[i]
                        duration = end - start
                        total_duration = duration

                        # Calculate average heating rate over the phase
                        Q_dot_avg = Q_dot_sum / (i - np.argmax(in_atmosphere))

                        # Calculate peak duration (above 90% of Q_dot_max)
                        for j in range(np.argmax(in_atmosphere), i):
                            if Q_dot[j] >= 0.9 * Q_dot_max:
                                duration_above_threshold += time[j + 1] - time[j]

                        config = (result['r_thrust_ascend'], result['r_p'])
                        color = config_color_dict[config]
                        atmospheric_phases.append((Q_dot_max, Q_dot_avg, duration, duration_above_threshold, color, config))
                        start = None
                        Q_dot_max = 0.0
                        Q_dot_sum = 0.0
                        duration_above_threshold = 0.0

                if start is not None:
                    end = time[-1]
                    duration = end - start
                    total_duration = duration
                    Q_dot_avg = Q_dot_sum / (i + 1 - np.argmax(in_atmosphere))

                    for j in range(np.argmax(in_atmosphere), len(time) - 1):
                        if Q_dot[j] >= 0.9 * Q_dot_max:
                            duration_above_threshold += time[j + 1] - time[j]

                    config = (result['r_thrust_ascend'], result['r_p'])
                    color = config_color_dict[config]
                    atmospheric_phases.append((Q_dot_max, Q_dot_avg, duration, duration_above_threshold, color, config))

            color_map = config_color_dict

        # Generate the data for heating rate plot
        compute_heating_rate_data(full_trajectory_results)

        # Plot the heating rate vs duration if data is available
        def plot_heating_rate_vs_duration():
            plt.figure(figsize=(7, 6))
            plt.subplots_adjust(right=0.8)

            # Extract the data for plotting
            Q_dot_values = np.array([phase[0] for phase in atmospheric_phases])
            Q_dot_avg_values = np.array([phase[1] for phase in atmospheric_phases])
            durations = np.array([phase[2] * 1440 for phase in atmospheric_phases])  # Convert durations from days to minutes
            durations_above = np.array([phase[3] * 1440 for phase in atmospheric_phases])  # Peak durations
            colors = [phase[4] for phase in atmospheric_phases]

            # Plotting the data points
            plt.scatter(durations, Q_dot_values, c=colors, marker='x', label='Max Heating Rate')
            plt.scatter(durations_above, Q_dot_avg_values, c=colors, marker='o', label='Average Heating Rate')

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
            Q_dot_ablative = Q_total_ablative / (duration_range*60) 
            plt.plot(duration_range, Q_dot_ablative, color='green', linestyle='-', label='Ablative Limit (PICA)')

            Q_total_heat_sink = 36.45e6  # J/m²
            Q_dot_heat_sink = Q_total_heat_sink / (duration_range*60)
            plt.plot(duration_range, Q_dot_heat_sink, color='orange', linestyle='-', label='Heat Sink Limit (Aluminum)')

            # Combine all x-values and y-values for setting axis limits
            all_x_values = np.concatenate([durations, durations_above, duration_range])
            all_y_values = np.concatenate([Q_dot_values, Q_dot_avg_values, Q_dot_ablative, Q_dot_heat_sink])
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
            from matplotlib.lines import Line2D
            color_legend = [
                Line2D([0], [0], marker='o', color=color, lw=0, label=f'r_p = {config[1] - R_titan:.0f} km')
                for config, color in color_map.items()
            ]

            # Combining legends
            plt.legend(handles=color_legend + [
                Line2D([0], [0], color='black', linestyle='--', linewidth=1, label='Space Shuttle Radiation Limit'),
                Line2D([0], [0], color='green', linestyle='-', linewidth=1, label='Ablative Limit (PICA)'),
                Line2D([0], [0], color='orange', linestyle='-', linewidth=1, label='Heat Sink Limit (Aluminum)')
            ], loc='center left', bbox_to_anchor=(1, 0.5))

            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.show()

        plot_heating_rate_vs_duration()
    #endregion

    plt.show()
#endregion
