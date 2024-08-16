import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from concurrent.futures import ProcessPoolExecutor, as_completed
import time as tm

# === Begin user input  ============================
m_0                         = 3725.0            # Initial spacecraft mass in [kg]
m_p                         = 3000.0            # Propellant mass in [kg]
thrust                      = 425.0             # Thrust in [N]
c_3                         = 13.78             # Characteristic energy in [km^2/s^2]
Isp                         = 321.0             # Specific impulse
c_d                         = 2.2               # Drag coefficient
A                           = 29.3              # Cross-sectional area of the spacecraft in [m^2]555555555555555555555555555555555555555555555
tint                        = 175               # Integration time in days
r_p_lowest                  = 75                # Lowest Periapsis tested
r_p_highest                 = 96                # Highest Periapsis tested  
r_p_step_size               = 5                 # Step size between r_p_lowest and r_p_highest
r_p_orbit                   = 400               # Periapsis of desired Orbit in [km]
r_a                         = 800               # Apoapsis of desired Orbit in [km]
r_a_limit_aero              = 150000            # Apoapsis limit to avoid excessive orbit duration for aerobraking runs in [km]
r_a_limit_thrust            = 300000            # Apoapsis limit to avoid excessive orbit duration for thrust_only run in [km]
Plot_Trajectory             = False             # Plot spacecraft trajectory?
Plot_Atmosphere             = False             # Plot atmospheric density over height?
Plot_Values                 = True              # Plot panel of values?
Orbit_Count                 = False             # Plot over orbit count if True, otherwise over time
Optimize_Thrust_Range       = True              # Optimize R_Thrust_Ascend?
Thrust_Only_Run             = True              # Calculate Thrust Only Run?
Plot_Comparison             = True              # Plot propellant consumed vs time to stable orbit?
# === End user input    ============================

#region Prepare the program
# General constants for the problem
mu              = 42828.37              # Gravitational parameter of Mars in [km^3/s^2]
R               = 3396.2                # Mars Equatorial radius in [km]
R_SOI           = 577269                # Mars Radius of standard Sphere of Influence in [km](Improvement later?)
g0_mars         = 3.72076               # Mars standard gravitational acceleration in [m/s^2]
g0_earth        = 9.80665               # Earth standard gravitational acceleration in [m/s^2]
v_inf           = np.sqrt(c_3)          # Hyperbolic excess speed at infinity in [km/s]

# Start values for later Iteration
tint_short          = 8.0                                   # Integration time for short runs
r_thrust_descending = 23500                                 # Altitude where thrust begins for Orbit Insertion Maneuver
r_thrust_ascending  = 17000                                 # Altitude where thrust stops for Orbit Insertion Maneuver
r_thrust_reduction  = 500                                   # Reduction of r_thrust_ascending in [km]
b_i                 = R*np.sqrt(2*mu/(R*v_inf**2)+1)+4500   # Initial guess for b based on impact radius + experimental value
r_p_values          = np.arange(R + r_p_lowest, R + r_p_highest, r_p_step_size)

# Convert input into "correct" units
tmax                = tint * 86400              # Convert Integration time in [s]
tmax_short          = tint_short * 86400        # Convert Integration time in [s]
A                   = A / 1e6                   # Convert Cross-Sectional area in [km^2]
thrust              = thrust / 1000.0           # Convert thrust to [kg*km/s^2]
ceff                = Isp * g0_earth / 1000.0   # Effective exhaust velocity in [km/s]
r_p_orbit           = R + r_p_orbit             # Periapsis of desired Orbit including Planetradius 
r_a                 = R + r_a                   # Apoapsis of desired Orbit including Planetradius 
r_a_limit_aero      = R + r_a_limit_aero        # Apoapsis limit to avoid excessive orbit duration including Planetradius for aerobraking runs
r_a_limit_thrust    = R + r_a_limit_thrust      # Apoapsis limit to avoid excessive orbit duration including Planetradius for thrust_only run

# Initialize mission phase variables
orbit_insertion_maneuver    = True      # Set to false after the first Thruster burn
full_aerobraking            = True      # Set to false when r_p is raised to r_p_slow to prevent apoapsis from dropping lower than r_a
is_descending               = True      # Track if spacecraft is ascending or descending
thrust_active               = False     # Track when thrust is active for plots
stable_orbit_time           = None      # Track the time when stable orbit is reached
slowdown_start_time         = None
stabilization_start_time    = None
aerobrake                   = True      # Use aerobraking for the main simulations
#endregion

#region Atmosphere
atmo_height     = 200                                   # Height of the atmosphere boundary in [km]
def atmospheric_density(r):
    height          = r - R                             # Altitude above Mars' surface in km
    atmo_density    = 0.06 * 1e9                        # Mars Atmosphere density in [kg/m^3]
    atmo_density_200= 0.095 * 1e-3                      # Density at 195 km
    H = 195 / np.log(atmo_density / atmo_density_200)   # Scale height based on the new model
    return atmo_density * np.exp(-height / H) if height <= atmo_height else 0
#endregion

#region Mission Phases
def handle_orbit_insertion_maneuver(r, phi, rhor, rhophi, m, mdry, t, r_thrust_descend, r_thrust_ascend):
    global is_descending, thrust_active, orbit_insertion_maneuver
    if is_descending and r < r_thrust_descend and m > mdry:
        return 1.0  # Throttle fully open to capture the Spacecraft
    elif not is_descending and r > r_thrust_ascend:
        orbit_insertion_maneuver = False  # End of Orbit Insertion Maneuver
        return 0.0  # Throttle fully closed
    return 1.0 if thrust_active else 0.0  # Maintain current state

def handle_slowdown_aerobraking(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t, r_p_slow):
    global thrust_active, full_aerobraking, slowdown_start_time

    if r_p_slow - R <= 90:
        dynamic_range = 2700            # r_p = 70 (1. Mal periapsis kleiner als 3500) 
    elif 80 < r_p_slow - R <= 91:
        dynamic_range = 1500            # r_p = 75 (1. Mal periapsis kleiner als 2300) 
    elif 86 < r_p_slow - R <= 97:       
        dynamic_range = 1000            # r_p = 80, 85 (1. Mal periapsis kleiner als 1800)
    elif 93 < r_p_slow - R <= 103:       
        dynamic_range = 400             # r_p = 90, 95 (1. Mal periapsis kleiner als 1200)
    else:                       
        dynamic_range = 200             # r_p = 100 (1. Mal periapsis kleiner als 1000)

    if r_a < apoapsis <= (r_a + dynamic_range) and abs(r - apoapsis) < 30 and periapsis < r_p_slow and m > mdry:
        if slowdown_start_time is None:
            slowdown_start_time = t 
        return 1.0  # Throttle fully open
    elif periapsis >= r_p_slow + 3:
        return 0.0  # Throttle fully closed
    elif apoapsis <= r_a and periapsis > r_p_slow:
        full_aerobraking = False
    return 1.0 if thrust_active else 0.0  # Maintain current state

def handle_orbit_stabilization(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t):
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

def handle_thrust_only(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t):
    global stable_orbit_time, full_aerobraking
    if apoapsis >= r_a and abs(r - periapsis) <= 10 and m > mdry:
        return 1.0  # Throttle fully open to reach the desired orbit
    elif apoapsis < r_a and abs(r - periapsis) <= 0.1:
        full_aerobraking = False
    return 0.0
#endregion

#region Equations of Motion
def eom(t, state, thrust, ceff, mdry, r_thrust_descend, r_thrust_ascend, r_p_slow):
    global is_descending, thrust_active, orbit_insertion_maneuver, full_aerobraking
    
    r, phi, rhor, rhophi, m = state

    v = np.sqrt(rhor**2 + (r * rhophi)**2)  # Total velocity
    alpha = np.arctan2(r * rhophi, rhor)  # Flight path angle
    E = v**2 / 2 - mu / r  # Orbital Energy
    h = r**2 * rhophi  # Specific Angular Momentum (Angular momentum per unit mass)
    a = -mu / (2 * E)  # Semi-major axis
    e = np.sqrt(1 + (2 * E * h**2) / (mu**2))  # Eccentricity
    apoapsis = a * (1 + e)  # Current apoapsis
    periapsis = a * (1 - e)  # Current periapsis
    is_descending = rhor < 0  # Check if the spacecraft is descending
    throttle = 0.0  # Default to no thrust
    atmo_density = atmospheric_density(r)
    D = (0.5 * c_d * atmo_density * A * v ** 2) if atmo_density > 0 else 0 # Atmospheric Drag Force in [kg/s^2]

    if not aerobrake:
        if orbit_insertion_maneuver:
            beta = alpha + np.pi
            throttle = handle_orbit_insertion_maneuver(r, phi, rhor, rhophi, m, mdry, t, r_thrust_descend, r_thrust_ascend)
        elif not orbit_insertion_maneuver and full_aerobraking:
            beta = alpha + np.pi
            throttle = handle_thrust_only(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t)
        elif not orbit_insertion_maneuver and not full_aerobraking:
            beta = alpha
            throttle = handle_orbit_stabilization(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t)
    else:
        # Choose appropriate thrust logic based on the mission phase
        if orbit_insertion_maneuver:
            beta = alpha + np.pi
            throttle = handle_orbit_insertion_maneuver(r, phi, rhor, rhophi, m, mdry, t, r_thrust_descend, r_thrust_ascend)
        elif not orbit_insertion_maneuver and full_aerobraking:
            beta = alpha
            throttle = handle_slowdown_aerobraking(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t, r_p_slow)
        elif not orbit_insertion_maneuver and not full_aerobraking:
            beta = alpha
            throttle = handle_orbit_stabilization(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t)

    dr = rhor
    dphi = rhophi
    drhor = r * rhophi**2 - mu / r**2 - D / m * np.cos(alpha) + throttle * thrust / m * np.cos(beta)
    drhophi = (-2 * rhor * rhophi - D / m * np.sin(alpha) + throttle * thrust / m * np.sin(beta)) / r
    dm = - throttle * thrust / ceff

    thrust_active = throttle > 0.0

    return [dr, dphi, drhor, drhophi, dm]

#endregion

#region Simulation termination conditions
def planet_crash(t, state, *args):
    r, phi, rhor, rhophi, m = state
    altitude = r - R  # Altitude above the Mars surface
    return altitude 

planet_crash.terminal   = True
planet_crash.direction  = -1    # Stops Simulation when crossing from above

def apoapsis_limit(t, state, *args):
    global aerobrake
    if aerobrake:
        r_a_limit = r_a_limit_aero
    else: r_a_limit = r_a_limit_thrust
    r, phi, rhor, rhophi, m = state
    return r - r_a_limit

apoapsis_limit.terminal   = True  
apoapsis_limit.direction  = 1     # Stops Simulation when crossing from within SOI to outside

def stable_orbit_reached(t, state, *args):
    global stable_orbit_time
    return t - stable_orbit_time - 86400 if stable_orbit_time is not None else -1

stable_orbit_reached.terminal = True
stable_orbit_reached.direction = 1

#endregion

#region Simulate Trajectory
def simulate_trajectory(b, r_thrust_descend, r_thrust_ascend, r_p_slow, short_run=True, max_step=5):
    global orbit_insertion_maneuver, is_descending, thrust_active, full_aerobraking, stable_orbit_time, slowdown_start_time, stabilization_start_time

    # Reset mission phase variables
    orbit_insertion_maneuver = True
    full_aerobraking = True
    is_descending = True
    thrust_active = False
    stable_orbit_time = None
    slowdown_start_time = None
    stabilization_start_time = None

    # Starting Location and parameters
    y_0 = -R_SOI                                    # Initial y-coordinate in[km]
    x_0 = b                                         # Initial x-coordinate in[km]
    r_0 = np.sqrt(x_0**2 + y_0**2)                  # Initial orbit radius in [km]
    phi_0 = np.arctan2(-R_SOI, b)                   # Initial orbit angle [rad]
    rho_r_0 = v_inf * np.cos(phi_0 - np.pi / 2)     # Initial radial velocity in [km/s]
    rho_phi_0 = -v_inf * np.sin(phi_0 - np.pi / 2) / r_0  # Initial angular velocity in [rad/s]

    # Solve the equations of motion
    init_val = [r_0, phi_0, rho_r_0, rho_phi_0, m_0]                                     # List with all initial conditions
    p = (thrust, ceff, m_0 - m_p, r_thrust_descend, r_thrust_ascend, r_p_slow)           # Array with all S/C parameters

    # Define the time span for the integration
    t_span = (0, tmax_short if short_run else tmax + 86400)

    # Perform the integration over the entire time span with the dynamic max_step
    trajectory = solve_ivp(eom, t_span, init_val, args=p, method='RK45', rtol=1e-7, atol=1e9, max_step=max_step, events=[planet_crash, apoapsis_limit, stable_orbit_reached])

    # Extract the results
    r = trajectory.y[0, :]  # Radial distances
    m = trajectory.y[4, :]  # Mass

    local_minima_indices = argrelextrema(r, np.less)[0]  # Find local minima in the radial distance data
    local_maxima_indices = argrelextrema(r, np.greater)[0]  # Find local maxima in the radial distance data

    if len(local_minima_indices) >= 2 and len(local_maxima_indices) >= 1:
        periapsis_2nd_orbit = r[local_minima_indices[1]]
        mass_at_2nd_periapsis = m[local_minima_indices[1]]
        apoapsis_1st_orbit = r[local_maxima_indices[0]]

        return periapsis_2nd_orbit, apoapsis_1st_orbit, trajectory, mass_at_2nd_periapsis, m[-1]
    else:
        return None, None, trajectory, None, m[-1]
#endregion

#region Iterate B-Plane-Offset
def find_optimal_b(initial_b, r_p):
    global best_fuel_usage, best_b, best_r_thrust_descend, best_r_thrust_ascend, best_configs, best_trajectory

    b = initial_b
    tolerance = 0.2  # Allowed difference between r_p and 2nd periapsis of trajectory #0.2
    max_iterations = 10
    base_max_step = 5.0
    iteration = 0
    error = None
    while max_iterations > 0:
        iteration += 1

        # Set max_step based on the error from the last iteration
        if error is None:
            max_step = base_max_step

        if iteration > 1 and error is not None:
            if abs(error) < 1:
                max_step = 0.25
            elif abs(error) < 5:
                max_step = 0.5
            elif abs(error) < 10:
                max_step = 1.0
            elif abs(error) < 20:
                max_step = 2.0
            else:
                max_step = base_max_step
        else:
            max_step = base_max_step

        # Run the trajectory simulation with dynamic max_step
        periapsis_2nd_orbit, apoapsis_1st_orbit, trajectory, mass_at_2nd_periapsis, final_mass = simulate_trajectory(b, r_thrust_descend, r_thrust_ascend, r_p_slow, short_run=True, max_step=max_step)

        if periapsis_2nd_orbit is None:
            termination_reason = "Unknown event"
            if trajectory.status == 1:
                if trajectory.t_events[0].size > 0:
                    termination_reason = "Planet Crash"
                    b += 200  # Increase b if we couldn't find a second periapsis
                elif trajectory.t_events[1].size > 0:
                    termination_reason = "Apoapsis limit exceeded"
                    b -= 100  # Decrease b if apoapsis limit is exceeded
            elif trajectory.status == 0:
                termination_reason = "Integration finished (end time reached)"
            elif trajectory.status == -1:
                termination_reason = "Integration step failed"

            max_iterations -= 1
            print(f"  Iteration: {iteration}, b: {b:.4f}, Target r_p: {r_p-R}, Periapsis: None, Error: None, Termination Reason: {termination_reason}")
            continue

        error = periapsis_2nd_orbit - r_p
        fuel_used = m_0 - final_mass  # Use final mass for fuel used calculation

        # Print detailed information including max_step
        print(f"Iteration: {iteration}, Target r_p: {r_p-R} km, Distance to Target: {error:.4f}, max_step: {max_step}") #If interesting add:  b: {b:.0f}, Apoapsis: {apoapsis_1st_orbit-R:.0f}, Actual r_p: {periapsis_2nd_orbit-R:.2f},  Fuel used: {fuel_used:.1f} kg,

        if abs(error) < tolerance:
            if fuel_used < best_fuel_usage:
                best_fuel_usage = fuel_used
                best_b = b
                best_r_thrust_descend = r_thrust_descend
                best_r_thrust_ascend = r_thrust_ascend
                print(f"New best found! Fuel used: {best_fuel_usage:.4f} kg for r_p: {r_p - R:.0f} km")

                best_configs.append({
                    'b': best_b,
                    'r_thrust_descend': best_r_thrust_descend,
                    'r_thrust_ascend': best_r_thrust_ascend,
                    'fuel_used': None,
                    'time_to_stable_orbit': None,
                    'full_trajectory': None,
                })

            break

        # Adjust b based on the error
        baseline_r_thrust_ascend = R + r_thrust_ascending
        scaling_factor = (r_thrust_ascend / baseline_r_thrust_ascend)**1.5

        if abs(error) > 100:
            b -= error*1.2*scaling_factor
        elif 40 < abs(error) < 100:
            b -= error*1.15*scaling_factor
        elif 5 < abs(error) < 40:
            b -= error*1.05*scaling_factor
        elif 0 < abs(error) < 5:
            b -= error*0.95*scaling_factor

        max_iterations -= 1

    return best_b is not None

#endregion

#region Optimize Thrustduration
def simulate_for_r_p(r_p):
    global r_thrust_descend, r_thrust_ascend, best_fuel_usage, best_b, best_r_thrust_descend, best_r_thrust_ascend, tested_combinations, best_configs, r_p_slow, best_trajectory
    
    if r_p - R <= 74:
        r_p_slow = r_p + 20         # r_p = 70 -> r_p_slow = 90         
    elif r_p - R <= 79:
        r_p_slow = r_p + 16         # r_p = 75 -> r_p_slow = 91
    elif r_p - R <= 89:             
        r_p_slow = r_p + 12          # r_p = 80/85 -> r_p_slow = 92/97
    elif r_p - R <= 99:
        r_p_slow = r_p + 8          # r_p = 90/95 -> r_p_slow = 98/103
    else:
        r_p_slow = r_p + 4          # r_p = 100 -> r_p_slow = 104
    
    r_thrust_descend = R + r_thrust_descending  # Reset to initial value
    r_thrust_ascend = R + r_thrust_ascending   # Reset to initial value

    best_fuel_usage = float('inf')
    best_b = None
    best_r_thrust_descend = r_thrust_descend
    best_r_thrust_ascend = r_thrust_ascend
    tested_combinations = set()
    best_configs = []
    best_trajectory = None  # Initialize to store the best trajectory
    
    initial_best_b = b_i
    if find_optimal_b(initial_best_b, r_p):
        print(f"Found initial best b: {best_b:.4f} km for r_p: {r_p-R:.0f} km")
    else:
        print("Failed to find initial best b")

    if Optimize_Thrust_Range and best_b is not None and aerobrake:
        ascend_done = False
        last_valid_b = best_b  # Start with the best initial b found
        previous_best_fuel_usage = best_fuel_usage

        while not ascend_done:
            improved = False

            if not ascend_done:
                r_thrust_ascend -= r_thrust_reduction
                if (r_thrust_descend, r_thrust_ascend) not in tested_combinations:
                    tested_combinations.add((r_thrust_descend, r_thrust_ascend))
                    print(f"\nLowering r_thrust_ascend to {r_thrust_ascend - R:.0f} km for r_p: {r_p - R:.0f} km")
                    found_valid_b = find_optimal_b(last_valid_b, r_p)
                    if found_valid_b:
                        improved = True
                        last_valid_b = best_b  # Update last valid b to the new best b

                        if best_fuel_usage >= previous_best_fuel_usage:
                            ascend_done = True
                            print(f"No lower fuel consumption achieved, stopping further optimization for r_p: {r_p - R:.0f} km at r_thrust: {r_thrust_ascend - R:.0f} km")
                        else:
                            previous_best_fuel_usage = best_fuel_usage
                    else:
                        ascend_done = True  # Stop further attempts if no valid b is found within max iterations
            if not improved:
                ascend_done = True

        if best_b is not None:
            print(f"\nOptimized b: {best_b:.4f} km")
            print(f"Optimized r_thrust_ascend: {best_r_thrust_ascend - R:.0f} km")
            print(f"Optimized fuel usage: {best_fuel_usage:.4f} kg")
        else:
            print("Failed to find an optimized solution.")

        r_thrust_descend = best_r_thrust_descend
        r_thrust_ascend = best_r_thrust_ascend

    if best_b is None:
        print("No valid trajectory found.")
        return {
            'r_p': r_p,
            'best_configs': [],  # Ensure best_configs is always present
            'r_p_slow': r_p_slow  # Ensure r_p_slow is always present
        }

    return {
        'r_p': r_p,
        'best_configs': best_configs,
        'r_p_slow': r_p_slow  # Include r_p_slow for the full trajectory simulation
    }
#endregion

#region Final Simulation of best configuarations
def run_full_trajectory(config):
    global stable_orbit_time, aerobrake, slowdown_start_time, stabilization_start_time

    r_p = config['r_p']
    b = config['b']
    r_thrust_descend = config['r_thrust_descend']
    r_thrust_ascend = config['r_thrust_ascend']
    r_p_slow = config['r_p_slow']
    thrust_only = config['thrust_only']

    # Set aerobrake to false if it's a thrust_only run
    if thrust_only:
        original_aerobrake = aerobrake
        aerobrake = False

    final_sim_start_time = tm.time()
    if r_p-R <= 80:
        _, _, full_trajectory, _, final_mass = simulate_trajectory(b, r_thrust_descend, r_thrust_ascend, r_p_slow, short_run=False, max_step=0.2)
    elif r_p-R <=90:
        _, _, full_trajectory, _, final_mass = simulate_trajectory(b, r_thrust_descend, r_thrust_ascend, r_p_slow, short_run=False, max_step=0.3)
    elif r_p-R <=95:
        _, _, full_trajectory, _, final_mass = simulate_trajectory(b, r_thrust_descend, r_thrust_ascend, r_p_slow, short_run=False, max_step=0.5)
    else:
        _, _, full_trajectory, _, final_mass = simulate_trajectory(b, r_thrust_descend, r_thrust_ascend, r_p_slow, short_run=False, max_step=1.25)

    final_sim_end_time = tm.time()
    final_sim_duration = final_sim_end_time - final_sim_start_time

    if stable_orbit_time is not None:
        stable_orbit_days = stable_orbit_time / 86400
    else:
        stable_orbit_days = None

    fuel_used = m_0 - final_mass

    if stable_orbit_time is not None:
        print(f"TTSO: {stable_orbit_days:.1f} days | r_p: {r_p-R:.1f} km | r_thrust: {r_thrust_ascend-R:.0f} km | Fuel: {fuel_used:.1f} kg | Computation time: {final_sim_duration/60:.2f} min")
    else:
        print(f"Stable orbit not reached for r_p: {r_p-R:.1f} km and r_thrust: {r_thrust_ascend-R:.0f} km with Fuel used: {fuel_used:.1f} kg and Simulation computation time: {final_sim_duration/60:.2f} min")
    
    # Reset aerobrake to its original value if it was modified
    if thrust_only:
        aerobrake = original_aerobrake

    return {
        'r_p': r_p,
        'b': b,
        'r_thrust_descend': r_thrust_descend,
        'r_thrust_ascend': r_thrust_ascend,
        'fuel_used': fuel_used,
        'time_to_stable_orbit': stable_orbit_days,
        'full_trajectory': full_trajectory,
        'slowdown_start_time': slowdown_start_time,  # Store slowdown start time
        'stabilization_start_time': stabilization_start_time  # Store stabilization start time
    }
#endregion

#region Multithreading & Plots
if __name__ == "__main__":
    
    program_start_time  = tm.time()     # Track the start time of the entire program
    results             = []            # Store results for each r_p
    all_configs         = []            # Collect all valid configurations first

    #Simulate Reference Trajectory with Thrusters to establish Orbit
    if Thrust_Only_Run is True:
        result_without_aerobraking = simulate_for_r_p(r_p_orbit)
        best_configs = result_without_aerobraking.get('best_configs', [])
        r_p_slow = result_without_aerobraking.get('r_p_slow', result_without_aerobraking['r_p'] + 5)
        for config in best_configs:
            config['r_p'] = result_without_aerobraking['r_p']
            config['r_p_slow'] = r_p_slow 
            config['thrust_only'] = True
            all_configs.append(config)
    aerobrake = True

    #Simulate Aerobraking Trajectorys until 2nd periapsis to get B-Plane Offset
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_for_r_p, r_p) for r_p in r_p_values]
        for future in as_completed(futures):
            result = future.result()
            best_configs = result.get('best_configs', [])
            r_p_slow = result.get('r_p_slow', result['r_p'] + 5)
            for config in best_configs:
                config['r_p'] = result['r_p']
                config['r_p_slow'] = r_p_slow 
                config['thrust_only'] = False
                all_configs.append(config)

    #Simulate full Aerobraking Trajectory until final orbit is reached
    full_trajectory_results = []
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_full_trajectory, config)
            for config in all_configs
        ]
        for future in as_completed(futures):
            full_trajectory_results.append(future.result())

    #Organize and sort results by Periapsis height (r_p)
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

    #region Plot Propellant consumed vs Time to reach stable Orbit
    if Plot_Comparison:
        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(sorted_r_p_values) + 1))

        for i, r_p in enumerate(sorted_r_p_values):
            results_for_r_p = results_by_r_p[r_p]
            fuel_used = [res['fuel_used'] for res in results_for_r_p]
            time_to_stable_orbit = [res['time_to_stable_orbit'] for res in results_for_r_p]
            labels = [f"{res['r_thrust_ascend'] - R:.0f} km" for res in results_for_r_p]  # Subtract R for the plot

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

                plt.plot(sorted_times, sorted_fuel, linestyle='-', color=colors[i], alpha=0.6, label=f'r_p = {r_p - R:.0f} km')
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

    #region Plot Values
    if Plot_Values:
        # Sort the full_trajectory_results by r_p
        full_trajectory_results_sorted = sorted(full_trajectory_results, key=lambda x: x['r_p'])

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        config_check_ax = plt.axes([0.01, 0.5, 0.15, 0.4], frameon=False)  # Adjust position to the left
        thrust_check_ax = plt.axes([0.01, 0.3, 0.15, 0.2], frameon=False)  # Position below the r_p checkboxes
        parameter_check_ax = plt.axes([0.91, 0.3, 0.08, 0.6], frameon=False)  # Adjust position to the right
        parameter_labels = ['Orbital Energy', 'Altitude', 'Apoapsis', 'Periapsis', 'Mass', 'Velocity']
        parameter_check = CheckButtons(parameter_check_ax, parameter_labels, [False] * len(parameter_labels))

        selected_params = set()
        current_config_index = None  # Initialize as None globally
        current_thrust_values = []  # List to hold current thrust values for the selected r_p

        # Get unique r_p values and corresponding configs
        rp_configs = {}
        for result in full_trajectory_results_sorted:
            rp = result['r_p']
            if rp not in rp_configs:
                rp_configs[rp] = []
            rp_configs[rp].append(result)

        config_labels = [f"r_p = {rp - R:.0f} km" for rp in rp_configs]
        config_check = CheckButtons(config_check_ax, config_labels, [False] * len(config_labels))
        
        # Initialize thrust_check globally
        thrust_check = CheckButtons(thrust_check_ax, [], [])

        def plot_selected_configuration(config_index, thrust_index):
            global current_config_index
            current_config_index = config_index

            if 0 <= config_index < len(full_trajectory_results_sorted):
                selected_results = rp_configs[sorted_r_p_values[config_index]]
                selected_result = selected_results[thrust_index]

                full_trajectory = selected_result['full_trajectory']
                time = full_trajectory.t / 86400  # Convert time to days
                r = full_trajectory.y[0, :]
                phi = full_trajectory.y[1, :]
                rhor = full_trajectory.y[2, :]
                rhophi = full_trajectory.y[3, :]
                mass = full_trajectory.y[4, :]
                height = r - R  # Spacecraft Altitude over ground in [km]
                v = np.sqrt(rhor ** 2 + (r * rhophi) ** 2)  # Velocity in [km/s]
                E = v ** 2 / 2 - mu / r  # Orbital Energy

                subsample_factor = 200  # Adjust this factor to control the amount of data plotted
                subsampled_indices = np.arange(0, len(time), subsample_factor)

                time_sub = time[subsampled_indices]
                height_sub = height[subsampled_indices]
                v_sub = v[subsampled_indices]
                E_sub = E[subsampled_indices]
                mass_sub = mass[subsampled_indices]

                local_minima_indices = argrelextrema(r[subsampled_indices], np.less)[0]
                local_maxima_indices = argrelextrema(r[subsampled_indices], np.greater)[0]
                periapsis_over_orbits = [r[i] - R for i in subsampled_indices[local_minima_indices]]
                apoapsis_over_orbits = [r[i] - R for i in subsampled_indices[local_maxima_indices]]
                orbit_count = range(1, min(len(periapsis_over_orbits), len(apoapsis_over_orbits)) + 1)

                def shade_atmosphere_regions(ax, time_sub, height_sub, atmo_height):
                    in_atmosphere = height_sub <= atmo_height
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
                    if thrust_active:  # If thrust is still active at the end
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

                    if current_config_index is None:
                        return  # No configuration is selected yet

                    ax.clear()

                    selected_results = rp_configs[sorted_r_p_values[current_config_index]]
                    selected_result = selected_results[thrust_index]
                    
                    slowdown_start_time = selected_result.get('slowdown_start_time')
                    stabilization_start_time = selected_result.get('stabilization_start_time')

                    if 'Orbital Energy' in selected_params:
                        ax.plot(time_sub, E_sub, label='Orbital Energy')
                        ax.set_ylabel('Energy (MJ/kg)')
                        ax.set_title('Orbital Energy vs Time')
                        shade_thrust_active(ax, time_sub, mass_sub)
                        shade_atmosphere_regions(ax, time_sub, height_sub, atmo_height)

                    if 'Altitude' in selected_params:
                        ax.plot(time_sub, height_sub, color="purple", label='Altitude')
                        ax.set_ylabel('Height (km)')
                        ax.set_title('Altitude vs Time')
                        ax.set_ylim(0, 50000)
                        ax.axhline(y=atmo_height, color='gray', linestyle='--', label='Atmosphere boundary')
                        ax.axhline(y=r_a - R, color='black', linestyle='--', label='Target Apoapsis')
                        ax.axhline(y=r_p_orbit - R, color='black', linestyle='--', label='Target Periapsis')
                        ax.axhline(y=selected_result['r_p'] - R, color='red', linestyle='--', label='Aerobraking Height')
                        shade_thrust_active(ax, time_sub, mass_sub)
                        shade_atmosphere_regions(ax, time_sub, height_sub, atmo_height)

                    if 'Apoapsis' in selected_params:
                        if Orbit_Count:
                            ax.plot(orbit_count, apoapsis_over_orbits[:len(orbit_count)], color="green", label='Apoapsis Altitude')
                            ax.set_xlabel('Orbit Count')
                        else:
                            time_apoapsis = [time_sub[i] for i in local_maxima_indices]
                            ax.plot(time_apoapsis, [apoapsis_over_orbits[i] for i in range(len(apoapsis_over_orbits))], color="green", label='Apoapsis Altitude')
                            ax.set_xlabel('Time (days)')
                        ax.set_ylabel('Apoapsis Altitude (km)')
                        ax.set_title('Apoapsis vs Time/Orbit Count')
                        ax.axhline(y=r_a - R, color='black', linestyle='--', label='Target Apoapsis')

                    if 'Periapsis' in selected_params:
                        if Orbit_Count:
                            ax.plot(orbit_count, periapsis_over_orbits[:len(orbit_count)], color="orange", label='Periapsis Altitude')
                            ax.set_xlabel('Orbit Count')
                        else:
                            time_periapsis = [time_sub[i] for i in local_minima_indices]
                            ax.plot(time_periapsis, [periapsis_over_orbits[i] for i in range(len(periapsis_over_orbits))], color="orange", label='Periapsis Altitude')
                            ax.set_xlabel('Time (days)')
                        ax.set_ylabel('Periapsis Altitude (km)')
                        ax.set_title('Periapsis vs Time/Orbit Count')
                        ax.axhline(y=r_p_orbit - R, color='black', linestyle='--', label='Target Periapsis')
                        ax.axhline(y=selected_result['r_p'] - R, color='red', linestyle='--', label='Aerobraking Height')

                    if 'Mass' in selected_params:
                        ax.plot(time_sub, mass_sub, color="blue", label='Mass')
                        ax.set_ylabel('Mass (kg)')
                        ax.set_title('Mass vs Time')
                        shade_thrust_active(ax, time_sub, mass_sub)
                        shade_atmosphere_regions(ax, time_sub, height_sub, atmo_height)

                    if 'Velocity' in selected_params:
                        ax.plot(time_sub, v_sub, color="green", label='Velocity')
                        ax.set_ylabel('Velocity (km/s)')
                        ax.set_title('Velocity vs Time')
                        shade_thrust_active(ax, time_sub, mass_sub)
                        shade_atmosphere_regions(ax, time_sub, height_sub, atmo_height)

                    if slowdown_start_time is not None:
                        ax.axvline(x=slowdown_start_time / 86400, color='black', linestyle='--', linewidth=1.5)
                        ax.annotate('Slowdown Aerobraking', xy=(slowdown_start_time / 86400, ax.get_ylim()[1]),
                                    xytext=(60, 10), textcoords='offset points',
                                    fontsize=9, rotation=45, color='black', ha='right')

                    if stabilization_start_time is not None:
                        ax.axvline(x=stabilization_start_time / 86400, color='black', linestyle='--', linewidth=1.5)
                        ax.annotate('Orbit Stabilization', xy=(stabilization_start_time / 86400, ax.get_ylim()[1]),
                                    xytext=(60, 10), textcoords='offset points',
                                    fontsize=9, rotation=45, color='black', ha='right')

                    if selected_params:
                        legend_elements = [
                            Line2D([0], [0], color='gray', linestyle='--', label='Atmosphere Boundary'),
                            Line2D([0], [0], color='black', linestyle='--', label='Target Apoapsis'),
                            Line2D([0], [0], color='black', linestyle='--', label='Target Periapsis'),
                            Line2D([0], [0], color='red', linestyle='--', label='Aerobraking Height'),
                            Patch(facecolor='lightskyblue', edgecolor='none', alpha=0.8, label='Thrust Active'),
                            Patch(facecolor='salmon', edgecolor='none', alpha=0.4, label='In Atmosphere')
                        ]
                        ax.legend(handles=legend_elements, loc='upper right')

                    ax.set_xlabel('Time (days)')
                    ax.grid(True)
                    plt.draw()

                parameter_check.on_clicked(update_plot)

        def update_config_selection(label):
            global thrust_check  # Ensure that we're modifying the global thrust_check
            config_index = config_labels.index(label)
            rp_value = sorted_r_p_values[config_index]
            thrust_values = rp_configs[rp_value]
            
            # Sort thrust_values by r_thrust_ascend in descending order
            sorted_thrust_values = sorted(thrust_values, key=lambda x: x['r_thrust_ascend'], reverse=True)
            thrust_labels = [f"r_thrust = {result['r_thrust_ascend'] - R:.0f} km" for result in sorted_thrust_values]

            # Remove previous thrust checkboxes and recreate them with new labels
            thrust_check_ax.clear()
            thrust_check_ax.set_position([0.01, 0.3, 0.15, 0.2])  # Ensure the position remains consistent below the r_p values
            thrust_check = CheckButtons(thrust_check_ax, thrust_labels, [False] * len(thrust_labels))

            # Set the first thrust value as active and plot it
            thrust_check.set_active(0)  
            plot_selected_configuration(config_index, 0)

            def update_thrust_selection(thrust_label):
                thrust_index = thrust_labels.index(thrust_label)
                plot_selected_configuration(config_index, thrust_index)

            thrust_check.on_clicked(update_thrust_selection)

        config_check.on_clicked(update_config_selection)
        plt.show()
    #endregion

    #region Plot Atmosphere
    if Plot_Atmosphere:
        heights = np.linspace(0, atmo_height, 500)
        densities = [atmospheric_density(height + R) for height in heights]

        plt.figure(figsize=(10, 6))
        plt.plot(densities, heights)
        plt.xscale('log')
        plt.ylabel('Height (km)')
        plt.xlabel('Atmospheric Density (kg/km^3)')
        plt.title('Atmospheric Density vs Height')
        plt.grid(True)
    #endregion
    
    #region Plot Trajectory
    if Plot_Trajectory:
        plt.rcParams["figure.figsize"] = (8, 8)

        fig, ax = plt.subplots()
        check_ax = plt.axes([0.01, 0.3, 0.1, 0.6], frameon=False)  # Position the checkbox

        # Generate labels based on r_p values
        config_labels = [f"r_p = {result['r_p'] - R:.0f} km" for result in full_trajectory_results]
        config_check = CheckButtons(check_ax, config_labels, [False] * len(full_trajectory_results))

        def plot_trajectory_for_selected_config(config_index):
            ax.clear()

            if 0 <= config_index < len(full_trajectory_results):
                selected_result = full_trajectory_results[config_index]
                full_trajectory = selected_result['full_trajectory']

                # Convert trajectory from polar to Cartesian coordinates
                r = full_trajectory.y[0, :]
                phi = full_trajectory.y[1, :]

                # Subsampling the data for plotting
                subsample_factor = 200  # Adjust this factor to control the amount of data plotted
                subsampled_indices = np.arange(0, len(r), subsample_factor)

                x = (r * np.cos(phi))[subsampled_indices]
                y = (r * np.sin(phi))[subsampled_indices]

                ax.plot(x / 1000, y / 1000, 'tab:blue', linewidth=0.4)  # Convert to 1000 km for plotting
                ax.tick_params(labeltop=True, labelright=True)
                ax.set_aspect('equal')

                mars = plt.Circle((0, 0), R / 1000, color='coral', fill=True)
                ax.add_patch(mars)

                atmosphere = plt.Circle((0, 0), (R + atmo_height) / 1000, color='salmon', fill=True, alpha=0.6)
                ax.add_patch(atmosphere)

                Sphere_of_Influence = plt.Circle((0, 0), R_SOI / 1000, color='grey', fill=False, alpha=0.6)
                ax.add_patch(Sphere_of_Influence)

                plt.xlabel("x [$10^3$ km]")
                plt.ylabel("y [$10^3$ km]")
            plt.draw()

        def update_trajectory_selection(label):
            config_index = config_labels.index(label)
            plot_trajectory_for_selected_config(config_index)

        config_check.on_clicked(update_trajectory_selection)
        plt.show()
    #endregion

    plt.show()
#endregion
