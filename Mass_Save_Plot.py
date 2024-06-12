import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema
from concurrent.futures import ProcessPoolExecutor, as_completed
import time as tm

# TRY DECREASING R_DESCENT INSTEAD - CHECK WHY HIGHER APOAPSIS DOESNT RESULT IN LONGER TIME TO REACH ORBIT
# RUN FULL SIMULATIONS PARALLEL FOR MULTIPLE R_P

# === Begin user input  ============================
m_0                 = 2500.0           # Initial spacecraft mass in [kg] TGO Initial mass 3755kg
m_p                 = 2000.0           # Propellant mass in [kg] TGO dry mass 1750kg -> m_p = 2005kg
thrust              = 424.0            # Thrust in [N] TGO Thrust 424N
Isp                 = 326.0            # Specific impulse of TGO in [s] 326.0
c_d                 = 2.2              # Drag coefficient
A                   = 29.3             # Cross-sectional area of the spacecraft in [m^2] TGO 29.3
tint                = 150              # Integration time in days
r_p_values          = np.arange(3396.2 + 110, 3396.2 + 111, 5)  # Start, end, step size
r_p_orbit           = 3396.2 + 400
r_a                 = 3396.2 + 800      # Apoapsis of desired Orbit
r_a_limit_aero      = 3396.2 + 80000    # Apoapsis limit to avoid excessive orbit duration
r_a_limit_thrust    = 3396.2 + 300000
v_inf               = 2.65              # Hyperbolic excess speed at infinity in [km/s]
Plot_Trajectory     = False             # Plot spacecraft trajectory?
Plot_Values         = True              # Plot panel of values?
Optimize_Thrust_Range = False           # Optimize First Descend Thrust range?
plot_configuration_index = 0            # Index of the configuration to plot if Plot_Values is True (1 for first r_p value, 2 for 2nd etc. last for thrust_only run)
Plot_Atmosphere     = False             # Plot atmospheric density over height?
Orbit_Count         = False              # Plot over orbit count if True, otherwise over time
Thrust_Only_Run     = False
# === End user input    ============================

# General constants for the problem
mu              = 42828.37              # Gravitational parameter of Mars in [km^3/s^2]
R               = 3396.2                # Mars Equatorial radius in [km]
R_SOI           = 577269                # Mars Radius of standard Sphere of Influence in [km](Improvement later?)
g0_mars         = 3.72076               # Mars standard gravitational acceleration in [m/s^2]
g0_earth        = 9.780327              # Earth standard gravitational acceleration in [m/s^2]
atmo_density    = 0.06                  # Mars Atmosphere density in [kg/m^3]
atmo_height     = 200                   # Height of the atmosphere boundary in [km]

# Start values for later Iteration
tint_short          = 8.0              # Integration time for fuel optimization
r_thrust_ascending  = R + 3600          ###### TGO Orbit Insertion Maneuver lasted 139mins = 0.0965days #######
r_thrust_descending = R + 9500
b_i                 = R*np.sqrt(2*mu/(R*v_inf**2)+1)+1100    # Impact radius of approach hyperbola

# Convert input into "correct" units
tmax            = tint * 86400          # Convert Integration time in [s]
tmax_short      = tint_short * 86400
A               = A / 1e6               # Convert Cross-Sectional area in [km^2]
atmo_density    = atmo_density * 1e9    # Convert Atmosphere density in [kg/km^3]
thrust          = thrust / 1000.0       # Convert thrust to [kg*km/s^2]
ceff            = Isp * g0_earth / 1000.0     # Effective exhaust velocity in [km/s]

# Initialize mission phase variables
first_descent = True
full_aerobraking = True
is_descending = True
thrust_active = False
stable_orbit_time = None  # Track the time when stable orbit is reached
aerobrake = True  # Use aerobraking for the main simulations

#region Atmosphere
def atmospheric_density(r):
    height = r - R  # Altitude above Mars' surface in km
    atmo_density_200= 0.000095  # Density at 195 km
    H = 195 / np.log(atmo_density / atmo_density_200)  # Scale height based on the new model
    return atmo_density * np.exp(-height / H) if height <= 200 else 0
#endregion

#region Mission Phases
def handle_first_descent(r, phi, rhor, rhophi, m, mdry, t):
    global is_descending, thrust_active, first_descent
    if is_descending and r < r_thrust_descend and m > mdry:
        return 1.0  # Throttle fully open to capture the Spacecraft
    elif not is_descending and r > r_thrust_ascend:
        first_descent = False  # End of first descent phase
        return 0.0  # Throttle fully closed
    return 1.0 if thrust_active else 0.0  # Maintain current state

def handle_slowdown_aerobraking(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t, r_p_slow):
    global thrust_active, full_aerobraking
    if r_a < apoapsis <= (r_a + 800) and abs(r - apoapsis) < 30 and periapsis < r_p_slow and m > mdry:
        return 1.0  # Throttle fully open
    elif periapsis >= r_p_slow+5:
        return 0.0  # Throttle fully closed
    elif apoapsis <= r_a and periapsis > r_p_slow:
        full_aerobraking = False
    return 1.0 if thrust_active else 0.0  # Maintain current state

def handle_orbit_stabilization(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t):
    global thrust_active, stable_orbit_time
    if apoapsis <= r_a and abs(r - apoapsis) < 0.1 and periapsis < r_p_orbit and m > mdry:
        return 1.0  # Throttle fully open at apoapsis once the desired apoapsis is reached to increase periapsis
    elif periapsis >= (r_p_orbit-5):
        if stable_orbit_time is None:
            stable_orbit_time = t 
        return 0.0  # Throttle fully closed once the desired periapsis (and with that the final orbit) is reached
    return 1.0 if thrust_active else 0.0  # Maintain current state

def handle_thrust_only(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t):
    global stable_orbit_time, full_aerobraking
    if apoapsis >= r_a and abs(r - periapsis) <= 2 and m > mdry:
        return 1.0  # Throttle fully open to reach the desired orbit
    elif apoapsis < r_a and abs(r - periapsis) <= 0.1:
        full_aerobraking = False
    return 0.0
#endregion

#region Equations of Motion
def eom(t, state, thrust, ceff, mdry):
    global is_descending, thrust_active, first_descent, full_aerobraking, r_p_slow
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
        if first_descent:
            beta = alpha + np.pi
            throttle = handle_first_descent(r, phi, rhor, rhophi, m, mdry, t)
        elif not first_descent and full_aerobraking:
            beta = alpha + np.pi
            throttle = handle_thrust_only(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t)
        elif not first_descent and not full_aerobraking:
            beta = alpha
            throttle = handle_orbit_stabilization(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t)
    else:
        # Choose appropriate thrust logic based on the mission phase
        if first_descent:
            beta = alpha + np.pi
            throttle = handle_first_descent(r, phi, rhor, rhophi, m, mdry, t)
        elif not first_descent and full_aerobraking:
            beta = alpha
            throttle = handle_slowdown_aerobraking(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t, r_p_slow)
        elif not first_descent and not full_aerobraking:
            beta = alpha
            throttle = handle_orbit_stabilization(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis, t)

    dr = rhor
    dphi = rhophi
    drhor = r * rhophi**2 - mu / r**2 - D / m * np.cos(alpha) + throttle * thrust / m * np.cos(beta)
    drhophi = (-2 * rhor * rhophi - D / m * np.sin(alpha) + throttle * thrust / m * np.sin(beta)) / r
    dm = - throttle * thrust / ceff

    thrust_active = throttle > 0.0

    # Print state variables to track overflow and other anomalies
    if np.any(np.abs([r, phi, m]) > 1e12):  # Adjust threshold as needed
        print(f"Overflow warning: t={t/86400:.4f}, r={r:.4f}, phi={phi},  m={m}")


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
    return t - stable_orbit_time if stable_orbit_time is not None else -1

stable_orbit_reached.terminal = True
stable_orbit_reached.direction = 1
#endregion

def simulate_trajectory(b, short_run=True):
    global first_descent, is_descending, thrust_active, full_aerobraking, stable_orbit_time

    # Reset mission phase variables
    first_descent = True
    full_aerobraking = True
    is_descending = True
    thrust_active = False
    stable_orbit_time = None
    y_0 = -R_SOI  # Negative as it's approaching from below
    x_0 = b
    r_0 = np.sqrt(x_0**2 + y_0**2)  # Initial orbit radius in [km]
    phi_0 = np.arctan2(-R_SOI, b)  # Initial orbit angle [rad]
    rho_r_0 = v_inf * np.cos(phi_0 - np.pi / 2)  # Initial radial velocity in [km/s]
    rho_phi_0 = -v_inf * np.sin(phi_0 - np.pi / 2) / r_0  # Initial angular velocity in [rad/s]

    # Solve the equations of motion
    init_val = [r_0, phi_0, rho_r_0, rho_phi_0, m_0]  # List with all initial conditions
    p = (thrust, ceff, m_0 - m_p)  # Array with all S/C parameters

    if short_run:
        trajectory = solve_ivp(eom, (0, tmax_short), init_val, args=p, method='RK45', rtol=2.5e-14, atol=1e-20, max_step = 4, events=[planet_crash, apoapsis_limit, stable_orbit_reached])
    else:
        trajectory = solve_ivp(eom, (0,tmax), init_val, args=p, method='RK45', rtol=2.5e-14, atol=1e-20, max_step = 8, events=[planet_crash, apoapsis_limit, stable_orbit_reached])

    # Convert trajectory from (r,phi) to (x,y) and other calculations for plots
    r = trajectory.y[0, :]
    m = trajectory.y[4, :]  # Extract the mass data from the trajectory

    local_minima_indices = argrelextrema(r, np.less)[0]  # Find local minima in the radial distance data
    local_maxima_indices = argrelextrema(r, np.greater)[0]  # Find local maxima in the radial distance data

    if len(local_minima_indices) >= 2 and len(local_maxima_indices) >= 1:
        periapsis_2nd_orbit = r[local_minima_indices[1]]
        mass_at_2nd_periapsis = m[local_minima_indices[1]]
        apoapsis_1st_orbit = r[local_maxima_indices[0]]

        return periapsis_2nd_orbit, apoapsis_1st_orbit, trajectory, mass_at_2nd_periapsis, m[-1]
    else:
        return None, None, trajectory, None, m[-1]

def find_optimal_b(initial_b, r_p):
    global best_fuel_usage, best_b, best_r_thrust_descend, best_r_thrust_ascend, best_configs, best_trajectory
    
    b = initial_b
    tolerance = 0.1 
    max_iterations = 10

    while max_iterations > 0:
        periapsis_2nd_orbit, apoapsis_1st_orbit, trajectory, mass_at_2nd_periapsis, final_mass = simulate_trajectory(b, short_run=True)

        if periapsis_2nd_orbit is None:
            termination_reason = "Unknown event"
            if trajectory.status == 1:
                if trajectory.t_events[0].size > 0:
                    termination_reason = "Planet Crash"
                    b += 200  # Increase b if we couldn't find a second periapsis
                elif trajectory.t_events[1].size > 0:
                    termination_reason = "Apoapsis limit exceeded"
                    b -= 50  # Decrease b if apoapsis limit is exceeded
            elif trajectory.status == 0:
                termination_reason = "Integration finished (end time reached)"
            elif trajectory.status == -1:
                termination_reason = "Integration step failed"

            max_iterations -= 1
            print(f"  Iteration: {10 - max_iterations}, b: {b:.4f}, Periapsis: None, Error: None, Termination Reason: {termination_reason}")
            continue

        error = periapsis_2nd_orbit - r_p
        fuel_used = m_0 - final_mass  # Use final mass for fuel used calculation
        print(f"  Iteration: {10 - max_iterations}, b: {b:.4f}, Periapsis: {periapsis_2nd_orbit-R:.2f}, Distance to r_p: {error:.4f}, Apoapsis: {apoapsis_1st_orbit-R:.0f}, Fuel used: {fuel_used:.4f} kg")

        if abs(error) < tolerance:
            if fuel_used < best_fuel_usage:
                best_fuel_usage = fuel_used
                best_b = b
                best_r_thrust_descend = r_thrust_descend - R
                best_r_thrust_ascend = r_thrust_ascend - R
                print(f"    New best found! Fuel used: {best_fuel_usage:.4f} kg")

                # Run full simulation for the new best configuration
                final_sim_start_time = tm.time()
                _, _, full_trajectory, mass_at_2nd_periapsis, final_mass = simulate_trajectory(best_b, short_run=False)
                final_sim_end_time = tm.time()
                final_sim_duration = final_sim_end_time - final_sim_start_time

                if stable_orbit_time is not None:
                    stable_orbit_days = stable_orbit_time / 86400
                    print(f"Time to reach stable Orbit {stable_orbit_days:.2f} days")
                else:
                    stable_orbit_days = None
                    print("Stable orbit was not reached.")

                print(f"Full Simulation computation time: {final_sim_duration/60:.2f} min")
                print(f"Fuel used: {m_0 - final_mass:.2f} kg")
                
                best_configs.append({
                    'b': best_b,
                    'r_thrust_descend': best_r_thrust_descend,
                    'r_thrust_ascend': best_r_thrust_ascend,
                    'fuel_used': m_0 - final_mass,
                    'time_to_stable_orbit': stable_orbit_days,
                })

                # Save the trajectory from the long run
                best_trajectory = full_trajectory

            break

        # Adjust b based on the error
        if abs(error) > 100:
            b -= error*1.175
        elif 1 < abs(error) < 100:
            b -= error*1.2
        elif abs(error)<2:
            b -= error

        max_iterations -= 1

    return best_b is not None

def simulate_for_r_p(r_p):
    # Adjust related parameters
    global r_thrust_descend, r_thrust_ascend, best_fuel_usage, best_b, best_r_thrust_descend, best_r_thrust_ascend, tested_combinations, best_configs, r_p_slow, best_trajectory
    
    # Set r_p_slow dynamically
    r_p_slow = r_p + 5
    
    # Initialize thrust parameters for each r_p
    r_thrust_descend = r_thrust_descending  # Reset to initial value
    r_thrust_ascend = r_thrust_ascending   # Reset to initial value

    best_fuel_usage = float('inf')
    best_b = None
    best_r_thrust_descend = r_thrust_descend - R
    best_r_thrust_ascend = r_thrust_ascend - R
    tested_combinations = set()
    best_configs = []
    best_trajectory = None  # Initialize to store the best trajectory
    
    # Use find_optimal_b for the initial search
    initial_best_b = b_i
    if find_optimal_b(initial_best_b, r_p):
        print(f"Found initial best b: {best_b:.4f} km")
    else:
        print("Failed to find initial best b")

    # Continue reducing r_thrust_ascend if Optimize_Thrust_Range is True
    if Optimize_Thrust_Range and best_b is not None and aerobrake:
        ascend_done = False
        last_valid_b = best_b  # Start with the best initial b found
        previous_best_fuel_usage = best_fuel_usage

        while not ascend_done:
            improved = False

            # Attempt to reduce r_thrust_ascend
            if not ascend_done:
                r_thrust_ascend -= 200
                if (r_thrust_descend, r_thrust_ascend) not in tested_combinations:
                    tested_combinations.add((r_thrust_descend, r_thrust_ascend))
                    print(f"\nTrying lower r_thrust_ascend = {r_thrust_ascend - R:.0f} km")
                    found_valid_b = find_optimal_b(last_valid_b, r_p)
                    if found_valid_b:
                        improved = True
                        last_valid_b = best_b  # Update last valid b to the new best b

                        # Check if the fuel usage improved
                        if best_fuel_usage >= previous_best_fuel_usage:
                            ascend_done = True
                            print("No lower fuel consumption achieved, stopping further optimization.")
                        else:
                            previous_best_fuel_usage = best_fuel_usage
                            # Perform the long run for the best b found
                            final_sim_start_time = tm.time()
                            _, _, full_trajectory, mass_at_2nd_periapsis, final_mass = simulate_trajectory(best_b, short_run=False)
                            final_sim_end_time = tm.time()
                            final_sim_duration = final_sim_end_time - final_sim_start_time

                            if stable_orbit_time is not None:
                                stable_orbit_days = stable_orbit_time / 86400
                                print(f"Time to reach stable Orbit {stable_orbit_days:.2f} days")
                            else:
                                stable_orbit_days = None
                                print("Stable orbit was not reached.")

                            print(f"Full Simulation computation time: {final_sim_duration/60:.2f} min")
                            print(f"Fuel used: {m_0 - final_mass:.2f} kg")
                            
                            best_configs.append({
                                'b': best_b,
                                'r_thrust_descend': best_r_thrust_descend,
                                'r_thrust_ascend': best_r_thrust_ascend,
                                'fuel_used': m_0 - final_mass,
                                'time_to_stable_orbit': stable_orbit_days,
                            })
                            best_trajectory = full_trajectory  # Save the last valid trajectory
                    else:
                        ascend_done = True  # Stop further attempts if no valid b is found within max iterations
            if not improved:
                ascend_done = True

        # Output the results of the optimization
        if best_b is not None:
            print(f"\nOptimized b: {best_b:.4f} km")
            print(f"Optimized r_thrust_ascend: {best_r_thrust_ascend:.0f} km")
            print(f"Optimized fuel usage: {best_fuel_usage:.4f} kg")
        else:
            print("Failed to find an optimized solution.")

        # Use optimized thrust parameters if optimization was performed
        r_thrust_descend = best_r_thrust_descend + R
        r_thrust_ascend = best_r_thrust_ascend + R

    if best_b is None or best_trajectory is None:
        print("No valid trajectory found. Using the last valid full run trajectory.")
        if best_trajectory is None:
            return {
                'r_p': r_p,
                'best_configs_fuel_used': [],
                'best_configs_time_to_stable_orbit': [],
                'best_configs_labels': []
            }

    # Use the saved trajectory from the long run
    full_trajectory = best_trajectory

    # Check and print the time to reach stable orbit
    if stable_orbit_time is not None:
        stable_orbit_days = stable_orbit_time / 86400
        print(f"Time to reach stable orbit: {stable_orbit_days:.2f} days")
    else:
        stable_orbit_days = None
        print("Stable orbit was not reached during the simulation.")

    # Final trajectory and other results
    r = full_trajectory.y[0, :]
    phi = full_trajectory.y[1, :]
    rhor = full_trajectory.y[2, :]
    rhophi = full_trajectory.y[3, :]
    mass = full_trajectory.y[4, :]
    x = r * np.cos(phi) 
    y = r * np.sin(phi)  
    vr = rhor  # Radial velocity in [km/s]
    vt = r * rhophi  # Transversal velocity in [km/s]
    v = np.sqrt(rhor**2 + (r * rhophi)**2)  # Velocity in [km/s]
    height = r - R  # Spacecraft height over ground in [km]
    E = 0.5 * (rhor**2 + (r * rhophi)**2) - mu / r  # Orbital energy in [MJ/kg]
    a = - mu / (2 * E)  # Semi-major axis in [km]
    h = r**2 * rhophi  # Orbital angular momentum in [km^2/s]
    e = np.sqrt(1 + (2 * E * h**2) / (mu**2))  # Eccentricity

    # Count the number of orbits by detecting periapsis passages
    local_minima_indices = argrelextrema(r, np.less)[0]  # Find local minima in the radial distance data
    local_maxima_indices = argrelextrema(r, np.greater)[0]  # Find local maxima in the radial distance data
    orbit_count = len(local_minima_indices)  # Count the number of orbits
    periapsis_over_orbits = [r[idx] for idx in local_minima_indices]  # Periapsis values for each orbit
    apoapsis_over_orbits = [r[idx] for idx in local_maxima_indices]  # Apoapsis values for each orbit

    # Save the best configurations for the current r_p value
    result = {
        'r_p': r_p,
        'best_configs_fuel_used': [config['fuel_used'] for config in best_configs],
        'best_configs_time_to_stable_orbit': [config['time_to_stable_orbit'] for config in best_configs if config['time_to_stable_orbit'] is not None],
        'best_configs_labels': [f"ascend: {config['r_thrust_ascend']}" for config in best_configs],
        # Store the trajectory data for later plotting
        'time': full_trajectory.t,
        'r': r,
        'rhor': rhor,
        'rhophi': rhophi,
        'mass': mass,
        'E': E,
        'height': height,
        'v': v,
    }
    return result

if __name__ == "__main__":
    # Track the start time of the entire program
    program_start_time = tm.time()

    # Store results for each r_p
    results = []

    # First run with aerobrake set to False
    if Thrust_Only_Run is True:
        aerobrake = False
        result_without_aerobraking = simulate_for_r_p(r_p_orbit)
        results.append(result_without_aerobraking)
    aerobrake = True

    # Use concurrent.futures to parallelize the simulation for each r_p value
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(simulate_for_r_p, r_p) for r_p in r_p_values]
        for future in as_completed(futures):
            results.append(future.result())

    # Sort results by pericenter height (r_p)
    results.sort(key=lambda x: x['r_p'])

    # Plot Propellant consumed vs Time to reach stable orbit for each r_p value
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(r_p_values) + 1))

    for i, result in enumerate(results):
        fuel_used = result['best_configs_fuel_used']
        time_to_stable_orbit = result['best_configs_time_to_stable_orbit']
        labels = result['best_configs_labels']
        
        if len(time_to_stable_orbit) > 0:
            sorted_indices = np.argsort(time_to_stable_orbit)
            sorted_times = np.array(time_to_stable_orbit)[sorted_indices]
            sorted_fuel = np.array(fuel_used)[sorted_indices]
            sorted_labels = np.array(labels)[sorted_indices]

            plt.plot(sorted_times, sorted_fuel, linestyle='-', color=colors[i], alpha=0.6, label=f'r_p = {result["r_p"] - 3396.2:.0f} km')
            plt.scatter(sorted_times, sorted_fuel, color=colors[i])

            # Add annotations for each point
            for j, (time, fuel, label) in enumerate(zip(sorted_times, sorted_fuel, sorted_labels)):
                plt.annotate(label, (time, fuel), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

    plt.xlabel('Time to Reach Stable Orbit (days)')
    plt.ylabel('Propellant Consumed (kg)')
    plt.title('Propellant Consumed vs Time to Reach Stable Orbit for Different r_p')
    plt.legend(title='Pericenter Heights', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()

    # Track the end time of the entire program
    program_end_time = tm.time()
    program_duration = (program_end_time - program_start_time)/60

    # Print the total duration
    print(f"Total program duration: {program_duration:.2f} minutes")

    # Plot the selected configuration if Plot_Values is True
    if Plot_Values:
        if 0 <= plot_configuration_index < len(results):
            selected_result = results[plot_configuration_index]
            
            time = selected_result['time'] / 86400  # Convert time to days
            r = selected_result['r']
            rhor = selected_result['rhor']
            rhophi = selected_result['rhophi']
            mass = selected_result['mass']
            E = selected_result['E']
            height = selected_result['height']
            v = selected_result['v']

            local_minima_indices = argrelextrema(r, np.less)[0]  # Find local minima in the radial distance data
            local_maxima_indices = argrelextrema(r, np.greater)[0]  # Find local maxima in the radial distance data
            periapsis_over_orbits = [r[idx] - R for idx in local_minima_indices]  # Periapsis values for each orbit
            apoapsis_over_orbits = [r[idx] - R for idx in local_maxima_indices]  # Apoapsis values for each orbit
            orbit_count = range(1, min(len(periapsis_over_orbits), len(apoapsis_over_orbits)) + 1)

            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            def shade_atmosphere_regions(ax, time, height, atmo_height):
                in_atmosphere = height <= atmo_height
                regions = []
                start = None

                for i in range(len(time)):
                    if in_atmosphere[i] and start is None:
                        start = time[i]
                    elif not in_atmosphere[i] and start is not None:
                        end = time[i]
                        regions.append((start, end))
                        start = None

                if start is not None:
                    regions.append((start, time[-1]))

                for start, end in regions:
                    ax.axvspan(start, end, color='salmon', alpha=0.4)

            def shade_thrust_active(ax, time, mass):
                thrust_active_regions = []
                thrust_active = False
                for i in range(1, len(mass)):
                    if mass[i] < mass[i - 1]:
                        if not thrust_active:
                            thrust_start = time[i - 1]
                            thrust_active = True
                    elif thrust_active:
                        thrust_end = time[i - 1]
                        thrust_active_regions.append((thrust_start, thrust_end))
                        thrust_active = False
                if thrust_active:  # If thrust is still active at the end
                    thrust_end = time[-1]
                    thrust_active_regions.append((thrust_start, thrust_end))

                for start, end in thrust_active_regions:
                    ax.axvspan(start, end, color='lightskyblue', alpha=0.8)

            if Orbit_Count:
                # Apoapsis Height vs Orbit Count
                axs[0, 1].plot(orbit_count, apoapsis_over_orbits[:len(orbit_count)], color="green", label='Apoapsis Height')
                axs[0, 1].set_yscale('log')
                axs[0, 1].set_xlabel('Orbit Count')
                axs[0, 1].set_ylabel('Apoapsis Height (km)')
                axs[0, 1].set_title('Apoapsis Height vs Orbit Count')
                axs[0, 1].axhline(y=r_a - R, color='black', linestyle='--', label='Target Apoapsis')
                axs[0, 1].grid(True)

                # Periapsis Height vs Orbit Count
                axs[1, 1].plot(orbit_count, periapsis_over_orbits[:len(orbit_count)], color="orange", label='Periapsis Height')
                axs[1, 1].set_xlabel('Orbit Count')
                axs[1, 1].set_ylabel('Periapsis Height (km)')
                axs[1, 1].set_title('Periapsis Height vs Orbit Count')
                axs[1, 1].axhline(y=r_p_orbit - R, color='black', linestyle='--', label='Target Periapsis')
                axs[1, 1].grid(True)
            else:
                # Time array for the length of apoapsis and periapsis arrays
                time_apoapsis = [time[i] for i in local_maxima_indices]
                time_periapsis = [time[i] for i in local_minima_indices]

                # Apoapsis Height vs Time
                axs[0, 1].plot(time_apoapsis, [apoapsis_over_orbits[i] for i in range(len(apoapsis_over_orbits))], color="green", label='Apoapsis Height')
                axs[0, 1].set_xlabel('Time (days)')
                axs[0, 1].set_ylabel('Apoapsis Height (km)')
                axs[0, 1].set_title('Apoapsis Height vs Time')
                axs[0, 1].axhline(y=r_a - R, color='black', linestyle='--', label='Target Apoapsis')
                axs[0, 1].grid(True)

                # Periapsis Height vs Time
                axs[1, 1].plot(time_periapsis, [periapsis_over_orbits[i] for i in range(len(periapsis_over_orbits))], color="orange", label='Periapsis Height')
                axs[1, 1].set_xlabel('Time (days)')
                axs[1, 1].set_ylabel('Periapsis Height (km)')
                axs[1, 1].set_title('Periapsis Height vs Time')
                axs[1, 1].axhline(y=r_p_orbit - R, color='black', linestyle='--', label='Target Periapsis')
                axs[1, 1].grid(True)

            # Spacecraft velocity vs Time
            axs[0, 0].plot(time, v*1000, label='Spacecraft Velocity')
            shade_thrust_active(axs[0, 0], time, mass)
            shade_atmosphere_regions(axs[0, 0], time, height, atmo_height)
            axs[0, 0].set_xlabel('Time (days)')
            axs[0, 0].set_ylabel('Spacecraft Velocity (m/s)')
            axs[0, 0].set_title('Spacecraft Velocity vs Time')
            axs[0, 0].grid(True)

            # Spacecraft Height vs Time
            axs[1, 0].plot(time, height, color="purple", label='Spacecraft Height')
            shade_thrust_active(axs[1, 0], time, mass)
            shade_atmosphere_regions(axs[1, 0], time, height, atmo_height)
            axs[1, 0].set_xlabel('Time (days)')
            axs[1, 0].set_ylabel('Height (km)')
            axs[1, 0].set_title('Spacecraft Height vs Time')
            axs[1, 0].set_ylim(0, 50000)
            axs[1, 0].axhline(y=atmo_height, color='gray', linestyle='--', label='Atmosphere boundary')
            axs[1, 0].axhline(y=r_a - R, color='black', linestyle='--', label='Target Apoapsis')
            axs[1, 0].axhline(y=r_p_orbit - R, color='black', linestyle='--', label='Target Periapsis')
            axs[1, 0].legend()
            axs[1, 0].grid(True)

            plt.tight_layout()
        else:
            print(f"Invalid configuration index: {plot_configuration_index}. Please choose a valid index.")

    if Plot_Atmosphere:
        heights = np.linspace(0, atmo_height, 500)
        densities = [atmospheric_density(height + R) for height in heights]

        plt.figure(figsize=(10, 6))
        plt.plot(densities, heights)
        plt.xscale('log')
        plt.ylabel('Height (km)')
        plt.xlabel('Atmospheric Density (kg/km^3)')
        plt.title('Atmospheric Density vs Height')
        plt.axvline(x=31.2, color='black', linestyle='--', label='Highest atmospheric density TGO measured at 102 km altitude')
        plt.axvline(x=1.722, color='grey', linestyle='--', label='Lowest atmospheric density TGO measured at 121 km altitude')
        plt.grid(True)

    plt.show()
