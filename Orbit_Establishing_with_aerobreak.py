import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.signal import argrelextrema

# === Begin user input  ============================
m_0                 = 1000.0            # Initial spacecraft mass in [kg]
m_p                 = 150.0             # Propellant mass in [kg]
thrust              = 424.0             # Thrust in [N] Trace Gas Orbiters Thrust
Isp                 = 4000.0            # Specific impulse in [s]
c_d                 = 2.2               # Drag coefficient
A                   = 10.0              # Cross-sectional area of the spacecraft in [m^2]
tint                = 90                # Integration time in days
points              = 432000.0          # Number of output points
r_p_target          = 3396.2 + 250      # Pericenter of Aiming approach hyperbola
r_p                 = 3396.2 + 150      # Pericenter of actual approach hyperbola (Aerobraking height) 
r_p_orbit           = 3396.2 + 400      # Pericenter of stable endorbit
r_a                 = 3396.2 + 1100     # Apoapsis of desired Orbit
v_inf               = 2.65              # Hyperbolic excess speed at infinity in [km/s]
r_thrust_descend    = 3396.2 + 2500     # Altitude for starting thrust in km
r_thrust_ascend     = 3396.2 + 800      # Altitude for stopping thrust in km
# === End user input    ============================

# General constants for the problem
mu              = 42828.37                                              # Gravitational parameter of Mars in [km^3/s^2]
R               = 3396.2                                                # Mars Equatorial radius in [km]
R_SOI           = 577269                                                # Mars Radius of standard Sphere of Influence in [km](Improvement later?)
g0              = 3.72076                                               # Mars standard gravitational acceleration in [m/s^2]
atmo_density    = 0.02                                                  # Mars Atmosphere density in [kg/m^3]
base_scale      = 11.1                                                  # Scale height of the atmosphere in [km]
atmo_height     = 200                                                   # Height of the atmosphere boundary in [km]
b_i             = R*np.sqrt(2*mu/(R*v_inf**2)+1)                        # Impact radius of approach hyperbola
b               = r_p_target*np.sqrt(2*mu/(r_p_target*v_inf**2)+1)      # Aiming radius

# Convert input into "correct" units
tmax            = tint * 86400          # Convert Integration time in [s]
A               = A / 1e6               # Convert Cross-Sectional area in [km^2]
atmo_density    = atmo_density * 1e9    # Convert Atmosphere density in [kg/km^3]
thrust          = thrust / 1000.0       # Convert thrust to [kg*km/s^2]
ceff            = Isp * g0 / 1000.0     # Effective exhaust velocity in [km/s]

# Initial Conditions
x_0         = b                                         # Rightward offset in x-direction
y_0         = -R_SOI                                    # Negative as it's approaching from below
r_0         = np.sqrt(x_0**2 + y_0**2)                  # Initial orbit radius in [km]
phi_0       = np.arctan2(-R_SOI, b)                     # Initial orbit angle [rad]           
rho_r_0     = v_inf * np.cos(phi_0 - np.pi / 2) * 1.0104         # Initial radial velocity in [km/s]   
rho_phi_0   = -v_inf * np.sin(phi_0 - np.pi / 2) / r_0  # Initial angular velocity in [rad/s] 
first_descent = True                                    # Start with first Descent
is_descending = True                                    # Start descending
thrust_active = False                                   # Start without thrust
thrust_activities = []

# Atmospheric density
def atmospheric_density(r):
    height = r - R  # Altitude above Mars' surface in km
    if height < 0:
        return 0
    # Adjust scale height based on altitude
    if height < 70:
        scale_height = base_scale
    elif height < 100:
        scale_height = base_scale * 1.5
    else:
        scale_height = base_scale * 2  # Doubling the scale height for higher altitudes

    return atmo_density * np.exp(-height / scale_height) if height <= 200 else 0

#region Mission Phases
def handle_first_descent(r, phi, rhor, rhophi, m, mdry):
    global is_descending, thrust_active, first_descent, thrust_activities
    if is_descending and r < r_thrust_descend and m > mdry:
        return 1.0                          # Throttle fully open to capture the Spacecraft
    elif not is_descending and r > r_thrust_ascend:
        first_descent = False               # End of first descent phase
        return 0.0                          # Throttle fully closed
    return 1.0 if thrust_active else 0.0    # Maintain current state

def handle_subsequent_phases(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis):
    global thrust_active
    if apoapsis <= r_a and abs(r - apoapsis) < 0.1 and periapsis < r_p_orbit and m > mdry:
        return 1.0                          # Throttle fully open at apoapsis once the desired apoapsis is reached
    elif periapsis >= r_p_orbit:
        return 0.0                          # Throttle fully closed once the desired periapsis (and with that the final orbit) is reached
    return 1.0 if thrust_active else 0.0    # Maintain current state
#endregion

#region Equations of Motion
def eom(t, state, thrust, ceff, mdry):
    global is_descending, thrust_active, first_descent
    r, phi, rhor, rhophi, m = state

    v = np.sqrt(rhor**2 + (r * rhophi)**2)      # Total velocity
    alpha = np.arctan2(r * rhophi, rhor)        # Flight path angle
    E = v**2 / 2 - mu / r                       # Orbital Energy
    h = r**2 * rhophi                           # Angular Momentum
    a = -mu / (2 * E)                           # Semi-major axis
    e = np.sqrt(1 + (2 * E * h**2) / (mu**2))   # Eccentricity
    apoapsis = a * (1 + e)                      # Current apoapsis
    periapsis = a * (1 - e)                     # Current periapsis
    is_descending = rhor < 0                    # Check if the spacecraft is descending
    throttle = 0.0                              # Default to no thrust
    atmo_density = atmospheric_density(r)
    D = (0.5 * c_d * atmo_density * A * v ** 2) / 1000 if atmo_density > 0 else 0  

    # Choose appropriate thrust logic based on the mission phase
    if first_descent:
        beta = alpha + np.pi
        throttle = handle_first_descent(r, phi, rhor, rhophi, m, mdry)
    else:
        beta = alpha
        throttle = handle_subsequent_phases(r, phi, rhor, rhophi, m, mdry, apoapsis, periapsis)

    dr = rhor
    dphi = rhophi
    drhor = r * rhophi**2 - mu / r**2 - D / m * np.cos(alpha) + throttle * thrust / m * np.cos(beta)
    drhophi = (-2 * rhor * rhophi - D / m * np.sin(alpha) + throttle * thrust / m * np.sin(beta)) / r
    dm = -throttle * thrust / ceff 

    thrust_active = throttle > 0.0
    thrust_activities.append(thrust_active)

    return [dr, dphi, drhor, drhophi, dm]
#endregion

#region Simulation termination conditions
def planet_crash(t, state, *args):
    r, phi, rhor, rhophi, m = state
    altitude = r - R  # Altitude above the Mars surface
    return altitude 

planet_crash.terminal   = True
planet_crash.direction  = -1    # Stops Simulation when crossing from above

def exit_soi(t, state, *args):
    r, phi, rhor, rhophi, m = state
    return r - R_SOI        # Returns negative when r is less than R_SOI

exit_soi.terminal   = True  
exit_soi.direction  = 1     # Stops Simulation when crossing from within SOI to outside

#endregion

# Solve the equations of motion
init_val  = [r_0, phi_0, rho_r_0, rho_phi_0, m_0]   # List with all initial conditions
p = (thrust, ceff, m_0 - m_p)                       # Array with all S/C parameters
t = np.arange(0.0, tmax, tmax / points)             # Simulation time range
trajectory = solve_ivp(eom, (0, tmax), init_val,args=p, method='DOP853', rtol=1e-13, atol=1e-14, t_eval=t, events=[planet_crash, exit_soi])

# Convert trajectory from (r,phi) to (x,y) and other calculations for plots
r       = trajectory.y[0, :]
phi     = trajectory.y[1, :]
rhor    = trajectory.y[2, :]
rhophi  = trajectory.y[3, :]
m       = trajectory.y[4, :]
x       = r * np.cos(phi) 
y       = r * np.sin(phi)  
vr      = rhor                                          # Radial velocity in [km/s]
vt      = r * rhophi                                    # Transversal velocity in [km/s]
v       = np.sqrt(rhor**2 + (r * rhophi)**2)            # Velocity in [km/s]
height  = r - R                                         # Spacecraft height over ground in [km]
E       = 0.5 * (rhor**2 + (r * rhophi)**2) - mu / r    # Orbital energy in [MJ/kg]
a       = - mu / (2 * E)                                # Semi-major axis in [km]
h       = r**2 * rhophi                                 # Orbital angular momentum in [km^2/s]
e       = np.sqrt(1 + (2 * E * h**2) / (mu**2))         # Eccentricity

#region Plots

inside_atmosphere = height <= atmo_height
def add_atmosphere_shading(ax, t, mask, color='salmon', alpha=0.3): # Define a function to add shaded regions for atmosphere presence
    start = None
    for i, in_atm in enumerate(mask):
        if in_atm and start is None:
            start = t[i]
        elif not in_atm and start is not None:
            ax.axvspan(start / 86400, t[i] / 86400, color=color, alpha=alpha)
            start = None
    if start is not None:
        ax.axvspan(start / 86400, t[-1] / 86400, color=color, alpha=alpha)

thrust_activities = [False] * len(trajectory.t) # Define Thrust Activity
prev_mass = m_0
for i, mass in enumerate(trajectory.y[4, :]):
    if mass < prev_mass:                        # Thrust active when mass decreases
        thrust_activities[i] = True
    prev_mass = mass

def add_thrust_shading(ax, t, thrust_activities, color='lightskyblue', alpha=0.3): # Function to add shaded regions where thrust is active
    start = None
    for i, active in enumerate(thrust_activities):
        if active and start is None:
            start = t[i]
        elif not active and start is not None:
            ax.axvspan(start / 86400, t[i] / 86400, color=color, alpha=alpha)
            start = None
    if start is not None:
        ax.axvspan(start / 86400, t[-1] / 86400, color=color, alpha=alpha)

# Plot the spacecraft trajectory
plt.rcParams["figure.figsize"] = (8, 8)
fig, ax = plt.subplots()
ax.plot(x / 1000, y / 1000, 'tab:blue', linewidth=0.4)
ax.tick_params(labeltop=True, labelright=True)
ax.set_aspect('equal')
mars = plt.Circle((0, 0), R / 1000, color='coral', fill=True)
ax.add_patch(mars)
atmosphere = plt.Circle((0, 0), (R + atmo_height) / 1000, color='salmon', fill=True, alpha=0.6)
ax.add_patch(atmosphere)
Sphere_of_Influence = plt.Circle((0, 0), (R_SOI) / 1000, color='grey', fill=False, alpha=0.6)
ax.add_patch(Sphere_of_Influence)
plt.title("Spacecraft Trajectory")
plt.xlabel("x [$10^3$ km]")
plt.ylabel("y [$10^3$ km]")

# Plot a panel of values
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Orbital Energy plot
axs[0, 0].plot(trajectory.t / 86400, E)
axs[0, 0].grid()
axs[0, 0].set_title("Orbital Energy")
axs[0, 0].set_ylabel("E [MJ/kg]")
add_atmosphere_shading(axs[0, 0], trajectory.t, inside_atmosphere)
add_thrust_shading(axs[0, 0], trajectory.t, thrust_activities)

# Semi-major axis plot
axs[0, 1].plot(trajectory.t / 86400, a / 1e3, 'tab:orange')
axs[0, 1].grid()
axs[0, 1].set_title("Semi-major Axis")
axs[0, 1].set_ylabel("a [km]")
add_atmosphere_shading(axs[0, 1], trajectory.t, inside_atmosphere)
add_thrust_shading(axs[0, 1], trajectory.t, thrust_activities)

# Spacecraft velocity plot
axs[1, 0].plot(trajectory.t / 86400, v, 'tab:green')
axs[1, 0].grid()
axs[1, 0].set_title("Spacecraft Velocity")
axs[1, 0].set_xlabel("Time [days]")
axs[1, 0].set_ylabel("v [km/s]")
add_atmosphere_shading(axs[1, 0], trajectory.t, inside_atmosphere)
add_thrust_shading(axs[1, 0], trajectory.t, thrust_activities)

# Height over Mars surface plot
axs[1, 1].plot(trajectory.t / 86400, height, color="purple")
axs[1, 1].grid()
axs[1, 1].set_title("Height over Mars Surface")
axs[1, 1].set_xlabel("Time [days]")
axs[1, 1].set_ylabel("Height [km]")
axs[1, 1].axhline(y=atmo_height, color='gray', linestyle='--', label='Atmosphere boundary')
axs[1, 1].axhline(y=r_a - R, color='black', linestyle='--', label='Target Apoapsis')
axs[1, 1].axhline(y=r_p_orbit - R, color='black', linestyle='--', label='Target Periapsis')
axs[1, 1].legend()
add_atmosphere_shading(axs[1, 1], trajectory.t, inside_atmosphere)
add_thrust_shading(axs[1, 1], trajectory.t, thrust_activities)

plt.tight_layout()
#endregion

# Data Output
local_minima_indices = argrelextrema(height, np.less)[0] # Find local minima in the height data

if local_minima_indices.size > 0:
    first_local_min_index = local_minima_indices[0]
    height_minima = height[first_local_min_index]
    print(f"First local minimum of height over Mars surface: {height_minima:.4f} km")
else:
    print("No local minima found in the height data.")

print(f"Propellant consumed was {m_0 - m[-1]:.4f} kg") # Propellant consumption 

plt.show()