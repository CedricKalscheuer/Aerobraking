import numpy as np
import matplotlib.pyplot as plt

# MAVEN Aerobraking Data
altitude_km = np.array([125, 130, 135, 140, 145, 150, 155, 160, 165, 170])
density_kg_km3 = np.array([3.5, 2.8, 2.2, 1.7, 1.3, 1.0, 0.8, 0.6, 0.4, 0.2])

# Extracted data from the provided plot for MRO
altitude_km_mro_outbound = np.array([150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100, 95])
density_kg_km3_mro_outbound = np.array([0.004, 0.012, 0.025, 0.055, 0.12, 0.23, 0.55, 1.1, 2.2, 3.3, 3.9, 4.6])

# Extracted data from the Mars Odyssey plot
altitude_km_ody_inbound = np.array([150, 145, 140, 135, 130, 125, 120, 115, 110, 105, 100])
density_kg_km3_ody_inbound = np.array([0.004, 0.015, 0.03, 0.06, 0.13, 0.25, 0.6, 1.2, 2.5, 3.6, 4.8])

# Extracted data from the Mars Global Surveyor plot for inbound and outbound
altitude_km_mgs_inbound = np.array([180, 175, 170, 165, 160, 155, 150, 145, 140, 135, 130, 125, 120, 115, 110])
density_kg_km3_mgs_inbound = np.array([0.001, 0.002, 0.005, 0.01, 0.015, 0.03, 0.05, 0.1, 0.2, 0.4, 0.7, 1.2, 1.8, 2.5, 3.0])

# Curiosity EDL Data (converted to kg/km^3)
curiosity_altitude_km = np.array([134, 129, 124, 119, 114, 109, 104, 99, 94, 89, 84, 79, 74, 69, 64, 59, 54, 49, 44, 39, 34, 29, 24, 19, 14])
curiosity_density_kg_m3 = np.array([5.689E-10, 1.326E-09, 3.418E-09, 5.950E-09, 9.010E-09, 1.376E-08, 2.086E-08, 3.118E-08, 4.517E-08, 6.338E-08, 8.646E-08, 1.151E-07, 1.498E-07, 1.912E-07, 2.399E-07, 2.964E-07, 3.615E-07, 4.359E-07, 5.204E-07, 6.157E-07, 7.228E-07, 8.424E-07, 9.756E-07, 1.123E-06, 1.285E-06])
curiosity_density_kg_km3 = curiosity_density_kg_m3 * 1e9

#TGO Data
altitude_km_tgo = np.array([102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121])
density_MG05_values = np.array([1.55802716e-11, 1.41028473e-11, 1.27804117e-11, 1.17645745e-11, 1.10564580e-11, 9.90207834e-12, 9.23533360e-12,8.13572699e-12, 6.84934833e-12, 5.97437350e-12, 5.06235285e-12,4.36777848e-12, 3.77485569e-12, 3.28938293e-12, 2.84045885e-12,2.45823520e-12, 2.15014638e-12, 1.87285213e-12, 1.64198303e-12,1.47364671e-12])
density_MG05_values = density_MG05_values * 1e12
density_TGO_values = np.array([3.12074810e-11, 2.83604659e-11, 2.52163036e-11, 2.19243383e-11, 1.94875012e-11, 1.78881175e-11, 1.70543330e-11,1.52521588e-11, 1.26247052e-11, 1.06819664e-11, 8.89636146e-12,7.39758874e-12, 6.20737891e-12, 5.16844809e-12, 4.15535870e-12,3.49807583e-12, 2.93303416e-12, 2.40532060e-12, 2.01875187e-12,1.72217821e-12])
density_TGO_values = density_TGO_values * 1e12

#Exponential model
altitude = np.linspace(0, 200, 1000)
rho_0 = 0.06*1e9  # Density at 0 km
rho_200 = 0.000095  # Density at 195 km
H = 195 / np.log(rho_0 / rho_200)
density = rho_0 * np.exp(-altitude / H)

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the Mars Global Surveyor aerobraking data for inbound in orange
plt.plot(density_kg_km3_mgs_inbound, altitude_km_mgs_inbound, marker='s', linestyle='-', color='orange', label='Mars Global Surveyor Aerobraking Data 1998')

# Plot the Mars Odyssey aerobraking data for inbound in green
plt.plot(density_kg_km3_ody_inbound, altitude_km_ody_inbound, marker='^', linestyle='-', color='green', label='Mars Odyssey Aerobraking Data 2001')

# Plot the MRO aerobraking data for outbound in blue
plt.plot(density_kg_km3_mro_outbound, altitude_km_mro_outbound, marker='x', linestyle='-', color='purple', label='MRO Aerobraking Data 2006')

# Plot the Curiosity EDL data in black
plt.plot(curiosity_density_kg_km3, curiosity_altitude_km, marker='*', linestyle='-', color='black', label='Curiosity EDL Data 2012')

plt.plot(density_TGO_values, altitude_km_tgo, marker='o', linestyle='-', color='salmon', label='TGO Aerobraking Data 2017')
plt.plot(density_MG05_values, altitude_km_tgo, marker='o', linestyle='-', color='red', label='Marsgram Values for TGO Altitudes 2017')

# Plot the MAVEN Aerobraking data
plt.plot(density_kg_km3, altitude_km, marker='o', linestyle='-', label='MAVEN Aerobraking Data 2019')

#Plot the Exponential Model
plt.plot(density, altitude, color='blue', label = "Exponential Atmospheric Density Model")

plt.xlabel('Atmospheric Density (kg/km^3)')
plt.ylabel('Altitude (km)')
plt.xscale('log')
plt.title('Atmospheric Density over Height for Martian Atmosphere')
plt.grid(True)
plt.legend()
plt.show()
