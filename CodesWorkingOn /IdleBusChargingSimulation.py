# Idle bus simulation with 2-stage charging (CC and CV phases) for a 
# bus fleet from an actual charging data for the 12th of June 2025

import numpy as np
import matplotlib.pyplot as plt

n_buses = 1
battery_capacity = 230  # kWh
slot_duration = 1/60  # 1 minute in hours
start_soc = 0.539
cc_fraction = 0.8
max_kw = 53.74
avg_kw = 32.84
end_time = 3.5  # hours (from 10:00pm to 1:30am)
n_slots = int(end_time / slot_duration)

def simulate_2stage_charging(power_profile, battery_capacity, slot_duration, initial_soc=0.539, cc_fraction=0.8, 
                             lambda_min=0.01, lambda_max=5.0, max_iterations=100, tolerance=0.001):
    n_slots = len(power_profile)
    soc = initial_soc
    soc_list = [soc * 100]
    energy_cc = 0
    t_cc = 0

    # Stage 1: CC phase (charge at assigned power per slot until SoC reaches 80%)
    for t in range(n_slots):
        if soc < cc_fraction:
            energy = power_profile[t] * slot_duration
            soc += energy / battery_capacity
            soc = min(soc, cc_fraction)
            energy_cc += energy
            soc_list.append(soc * 100)
            t_cc = t + 1
        else:
            break

    # If battery is full in CC phase, fill rest with last value
    if soc >= 1.0 or t_cc >= n_slots:
        soc_list += [soc_list[-1]] * (n_slots - len(soc_list) + 1)
        return soc_list[:n_slots+1], power_profile, t_cc, None

    # Stage 2: CV phase (exponential decay)
    energy_needed_cv = battery_capacity * (1.0 - soc)
    cc_power = power_profile[t_cc-1] if t_cc > 0 else power_profile[0]
    total_cv_time = (n_slots - t_cc) * slot_duration

    # Bisection method to find lambda
    lambda_low = lambda_min
    lambda_high = lambda_max
    lambda_current = (lambda_low + lambda_high) / 2
    iteration = 0

    while iteration < max_iterations:
        cv_energies = []
        for i in range(n_slots - t_cc):
            t_hr = i * slot_duration
            power = cc_power * np.exp(-lambda_current * t_hr)
            cv_energies.append(power * slot_duration)
        total_cv_energy = sum(cv_energies)
        diff = total_cv_energy - energy_needed_cv

        if abs(diff) <= tolerance:
            break
        elif diff > 0:
            lambda_low = lambda_current
        else:
            lambda_high = lambda_current
        lambda_current = (lambda_low + lambda_high) / 2
        iteration += 1

    # Build full power and soc profile
    power_list = list(power_profile[:t_cc])
    soc_now = soc
    for i in range(n_slots - t_cc):
        t_hr = i * slot_duration
        power = cc_power * np.exp(-lambda_current * t_hr)
        energy = power * slot_duration
        soc_now += energy / battery_capacity
        soc_now = min(soc_now, 1.0)
        power_list.append(power)
        soc_list.append(soc_now * 100)

    # Pad if needed
    if len(soc_list) < n_slots + 1:
        soc_list += [soc_list[-1]] * (n_slots + 1 - len(soc_list))
    if len(power_list) < n_slots:
        power_list += [0] * (n_slots - len(power_list))

    return soc_list[:n_slots+1], power_list, t_cc, lambda_current

# Build power profile: CC phase at max_kw, then CV phase (tapering)
cc_minutes = int(((cc_fraction - start_soc) * battery_capacity) / max_kw * 60)
power_profile = [max_kw] * cc_minutes + [avg_kw] * (n_slots - cc_minutes)

soc_list, power_list, t_cc, lambda_current = simulate_2stage_charging(
    power_profile, battery_capacity, slot_duration, initial_soc=start_soc, cc_fraction=cc_fraction
)

# Time axis for plotting
time_axis = np.arange(n_slots + 1) * slot_duration * 60  # in minutes from 10:00pm

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis, soc_list, label='SoC (%)')
plt.ylabel('State of Charge (%)')
plt.title('Idle Bus Charging Simulation (10:00pm Start, Full by 1:30am)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(time_axis[:-1], power_list, label='Charging Power (kW)', color='orange')
plt.xlabel('Time (minutes from 10:00pm)')
plt.ylabel('Charging Power (kW)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



