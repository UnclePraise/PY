#This code is a simulation of a bus charging optimization problem using linear programming that helps the user 
# find the optimal charging schedule for a fleet of buses while minimizing peak demand and ensuring that all buses are 
# charged within a specified time window. 
# while also figuring out the optimum demand charge capacity. try multiple seeds for randomization

import random
import pulp as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math

# PARAMETERS
n_buses = 10
battery_capacity = 230  # kWh
charging_window = 12  # from 10pm to 4am (half-hour slots)
slot_duration = 0.5  # hours
max_rate_per_bus = 60  # kW
min_rate_per_bus = 30  # kW

CONTINUOUS_CHARGING = True
MAX_DEMAND = 700

arrival_soc = [random.uniform(0.25, 0.4) for _ in range(n_buses)]
energy_needed = [(1 - soc) * battery_capacity for soc in arrival_soc]

time_labels = [f"{10 + t//2}:{'00' if t % 2 == 0 else '30'} PM" if t < 4 else f"{(t-4)//2}:{'00' if t % 2 == 0 else '30'} AM" for t in range(charging_window)]

model = pl.LpProblem("Bus_Charging_Optimization", pl.LpMinimize)

# DECISION VARIABLES
x = pl.LpVariable.dicts("x", ((i, t) for i in range(n_buses) for t in range(charging_window)), cat="Binary")
p = pl.LpVariable.dicts("p", ((i, t) for i in range(n_buses) for t in range(charging_window)), lowBound=0)
peak_demand = pl.LpVariable("peak_demand", lowBound=0)

# OBJECTIVE: Minimize the peak demand
model += peak_demand

# CONSTRAINT 1: Link power to charging status
for i in range(n_buses):
    for t in range(charging_window):
        model += p[i, t] <= max_rate_per_bus * x[i, t]
        model += p[i, t] >= min_rate_per_bus * x[i, t]

# CONSTRAINT 2: Energy requirement must be met
for i in range(n_buses):
    model += pl.lpSum(p[i, t] * slot_duration for t in range(charging_window)) == (1 - arrival_soc[i]) * battery_capacity

# CONSTRAINT 3: Peak demand constraint for each slot
for t in range(charging_window):
    model += pl.lpSum(p[i, t] for i in range(n_buses)) <= peak_demand

# CONSTRAINT 4: Continuous charging (if enabled)
if CONTINUOUS_CHARGING:
    for i in range(n_buses):
        for t in range(charging_window - 1):
            model += x[i, t] >= x[i, t + 1]

# CONSTRAINT 5: Prevent SoC from exceeding 100%
for i in range(n_buses):
    soc = arrival_soc[i]
    for t in range(charging_window):
        if t == 0:
            soc_expr = arrival_soc[i] + (p[i, t] * slot_duration) / battery_capacity
        else:
            soc_expr = soc + (p[i, t] * slot_duration) / battery_capacity
        model += soc_expr <= 1.0
        soc = soc_expr

# SOLVE
solver = pl.PULP_CBC_CMD(msg=True)
model.solve(solver)

# OUTPUTS
print(f"Status: {pl.LpStatus[model.status]}")

if pl.LpStatus[model.status] != "Optimal":
    print("Warning: Model did not find an optimal solution. Some variables may be undefined.")

# Calculate total demand per slot
demand_per_slot = [sum(p[i, t].varValue for i in range(n_buses) if p[i, t].varValue is not None) for t in range(charging_window)]

# Print the optimum max demand charge
optimum_max_demand = max(demand_per_slot)
print(f"\nOptimum Maximum Demand Charge (kW): {optimum_max_demand:.2f}")
print(f"Optimized max_rate_per_bus: {max_rate_per_bus} kW")
print(f"Optimized min_rate_per_bus: {min_rate_per_bus} kW")

# --- Check if all buses reach 100% SoC ---
all_full = True
for i in range(n_buses):
    soc = arrival_soc[i]
    for t in range(charging_window):
        power = p[i, t].varValue if p[i, t].varValue is not None else 0
        if x[i, t].varValue == 1:
            soc += (power * slot_duration) / battery_capacity
    if soc < 1.0 - 1e-4:  # Allowing for floating point tolerance
        all_full = False
        break

if not all_full:
    print("\nNot all buses reach 100% SoC. Calculating minimum max_rate_per_bus and min_rate_per_bus needed...")
    # Try increasing max_rate_per_bus until all buses can reach 100%
    test_max = max_rate_per_bus
    test_min = min_rate_per_bus
    found = False
    while not found and test_max <= 200:  # Set a reasonable upper bound
        model_tmp = pl.LpProblem("Bus_Charging_Optimization_tmp", pl.LpMinimize)
        x_tmp = pl.LpVariable.dicts("x", ((i, t) for i in range(n_buses) for t in range(charging_window)), cat="Binary")
        p_tmp = pl.LpVariable.dicts("p", ((i, t) for i in range(n_buses) for t in range(charging_window)), lowBound=0)
        peak_demand_tmp = pl.LpVariable("peak_demand", lowBound=0)
        model_tmp += peak_demand_tmp
        for i in range(n_buses):
            for t in range(charging_window):
                model_tmp += p_tmp[i, t] <= test_max * x_tmp[i, t]
                model_tmp += p_tmp[i, t] >= test_min * x_tmp[i, t]
        for i in range(n_buses):
            model_tmp += pl.lpSum(p_tmp[i, t] * slot_duration for t in range(charging_window)) == (1 - arrival_soc[i]) * battery_capacity
        for t in range(charging_window):
            model_tmp += pl.lpSum(p_tmp[i, t] for i in range(n_buses)) <= peak_demand_tmp
        if CONTINUOUS_CHARGING:
            for i in range(n_buses):
                for t in range(charging_window - 1):
                    model_tmp += x_tmp[i, t] >= x_tmp[i, t + 1]
        solver_tmp = pl.PULP_CBC_CMD(msg=False)
        model_tmp.solve(solver_tmp)
        if pl.LpStatus[model_tmp.status] == "Optimal":
            # Check if all buses reach 100%
            all_full_tmp = True
            for i in range(n_buses):
                soc = arrival_soc[i]
                for t in range(charging_window):
                    power = p_tmp[i, t].varValue if p_tmp[i, t].varValue is not None else 0
                    if x_tmp[i, t].varValue == 1:
                        soc += (power * slot_duration) / battery_capacity
                if soc < 1.0 - 1e-4:
                    all_full_tmp = False
                    break
            if all_full_tmp:
                found = True
                print(f"Minimum max_rate_per_bus required: {test_max} kW")
                print(f"Minimum min_rate_per_bus required: {test_min} kW")
                break
        test_max += 5  # Increment and try again
    if not found:
        print("Could not find feasible max_rate_per_bus up to 200 kW. Try increasing charging window or reducing bus energy needs.")

# --- Bus Assignment Table (linear SoC progression) ---
print("\nBus Assignments (SoC Progression):")
header_soc = "{:<5} {:<18} {:<15} {:<60}".format(
    "Bus", "First Slot", "Init SoC (%)", "SoC Progression (%)"
)
print(header_soc)
print("-" * len(header_soc))
for i in range(n_buses):
    charging_slots = [t for t in range(charging_window) if x[i, t].varValue == 1]
    first_slot = charging_slots[0] if charging_slots else None
    socs = []
    soc = arrival_soc[i]
    for t in range(charging_window):
        power = p[i, t].varValue if p[i, t].varValue is not None else 0
        if x[i, t].varValue == 1:
            soc += (power * slot_duration) / battery_capacity
        socs.append(f"{soc*100:.1f}")
    print("{:<5} {:<18} {:<15.1f} {:<60}".format(
        i+1, str(first_slot), arrival_soc[i]*100, " | ".join(socs)
    ))

# --- 2-Stage Charging Simulation Function ---
def simulate_2stage_charging(power_profile, battery_capacity, slot_duration, initial_soc=0.25, cc_fraction=0.8, 
                             lambda_min=0.1, lambda_max=20.0, max_iterations=100, tolerance=0.005):
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
    power_list = power_profile[:t_cc]
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

# --- 2-Stage Charging Simulation Function (1-min resolution) ---
def simulate_2stage_charging_minute(power_profile, battery_capacity, slot_duration, initial_soc=0.25, cc_fraction=0.8, 
                                    lambda_min=0.1, lambda_max=20.0, max_iterations=100, tolerance=0.005):
    n_slots = len(power_profile)
    total_minutes = int(n_slots * slot_duration * 60)
    slot_times = np.arange(0, n_slots * slot_duration, slot_duration) * 60  # in minutes
    minute_times = np.arange(0, total_minutes)
    power_profile_min = np.interp(minute_times, slot_times, power_profile)
    minute_duration = 1 / 60  # 1 minute in hours

    soc = initial_soc
    soc_list = [soc * 100]
    energy_cc = 0
    t_cc = 0

    # Stage 1: CC phase (charge at assigned power per slot until SoC reaches 80%)
    for t in range(total_minutes):
        if soc < cc_fraction:
            energy = power_profile_min[t] * minute_duration
            soc += energy / battery_capacity
            soc = min(soc, cc_fraction)
            energy_cc += energy
            soc_list.append(soc * 100)
            t_cc = t + 1
        else:
            break

    # If battery is full in CC phase, fill rest with last value
    if soc >= 1.0 or t_cc >= total_minutes:
        soc_list += [soc_list[-1]] * (total_minutes - len(soc_list) + 1)
        return soc_list[:total_minutes+1], power_profile_min, t_cc, None

    # Stage 2: CV phase (exponential decay)
    energy_needed_cv = battery_capacity * (1.0 - soc)
    cc_power = power_profile_min[t_cc-1] if t_cc > 0 else power_profile_min[0]
    total_cv_time = (total_minutes - t_cc) * minute_duration

    # Bisection method to find lambda
    lambda_low = lambda_min
    lambda_high = lambda_max
    lambda_current = (lambda_low + lambda_high) / 2
    iteration = 0

    while iteration < max_iterations:
        cv_energies = []
        for i in range(total_minutes - t_cc):
            t_hr = i * minute_duration
            power = cc_power * np.exp(-lambda_current * t_hr)
            cv_energies.append(power * minute_duration)
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
    power_list = list(power_profile_min[:t_cc])
    soc_now = soc
    for i in range(total_minutes - t_cc):
        t_hr = i * minute_duration
        power = cc_power * np.exp(-lambda_current * t_hr)
        energy = power * minute_duration
        soc_now += energy / battery_capacity
        soc_now = min(soc_now, 1.0)
        power_list.append(power)
        soc_list.append(soc_now * 100)

    # Pad if needed
    if len(soc_list) < total_minutes + 1:
        soc_list += [soc_list[-1]] * (total_minutes + 1 - len(soc_list))
    if len(power_list) < total_minutes:
        power_list += [0] * (total_minutes - len(power_list))

    return soc_list[:total_minutes+1], power_list, t_cc, lambda_current

# --- Bus Assignments (SoC Progression, 2-Stage Simulation) ---
print("\nBus Assignments (SoC Progression, 2-Stage Simulation):")
header_soc = "{:<5} {:<18} {:<15} {:<60}".format(
    "Bus", "First Slot", "Init SoC (%)", "SoC Progression (%)"
)
print(header_soc)
print("-" * len(header_soc))
soc_matrix_2stage = np.zeros((n_buses, charging_window+1))
for i in range(n_buses):
    power_profile = [p[i, t].varValue if p[i, t].varValue is not None else 0 for t in range(charging_window)]
    soc_list, _, _, _ = simulate_2stage_charging(
        power_profile, battery_capacity, slot_duration, initial_soc=arrival_soc[i]
    )
    soc_matrix_2stage[i, :] = soc_list
    charging_slots = [t for t in range(charging_window) if x[i, t].varValue == 1]
    first_slot = charging_slots[0] if charging_slots else None
    socs = [f"{soc_matrix_2stage[i][t]:.1f}" for t in range(charging_window+1)]
    print("{:<5} {:<18} {:<15.1f} {:<60}".format(
        i+1, str(first_slot), arrival_soc[i]*100, " | ".join(socs)
    ))

# --- Power per Slot Table ---
print("\nBus Power per Slot (kW):")
header_kw = "{:<5} {:<18} {:<60}".format(
    "Bus", "First Slot", "Power per Slot (kW)"
)
print(header_kw)
print("-" * len(header_kw))
for i in range(n_buses):
    charging_slots = [t for t in range(charging_window) if x[i, t].varValue == 1]
    first_slot = charging_slots[0] if charging_slots else None
    powers = []
    for t in range(charging_window):
        if x[i, t].varValue == 1:
            power = p[i, t].varValue if p[i, t].varValue is not None else 0
            powers.append(f"{power:.1f}")
        else:
            powers.append("")  # Blank if not charging
    print("{:<5} {:<18} {:<60}".format(
        i+1, str(first_slot), " | ".join(powers)
    ))

# Add total kW per slot as the last row
total_per_slot = [sum(p[i, t].varValue for i in range(n_buses) if p[i, t].varValue is not None) for t in range(charging_window)]
total_str = " | ".join(f"{val:.1f}" for val in total_per_slot)
print("{:<5} {:<18} {:<60}".format("Total", "", total_str))

plt.figure(figsize=(12, 4))
plt.bar(range(charging_window), demand_per_slot, color='skyblue', alpha=0.7)
plt.plot(range(charging_window), demand_per_slot, marker='o', color='b', label='Total Demand (kW)')
plt.axhline(MAX_DEMAND, color='r', linestyle='--', label=f'Max Demand = {MAX_DEMAND:.1f} kW')
plt.xticks(range(charging_window), time_labels, rotation=45)
plt.xlabel("Time Slot")
plt.ylabel("Power (kW)")
plt.title("Power Demand Profile (kW per Time Slot)")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Annotate total demand value on each bar
for t, val in enumerate(demand_per_slot):
    plt.text(t, val + 5, f"{val:.1f}", ha='center', va='bottom', fontsize=9, color='black')

# Calculate SoC matrix (linear, for Gantt)
soc_matrix = np.zeros((n_buses, charging_window))
for i in range(n_buses):
    soc = arrival_soc[i]
    for t in range(charging_window):
        power = p[i, t].varValue if p[i, t].varValue is not None else 0
        if x[i, t].varValue == 1:
            soc += (power * slot_duration) / battery_capacity
        soc_matrix[i][t] = soc * 100  # Convert to percentage

# Plot Gantt Chart for Bus Charging Schedule
bus_colors = plt.cm.tab20(np.linspace(0, 1, n_buses))
plt.figure(figsize=(14,8))
for i in range(n_buses):
    charging_slots = [t for t in range(charging_window) if x[i, t].varValue == 1]
    if charging_slots:
        plt.barh(i, len(charging_slots), left=min(charging_slots), color=bus_colors[i], alpha=0.8)
        for t in charging_slots:
            plt.text(t + 0.1, i, f"{int(soc_matrix[i][t])}%", va='center', fontsize=7, color='white')
plt.yticks(range(n_buses), [f'Bus {i+1}' for i in range(n_buses)])
plt.xticks(range(charging_window), time_labels, rotation=45)
plt.xlabel("Time Slot")
plt.ylabel("Bus")
plt.title("Gantt Chart: Bus Charging Schedule")
plt.grid(True, axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()

# Count number of buses charging in each time slot
buses_per_slot = [sum(1 for i in range(n_buses) if x[i, t].varValue == 1) for t in range(charging_window)]

plt.figure(figsize=(12, 4))
plt.bar(range(charging_window), buses_per_slot, color='orange', alpha=0.7)
plt.plot(range(charging_window), buses_per_slot, marker='o', color='red', label='Buses Charging')
plt.xticks(range(charging_window), time_labels, rotation=45)
plt.xlabel("Time Slot")
plt.ylabel("Number of Buses Charging")
plt.title("Number of Buses Charging per Time Slot")
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()

# Stacked bar plot: Each bus's presence in each slot, colored as in the Gantt chart
charging_matrix = np.array([[1 if x[i, t].varValue == 1 else 0 for t in range(charging_window)] for i in range(n_buses)])

plt.figure(figsize=(12, 5))
bottom = np.zeros(charging_window)
for i in range(n_buses):
    plt.bar(range(charging_window), charging_matrix[i], bottom=bottom, color=bus_colors[i], edgecolor='none', label=f'Bus {i+1}' if i < 20 else None)
    bottom += charging_matrix[i]

plt.xticks(range(charging_window), time_labels, rotation=45)
plt.xlabel("Time Slot")
plt.ylabel("Number of Buses Charging")
plt.title("Number of Buses Charging per Time Slot (by Bus)")
plt.grid(True, linestyle='--', alpha=0.7)
if n_buses <= 20:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()

# --- Calculate SoC matrix for all buses using 2-stage charging (1-min resolution) ---
soc_matrix_2stage_min = []
for i in range(n_buses):
    power_profile = [p[i, t].varValue if p[i, t].varValue is not None else 0 for t in range(charging_window)]
    soc_list, _, _, _ = simulate_2stage_charging_minute(
        power_profile, battery_capacity, slot_duration, initial_soc=arrival_soc[i]
    )
    soc_matrix_2stage_min.append(soc_list)

# --- Plot SoC progression for all buses (2-stage charging, 1-min resolution) ---
plt.figure(figsize=(14, 7))
total_minutes = int(charging_window * slot_duration * 60)
minute_ticks = np.arange(0, total_minutes+1, 30)  # every 30 minutes
minute_labels = []
for m in minute_ticks:
    hr = 22 + m // 60
    mn = m % 60
    if hr >= 24:
        hr -= 24
    label = f"{hr:02d}:{mn:02d}"
    minute_labels.append(label)

for i, soc_list in enumerate(soc_matrix_2stage_min):
    plt.plot(np.arange(len(soc_list)), soc_list, label=f'Bus {i+1}' if n_buses <= 10 else None)
plt.xlabel("Time (minutes from 10:00pm)")
plt.ylabel("State of Charge (%)")
plt.title("SoC Progression Over Time for Each Bus (2-Stage Charging, 1-min Resolution)")
plt.xticks(minute_ticks, minute_labels, rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
if n_buses <= 10:
    plt.legend()
plt.tight_layout()
plt.show()

# --- Plot SoC progression for each bus individually (2-stage charging, 1-min resolution) ---
n_per_fig = 5  # Number of buses per figure
n_figs = math.ceil(n_buses / n_per_fig)

for fig_idx in range(n_figs):
    start = fig_idx * n_per_fig
    end = min((fig_idx + 1) * n_per_fig, n_buses)
    ncols = 1
    nrows = end - start
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, nrows * 3), sharex=True, sharey=True)
    if nrows == 1:
        axes = [axes]
    for idx, i in enumerate(range(start, end)):
        soc_list = soc_matrix_2stage_min[i]
        ax = axes[idx]
        ax.plot(np.arange(len(soc_list)), soc_list, label=f'Bus {i+1}')
        ax.set_xlabel("Time (minutes from 10:00pm)")
        ax.set_ylabel("State of Charge (%)")
        ax.set_title(f"Bus {i+1}")
        ax.set_xticks(minute_ticks)
        ax.set_xticklabels(minute_labels, rotation=45)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
    plt.tight_layout()
    plt.suptitle(f"SoC Progression Over Time (Buses {start+1}-{end})", y=1.02, fontsize=14)
    plt.show()

