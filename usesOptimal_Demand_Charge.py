#This code is a simulation of a bus charging optimization problem using linear programming that helps the user 
# find the optimal charging schedule for a fleet of buses while minimizing peak demand and ensuring that all buses are 
# charged within a specified time window. 
# while also figuring out the optimum demand charge capacity. try multiple seeds for randomization

import random
import pulp as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# PARAMETERS
n_buses = 10
battery_capacity = 230  # kWh
charging_window = 12  # from 10pm to 4am (half-hour slots)
slot_duration = 0.5  # hours
max_rate_per_bus = 60  # kW
min_rate_per_bus = 20  # kW

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
    model += pl.lpSum(p[i, t] * slot_duration for t in range(charging_window)) >= energy_needed[i]

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
        # Prevent SoC from exceeding 1.0 (100%)
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

# --- Bus Assignment Table (without kW per slot) ---
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

# Calculate SoC matrix
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
# Optional: show legend for first 20 buses only to avoid clutter
if n_buses <= 20:
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
# plt.show()

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

# --- Calculate SoC matrix for all buses using 2-stage charging ---
soc_matrix_2stage = np.zeros((n_buses, charging_window+1))
for i in range(n_buses):
    power_profile = [p[i, t].varValue if p[i, t].varValue is not None else 0 for t in range(charging_window)]
    soc_list, _, _, _ = simulate_2stage_charging(
        power_profile, battery_capacity, slot_duration, initial_soc=arrival_soc[i]
    )
    soc_matrix_2stage[i, :] = soc_list

# --- Plot SoC progression for all buses (2-stage charging) ---
plt.figure(figsize=(12, 6))
for i in range(n_buses):
    plt.plot(range(charging_window+1), soc_matrix_2stage[i], marker='o', label=f'Bus {i+1}' if n_buses <= 10 else None)
plt.xlabel("Time Slot")
plt.ylabel("State of Charge (%)")
plt.title("SoC Progression Over Time for Each Bus (2-Stage Charging)")
plt.xticks(range(charging_window+1), time_labels + ["End"], rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)
if n_buses <= 10:
    plt.legend()
plt.tight_layout()
plt.show()

# Plot SoC progression for each bus separately (2-stage charging)
# for i in range(n_buses):
#     plt.figure(figsize=(8, 4))
#     plt.plot(range(charging_window+1), soc_matrix_2stage[i], marker='o', color='tab:blue')
#     plt.xlabel("Time Slot")
#     plt.ylabel("State of Charge (%)")
#     plt.title(f"SoC Progression for Bus {i+1} (2-Stage Charging)")
#     plt.xticks(range(charging_window+1), time_labels + ["End"], rotation=45)
#     plt.ylim(0, 105)
#     plt.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.show()
