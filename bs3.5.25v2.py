import random
import pulp as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# PARAMETERS
n_buses = 20
battery_capacity = 230  # kWh
charging_window = 12  # from 10pm to 4am (half-hour slots)
n_groups = 5
slot_duration = 0.5  # hours
max_rate_per_bus = 60  # kW
min_rate_per_bus = 20  # kW

# Toggle for continuous charging
CONTINUOUS_CHARGING = True  # Set to False to allow breaks

MAX_DEMAND = 700  # Set your desired fixed value here

# Random SOC between 25% and 40%
arrival_soc = [random.uniform(0.25, 0.4) for _ in range(n_buses)]
energy_needed = [(1 - soc) * battery_capacity for soc in arrival_soc]

# Group start times (in half-hour slots)
# group_start_slots = [i for i in range(n_groups)]
group_start_slots = [0, 3, 6, 9, 11] 

time_labels = [f"{10 + t//2}:{'00' if t % 2 == 0 else '30'} PM" if t < 4 else f"{(t-4)//2}:{'00' if t % 2 == 0 else '30'} AM" for t in range(charging_window)]

model = pl.LpProblem("Bus_Charging_Optimization", pl.LpMinimize)

# DECISION VARIABLES
x = pl.LpVariable.dicts("x", ((i, t) for i in range(n_buses) for t in range(charging_window)), cat="Binary")
g = pl.LpVariable.dicts("g", ((i, k) for i in range(n_buses) for k in range(n_groups)), cat="Binary")
r = pl.LpVariable.dicts("r", (k for k in range(n_groups)), lowBound=min_rate_per_bus, upBound=max_rate_per_bus)
p = pl.LpVariable.dicts("p", ((i, t) for i in range(n_buses) for t in range(charging_window)), lowBound=0)

# OBJECTIVE: Minimize total energy delivered
model += pl.lpSum(p[i, t] for i in range(n_buses) for t in range(charging_window))

# CONSTRAINT 1: Each bus is assigned to exactly one group
for i in range(n_buses):
    model += pl.lpSum(g[i, k] for k in range(n_groups)) == 1

# CONSTRAINT 2: Buses can only start charging from their group's start time
for i in range(n_buses):
    for t in range(charging_window):
        for k in range(n_groups):
            if t < group_start_slots[k]:
                model += x[i, t] <= 1 - g[i, k]

# CONSTRAINT 3: Link power to charging status and group rate
for i in range(n_buses):
    for t in range(charging_window):
        # Power must be zero if not charging
        model += p[i, t] <= max_rate_per_bus * x[i, t]
        # Power must respect group rate if assigned
        for k in range(n_groups):
            model += p[i, t] <= r[k] + max_rate_per_bus * (1 - g[i, k])

# CONSTRAINT 4: Energy requirement must be met
for i in range(n_buses):
    model += pl.lpSum(p[i, t] * slot_duration for t in range(charging_window) if CONTINUOUS_CHARGING or x[i, t].varValue == 1) >= energy_needed[i]

# CONSTRAINT 5: Hard maximum demand constraint
for t in range(charging_window):
    model += pl.lpSum(p[i, t] for i in range(n_buses)) <= MAX_DEMAND

# CONSTRAINT 6: Continuous charging (if enabled)
if CONTINUOUS_CHARGING:
    for i in range(n_buses):
        for t in range(charging_window - 1):
            model += x[i, t] >= x[i, t + 1]

# SOLVE
solver = pl.PULP_CBC_CMD(msg=True)
model.solve(solver)

# OUTPUTS
print(f"Status: {pl.LpStatus[model.status]}")

if pl.LpStatus[model.status] != "Optimal":
    print("Warning: Model did not find an optimal solution. Some variables may be undefined.")

print("\nCharging Rates per Group:")
for k in range(n_groups):
    print(f" Group {k+1}: {r[k].varValue:.2f} kW")

# --- Bus Assignment Table (without kW per slot) ---
print("\nBus Assignments (SoC Progression):")
header_soc = "{:<5} {:<6} {:<18} {:<15} {:<60}".format(
    "Bus", "Group", "First Slot", "Init SoC (%)", "SoC Progression (%)"
)
print(header_soc)
print("-" * len(header_soc))
for i in range(n_buses):
    assigned_groups = [k for k in range(n_groups) if g[i, k].varValue == 1]
    if assigned_groups:
        assigned_group = assigned_groups[0]
        group_str = str(assigned_group + 1)
    else:
        assigned_group = None
        group_str = "Unassigned"
    charging_slots = [t for t in range(charging_window) if x[i, t].varValue == 1]
    first_slot = charging_slots[0] if charging_slots else None
    socs = []
    soc = arrival_soc[i]
    for t in range(charging_window):
        power = p[i, t].varValue if p[i, t].varValue is not None else 0
        if x[i, t].varValue == 1:
            soc += (power * slot_duration) / battery_capacity
        socs.append(f"{soc*100:.1f}")
    print("{:<5} {:<6} {:<18} {:<15.1f} {:<60}".format(
        i+1, group_str, str(first_slot), arrival_soc[i]*100, " | ".join(socs)
    ))

# --- Power per Slot Table ---
print("\nBus Power per Slot (kW):")
header_kw = "{:<5} {:<6} {:<18} {:<60}".format(
    "Bus", "Group", "First Slot", "Power per Slot (kW)"
)
print(header_kw)
print("-" * len(header_kw))
for i in range(n_buses):
    assigned_groups = [k for k in range(n_groups) if g[i, k].varValue == 1]
    if assigned_groups:
        assigned_group = assigned_groups[0]
        group_str = str(assigned_group + 1)
    else:
        assigned_group = None
        group_str = "Unassigned"
    charging_slots = [t for t in range(charging_window) if x[i, t].varValue == 1]
    first_slot = charging_slots[0] if charging_slots else None
    powers = []
    for t in range(charging_window):
        if x[i, t].varValue == 1:
            power = p[i, t].varValue if p[i, t].varValue is not None else 0
            powers.append(f"{power:.1f}")
        else:
            powers.append("")  # Blank if not charging
    print("{:<5} {:<6} {:<18} {:<60}".format(
        i+1, group_str, str(first_slot), " | ".join(powers)
    ))

# Calculate total demand per slot
demand_per_slot = [sum(p[i, t].varValue for i in range(n_buses) if p[i, t].varValue is not None) for t in range(charging_window)]

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
plt.figure(figsize=(14, 8))
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
plt.show()

