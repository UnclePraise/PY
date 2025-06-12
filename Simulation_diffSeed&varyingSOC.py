# Simulation_diffSeed%26varyingSOC.py

import random
import pulp as pl
import matplotlib.pyplot as plt
import numpy as np

# Define SoC ranges as (lower, upper) tuples
soc_ranges = [(0.20, 0.25), (0.20, 0.30), (0.25, 0.35), (0.25, 0.40), (0.05, 0.35)]
range_labels = [ "20-25%", "20-30%", "25-35%" ,"25-40%" , "5-35%"]  # Labels for the x-axis

n_buses = 10
battery_capacity = 230  # kWh
charging_window = 12
slot_duration = 0.5
max_rate_per_bus = 60
min_rate_per_bus = 20
CONTINUOUS_CHARGING = True

seeds_to_test = random.sample(range(1, 1000000), 10)  # 10 unique random seeds each run

all_max_demand = []

for soc_range in soc_ranges:
    lower_bound, upper_bound = soc_range
    max_demands = []
    for seed in seeds_to_test:
        random.seed(seed)
        arrival_soc = [random.uniform(lower_bound, upper_bound) for _ in range(n_buses)]
        energy_needed = [(1 - soc) * battery_capacity for soc in arrival_soc]

        model = pl.LpProblem("Bus_Charging_Optimization", pl.LpMinimize)
        x = pl.LpVariable.dicts("x", ((i, t) for i in range(n_buses) for t in range(charging_window)), cat="Binary")
        p = pl.LpVariable.dicts("p", ((i, t) for i in range(n_buses) for t in range(charging_window)), lowBound=0)
        peak_demand = pl.LpVariable("peak_demand", lowBound=0)
        model += peak_demand

        for i in range(n_buses):
            for t in range(charging_window):
                model += p[i, t] <= max_rate_per_bus * x[i, t]
                model += p[i, t] >= min_rate_per_bus * x[i, t]
        for i in range(n_buses):
            model += pl.lpSum(p[i, t] * slot_duration for t in range(charging_window)) >= energy_needed[i]
        for t in range(charging_window):
            model += pl.lpSum(p[i, t] for i in range(n_buses)) <= peak_demand
        if CONTINUOUS_CHARGING:
            for i in range(n_buses):
                for t in range(charging_window - 1):
                    model += x[i, t] >= x[i, t + 1]

        solver = pl.PULP_CBC_CMD(msg=False)
        model.solve(solver)

        if pl.LpStatus[model.status] == "Optimal":
            demand_per_slot = [sum(p[i, t].varValue for i in range(n_buses) if p[i, t].varValue is not None) for t in range(charging_window)]
            optimum_max_demand = max(demand_per_slot)
            max_demands.append(optimum_max_demand / n_buses)  # Per bus

    all_max_demand.append(max_demands)

# Scatter plot (per bus)
plt.figure(figsize=(8, 5))
for idx, label in enumerate(range_labels):
    y = all_max_demand[idx]
    x = [label] * len(y)
    plt.scatter(x, y, s=20, alpha=0.7, label=label)  # <-- s=20 for smaller dots
    for xi, yi in zip(x, y):
        plt.text(xi, yi + 0.5, f"{yi:.2f}", ha='center', va='bottom', fontsize=9, color='black')
plt.xlabel("Arrival SoC Range")
plt.ylabel("Maximum Demand Charge per Bus (kW)")
plt.title("Max Demand Charge per Bus for Different Arrival SoC Ranges")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Box plot (per bus)
plt.figure(figsize=(8, 5))
plt.boxplot(all_max_demand, positions=range(len(range_labels)), widths=0.5, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2))
plt.xlabel("Arrival SoC Range")
plt.ylabel("Maximum Demand Charge per Bus (kW)")
plt.title("Box Plot: Max Demand Charge per Bus for Different Arrival SoC Ranges")
plt.xticks(range(len(range_labels)), range_labels)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
