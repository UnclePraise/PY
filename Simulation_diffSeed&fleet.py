# This script simulates the optimization of bus charging schedules with varying fleet sizes and random seeds.

import random
import pulp as pl
import matplotlib.pyplot as plt
import numpy as np

bus_sizes = [10, 20, 30, 50, 80, 130, 40, 60, 70, 90, 100, 110, 120]  # Different fleet sizes to test
all_max_demand_per_bus = []

seeds_to_test = random.sample(range(1, 1000000), 10)  # 10 unique random seeds each run

for n_buses in bus_sizes:
    max_demands = []
    for seed in seeds_to_test:
        random.seed(seed)
        battery_capacity = 230  # kWh
        charging_window = 12  # from 10pm to 4am (half-hour slots)
        slot_duration = 0.5  # hours
        max_rate_per_bus = 60  # kW
        min_rate_per_bus = 20  # kW
        CONTINUOUS_CHARGING = True

        lower_bound = 0.25
        mid_bound = 0.30
        upper_bound = 0.35

        arrival_soc = []
        for _ in range(n_buses):
            if random.random() < 0.6:
                arrival_soc.append(random.uniform(lower_bound, mid_bound))
            else:
                arrival_soc.append(random.uniform(mid_bound, upper_bound))

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
            max_demands.append(optimum_max_demand / n_buses)  # Demand per bus

    all_max_demand_per_bus.append(max_demands)

# Scatter plot
plt.figure(figsize=(8, 5))
for idx, n_buses in enumerate(bus_sizes):
    y = all_max_demand_per_bus[idx]
    x = [n_buses] * len(y)
    plt.scatter(x, y, s=20, alpha=0.7, label=f"{n_buses} buses" if idx == 0 else "")  # s=20 for smaller dots
    # Annotate each point
    for xi, yi in zip(x, y):
        plt.text(xi, yi, f"{yi:.2f}", ha='center', va='bottom', fontsize=6, color='black')

plt.xlabel("Number of Buses")
plt.ylabel("Optimum Maximum Demand Charge per Bus (kW)")
plt.title("Distribution of Maximum Demand Charge per Bus vs. Fleet Size")
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Box plot
plt.figure(figsize=(8, 5))
plt.boxplot(all_max_demand_per_bus, positions=bus_sizes, widths=4, patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='blue'),
            medianprops=dict(color='red', linewidth=2))
plt.xlabel("Number of Buses")
plt.ylabel("Optimum Maximum Demand Charge per Bus (kW)")
plt.title("Box Plot:Optimum Maximum Demand Charge per Bus vs. Fleet Size")
plt.xticks(bus_sizes)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
