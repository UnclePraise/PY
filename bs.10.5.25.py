#This code is a simulation of a bus charging optimization problem using linear programming that helps the user 
# find the optimal charging schedule for a fleet of buses while minimizing peak demand and ensuring that all buses are 
# charged within a specified time window. 
# while also figuring out the optimum demand charge capacity. try multiple seeds for randomization


import random
import pulp as pl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import norm

# seeds_to_test = [1, 42, 123, 2024, 9999]  # You can add more seeds if you like
seeds_to_test = random.sample(range(1, 1000000), 10)  # 10 unique random seeds each run

for seed in seeds_to_test:
    print(f"\n=== Running simulation with random seed: {seed} ===")
    random.seed(seed)

    # PARAMETERS
    n_buses = 1
    battery_capacity = 230  # kWh
    charging_window = 12  # from 10pm to 4am (half-hour slots)
    slot_duration = 0.5  # hours
    max_rate_per_bus = 60  # kW
    min_rate_per_bus = 20  # kW

    CONTINUOUS_CHARGING = True
    MAX_DEMAND = 700

    lower_bound = 0.25
    mid_bound = 0.30
    upper_bound = 0.35

    arrival_soc = []
    for _ in range(n_buses):
        if random.random() < 0.6:
            # 60% chance: pick from lower range
            arrival_soc.append(random.uniform(lower_bound, mid_bound))
        else:
            # 40% chance: pick from mid to upper range
            arrival_soc.append(random.uniform(mid_bound, upper_bound))

    energy_needed = [(1 - soc) * battery_capacity for soc in arrival_soc]

    time_labels = [f"{10 + t//2}:{'00' if t % 2 == 0 else '30'} PM" if t < 4 else f"{(t-4)//2}:{'00' if t % 2 == 0 else '30'} AM" for t in range(charging_window)]

    model = pl.LpProblem("Bus_Charging_Optimization", pl.LpMinimize)

    # DECISION VARIABLES
    x = pl.LpVariable.dicts("x", ((i, t) for i in range(n_buses) for t in range(charging_window)), cat="Binary")
    p = pl.LpVariable.dicts("p", ((i, t) for i in range(n_buses) for t in range(charging_window)), lowBound=0)
    peak_demand = pl.LpVariable("peak_demand", lowBound=0)

    # OBJECTIVE: Minimize the peak demand
    model += peak_demand

    # CONSTRAINT 2: Link power to charging status
    for i in range(n_buses):
        for t in range(charging_window):
            model += p[i, t] <= max_rate_per_bus * x[i, t]
            model += p[i, t] >= min_rate_per_bus * x[i, t]

    # CONSTRAINT 3: Energy requirement must be met
    for i in range(n_buses):
        model += pl.lpSum(p[i, t] * slot_duration for t in range(charging_window)) >= energy_needed[i]

    # CONSTRAINT 4: Peak demand constraint for each slot
    for t in range(charging_window):
        model += pl.lpSum(p[i, t] for i in range(n_buses)) <= peak_demand

    # CONSTRAINT 5: Continuous charging (if enabled)
    if CONTINUOUS_CHARGING:
        for i in range(n_buses):
            for t in range(charging_window - 1):
                model += x[i, t] >= x[i, t + 1]





    # SOLVE
    solver = pl.PULP_CBC_CMD(msg=False)
    model.solve(solver)

    print(f"Status: {pl.LpStatus[model.status]}")
    if pl.LpStatus[model.status] != "Optimal":
        print("Warning: Model did not find an optimal solution for seed", seed)
    else:
        # Calculate total demand per slot
        demand_per_slot = [sum(p[i, t].varValue for i in range(n_buses) if p[i, t].varValue is not None) for t in range(charging_window)]
        optimum_max_demand = max(demand_per_slot)
        print(f"Optimum Maximum Demand Charge (kW): {optimum_max_demand:.2f}")
        print("Arrival SoCs:", [f"{soc:.2f}" for soc in arrival_soc])

        # Plot Gaussian distribution of arrival SoC for this seed
        # soc_array = np.array(arrival_soc)
        # mu, std = soc_array.mean(), soc_array.std()
        # xmin, xmax = soc_array.min(), soc_array.max()
        # x = np.linspace(xmin, xmax, 100)
        # p_gauss = norm.pdf(x, mu, std) * n_buses * (xmax - xmin) / 8  # scale to match histogram bins

        # plt.figure(figsize=(7, 4))
        # plt.hist(soc_array, bins=8, density=False, alpha=0.6, color='g', label='Arrival SoC Histogram')
        # plt.plot(x, p_gauss, 'k', linewidth=2, label=f'Gaussian Fit\n$\mu$={mu:.2f}, $\sigma$={std:.2f}')
        # plt.title(f'Arrival SoC Distribution (Seed {seed})')
        # plt.xlabel('Arrival SoC')
        # plt.ylabel('Number of Buses')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
