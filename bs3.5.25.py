import random
import pulp as pl

# PARAMETERS
n_buses = 20
battery_capacity = 300  # kWh
charging_window = 12    # from 10pm to 4am (half-hour slots)
n_groups = 5
slot_duration = 0.5    # hours
max_rate_per_bus = 60  # kW
min_rate_per_bus = 20  # kW

# Random SOC between 25% and 40%
arrival_soc = [random.uniform(0.25, 0.4) for _ in range(n_buses)]
energy_needed = [(1 - soc) * battery_capacity for soc in arrival_soc]

# Group start times (in half-hour slots)
group_start_slots = [i for i in range(n_groups)]

# INIT MODEL
model = pl.LpProblem("Bus_Charging_Optimization", pl.LpMinimize)

# DECISION VARIABLES
x = pl.LpVariable.dicts("x", ((i, t) for i in range(n_buses) for t in range(charging_window)), cat="Binary")
g = pl.LpVariable.dicts("g", ((i, k) for i in range(n_buses) for k in range(n_groups)), cat="Binary")
r = pl.LpVariable.dicts("r", (k for k in range(n_groups)), lowBound=min_rate_per_bus, upBound=max_rate_per_bus)
# New variable for actual power delivered
p = pl.LpVariable.dicts("p", ((i, t) for i in range(n_buses) for t in range(charging_window)), lowBound=0)
peak = pl.LpVariable("peak", lowBound=0)

# OBJECTIVE: Minimize peak demand
model += peak

# CONSTRAINT 1: Each bus is assigned to exactly one group
for i in range(n_buses):
    model += pl.lpSum(g[i, k] for k in range(n_groups)) == 1

# CONSTRAINT 2: Buses can only charge from their group's start time
for i in range(n_buses):
    for t in range(charging_window):
        for k in range(n_groups):
            if t < group_start_slots[k]:
                model += x[i, t] <= 1 - g[i, k]

# CONSTRAINT 3: Link power to charging status and rate
for i in range(n_buses):
    for t in range(charging_window):
        # Power must be zero if not charging
        model += p[i, t] <= max_rate_per_bus * x[i, t]
        # Power must respect group rate
        for k in range(n_groups):
            model += p[i, t] <= r[k] + max_rate_per_bus * (1 - g[i, k])

# CONSTRAINT 4: Energy requirement must be met
for i in range(n_buses):
    model += pl.lpSum(p[i, t] * slot_duration for t in range(charging_window)) >= energy_needed[i]

# CONSTRAINT 5: Peak power constraint
for t in range(charging_window):
    model += pl.lpSum(p[i, t] for i in range(n_buses)) <= peak

# CONSTRAINT 6: Continuous charging (no breaks once started)
for i in range(n_buses):
    for t in range(charging_window-1):
        model += x[i, t] >= x[i, t+1]

# Solve
solver = pl.PULP_CBC_CMD(msg=False)
status = model.solve(solver)

# Print results
print(f"Status: {pl.LpStatus[model.status]}")
print(f"Minimum Peak Demand: {pl.value(peak):.2f} kW")

print("\nCharging Rates per Group:")
for k in range(n_groups):
    print(f" Group {k+1}: {pl.value(r[k]):.2f} kW")

print("\nBus Assignments:")
for i in range(n_buses):
    assigned_group = [k for k in range(n_groups) if pl.value(g[i, k]) == 1][0]
    charging_slots = [t for t in range(charging_window) if pl.value(x[i, t]) == 1]
    total_energy = sum(pl.value(p[i, t]) * slot_duration for t in range(charging_window))
    print(f" Bus {i+1}: Group {assigned_group+1}")
    print(f"     Initial SoC: {arrival_soc[i]*100:.1f}%")
    print(f"     Energy Delivered: {total_energy:.1f} kWh")
    print(f"     Charging Slots: {charging_slots}")
