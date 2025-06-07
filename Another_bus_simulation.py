from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, LpContinuous, LpStatus, value
import numpy as np
from scipy import stats
import random

# Parameters
num_buses = 20
energy_per_bus = 230  # kWh
r_min = 20            # kW
r_max = 60            # kW

# Group charging windows (in hours until 4am)
group_times = {
    1: 6.0,  # starts 10:00pm
    2: 5.5,  # starts 10:30pm
    3: 5.0,  # starts 11:00pm
    4: 4.5,  # starts 11:30pm
    5: 4.0   # starts 12:00am
}

# Group start times (decimal hours from 22:00)
group_start = {
    1: 22.0,
    2: 22.5,
    3: 23.0,
    4: 23.5,
    5: 24.0
}

# Time slots (every 30 min from 22:00 to 04:00)
time_slots = [22 + 0.5 * i for i in range(13)]  # 22.0 to 28.0

# Create the problem
prob = LpProblem("Minimize_Peak_Demand", LpMinimize)

# Decision variables
b = {g: LpVariable(f"b_{g}", lowBound=0, cat=LpInteger) for g in group_times}
r = {g: LpVariable(f"r_{g}", lowBound=r_min, upBound=r_max, cat=LpContinuous) for g in group_times}
P_peak = LpVariable("P_peak", lowBound=0, cat=LpContinuous)

# Auxiliary variables for the product of r[g] and b[g]
rb = {g: LpVariable(f"rb_{g}", lowBound=0, cat=LpContinuous) for g in group_times}

# Linearize the product of r[g] and b[g]
for g in group_times:
    prob += rb[g] <= r[g] * num_buses
    prob += rb[g] <= b[g] * r_max
    prob += rb[g] >= r[g] + b[g] - (num_buses + r_max)

# Objective: minimise peak power
prob += P_peak

# Constraint: all buses must be assigned
prob += lpSum(b[g] for g in group_times) == num_buses

# Generate arrival times using skewed normal distribution
random.seed(42)  # For reproducibility
alpha = 4  # Skewness parameter (positive for right skew)
loc = 22.0  # Location parameter (10 PM = 22:00)
scale = 1.0  # Scale parameter

# Generate arrival times for all buses
arrival_times = stats.skewnorm.rvs(alpha, loc=loc, scale=scale, size=num_buses)
# Clip arrival times to be between 22:00 and 24:00
arrival_times = np.clip(arrival_times, 22.0, 24.0)
# Sort arrival times
arrival_times.sort()

# Create dictionary to store buses per group based on arrival times
buses_per_group = {g: 0 for g in group_times}
for arrival_time in arrival_times:
    # Assign to the earliest possible group that starts after arrival
    for g in sorted(group_start.keys()):
        if group_start[g] >= arrival_time:
            buses_per_group[g] += 1
            break

# Update constraints to respect arrival times
prob += lpSum(b[g] for g in group_times) == num_buses

for g in group_times:
    prob += b[g] <= buses_per_group[g]  # Can't assign more buses than available

# Update the charging rate constraint
for g, T in group_times.items():
    prob += rb[g] >= energy_per_bus * b[g] / T

# Constraint: demand at each time slot ≤ P_peak
for t in time_slots:
    active_groups = [g for g in group_times if group_start[g] <= t <= group_start[g] + group_times[g]]
    prob += lpSum(rb[g] for g in active_groups) <= P_peak

# Solve
prob.solve()
# Update the output to include arrival time information
if LpStatus[prob.status] == "Optimal":
    print(f"Minimum Peak Demand: {value(P_peak):.2f} kW")
    
    print("\nBus Distribution and Charging Rates:")
    for g in sorted(group_times):
        print(f"Group {g} (starts {group_start[g]:02.1f}:00):")
        print(f"  Available buses: {buses_per_group[g]}")
        print(f"  Assigned buses: {int(b[g].varValue)}")
        print(f"  Charging rate: {r[g].varValue:.2f} kW")
    
    print("\nArrival Time Distribution:")
    for hour in range(22, 25):
        count = sum(1 for t in arrival_times if hour <= t < hour + 1)
        print(f"{hour:02d}:00-{(hour+1):02d}:00: {'*' * count} ({count} buses)")
    
    print("\nIndividual Bus Arrival Times:")
    for i, arrival_time in enumerate(arrival_times, 1):
        # Convert decimal hours to HH:MM format
        hours = int(arrival_time)
        minutes = int((arrival_time - hours) * 60)
        print(f"Bus {i:2d}: {hours:02d}:{minutes:02d}")
else:
    print("Solver did not find an optimal solution.")
