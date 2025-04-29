from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpInteger, LpContinuous, LpStatus, value

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
    prob += rb[g] >= r[g] * b[g]

# Objective: minimise peak power
prob += P_peak

# Constraint: all buses must be assigned
prob += lpSum(b[g] for g in group_times) == num_buses

# Update the charging rate constraint
for g, T in group_times.items():
    prob += rb[g] >= energy_per_bus * b[g] / T

# Constraint: demand at each time slot ≤ P_peak
for t in time_slots:
    active_groups = [g for g in group_times if group_start[g] <= t <= group_start[g] + group_times[g]]
    prob += lpSum(b[g] * r[g] for g in active_groups) <= P_peak

# Solve
prob.solve()
if LpStatus[prob.status] == "Optimal":
    print(f"Minimum Peak Demand: {value(P_peak):.2f} kW")
    for g in sorted(group_times):
        print(f"Group {g}: {int(b[g].varValue)} buses at {r[g].varValue:.2f} kW")
else:
    print("Solver did not find an optimal solution.")
