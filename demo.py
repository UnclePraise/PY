import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pulp

# Parameters
time_slots = 32  # 15-minute intervals from 10 PM to 6 AM
charging_power = 50  # kW
slot_duration = 0.25  # 15 mins = 0.25 hour
max_chargers = 60

# Mock data for 5 buses
buses = [
    {"id": "Bus1", "arrival": 0, "departure": 32, "energy_needed": 100},
    {"id": "Bus2", "arrival": 4, "departure": 28, "energy_needed": 80},
    {"id": "Bus3", "arrival": 8, "departure": 32, "energy_needed": 120},
    {"id": "Bus4", "arrival": 0, "departure": 16, "energy_needed": 60},
    {"id": "Bus5", "arrival": 12, "departure": 32, "energy_needed": 70},
]

# Create MILP model
model = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMinimize)

# Variables: x[b,t] = 1 if bus b charges in time slot t
x = {
    (b['id'], t): pulp.LpVariable(f"x_{b['id']}_{t}", cat='Binary')
    for b in buses for t in range(time_slots)
}

# Variable to track peak demand
peak = pulp.LpVariable("peak", lowBound=0)

# Objective: Minimize peak demand
model += peak

# Constraint: Each bus must get enough energy
for b in buses:
    model += pulp.lpSum(x[(b['id'], t)] * charging_power * slot_duration for t in range(b['arrival'], b['departure'])) >= b['energy_needed']

# Constraint: Limit total buses charging per time slot
for t in range(time_slots):
    model += pulp.lpSum(x[(b['id'], t)] for b in buses if t >= b['arrival'] and t < b['departure']) * charging_power <= peak
    model += pulp.lpSum(x[(b['id'], t)] for b in buses if t >= b['arrival'] and t < b['departure']) <= max_chargers

# Solve the problem
model.solve()

# Extract schedule and compute demand profile
schedule = pd.DataFrame(0, index=[b['id'] for b in buses], columns=range(time_slots))
demand = np.zeros(time_slots)

for b in buses:
    for t in range(b['arrival'], b['departure']):
        if pulp.value(x[(b['id'], t)]) == 1:
            schedule.at[b['id'], t] = 1
            demand[t] += charging_power

# Plot Gantt chart
plt.figure(figsize=(12, 6))
for i, bus in enumerate(schedule.index):
    times = schedule.columns[schedule.loc[bus] == 1]
    plt.barh(i, len(times), left=min(times) if len(times) > 0 else 0, height=0.6)

plt.yticks(range(len(schedule.index)), schedule.index)
plt.xlabel("Time Slot (15-min intervals from 10PM)")
plt.title("Bus Charging Schedule (Gantt Chart)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot demand profile
plt.figure(figsize=(12, 4))
plt.plot(demand, marker='o', label='Total Demand (kW)')
plt.axhline(pulp.value(peak), color='r', linestyle='--', label=f'Max Demand = {pulp.value(peak):.0f} kW')
plt.xlabel("Time Slot (15-min intervals from 10PM)")
plt.ylabel("Power (kW)")
plt.title("Power Demand Profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
