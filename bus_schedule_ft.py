import pulp
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd

# Parameters
num_buses = 12
battery_capacity = 230  # kWh
min_initial_soc = 0.25
max_initial_soc = 0.4
charging_power_full = 50  # kW
charging_power_taper = 15  # kW
slot_duration = 0.5  # hours
num_slots = 12  # 10 PM to 4 AM
start_hour = 22  # Start time
max_demand = 400  # kW

# Random schedule assignments
schedule_assignments = {t: 0 for t in range(num_slots)}
for _ in range(num_buses):
    slot = random.randint(0, num_slots - 1)
    schedule_assignments[slot] += 1
print("Random Schedule Assignments:", schedule_assignments)

# Generate bus data
buses = []
bus_id = 0
for schedule, count in schedule_assignments.items():
    for _ in range(count):
        initial_soc = random.uniform(min_initial_soc, max_initial_soc)
        energy_needed = battery_capacity * (1 - initial_soc)
        buses.append({
            'id': f'Bus_{bus_id}',
            'schedule': schedule,
            'initial_soc': initial_soc,
            'energy_needed': energy_needed
        })
        bus_id += 1

# MILP model
model = pulp.LpProblem("EV_Charging_Schedule", pulp.LpMinimize)
charging = pulp.LpVariable.dicts("Charging", ((bus['id'], t) for bus in buses for t in range(bus['schedule'], num_slots)), cat='Binary')

model += pulp.lpSum(charging[bus['id'], t] * charging_power_full * slot_duration for bus in buses for t in range(bus['schedule'], num_slots))

# Charging requirement
for bus in buses:
    available_slots = list(range(bus['schedule'], num_slots))
    model += pulp.lpSum(charging[bus['id'], t] * charging_power_full * slot_duration for t in available_slots) >= bus['energy_needed']

# Demand limit
for t in range(num_slots):
    model += pulp.lpSum(charging[bus['id'], t] * charging_power_full for bus in buses if t >= bus['schedule']) <= max_demand

# Solve
model.solve()

# Results matrices
schedule_df = pd.DataFrame(0, index=[bus['id'] for bus in buses], columns=range(num_slots))
soc_matrix = pd.DataFrame(0.0, index=[bus['id'] for bus in buses], columns=range(num_slots))
demand = np.zeros(num_slots)

for bus in buses:
    soc = bus['initial_soc'] * battery_capacity
    for t in range(bus['schedule'], num_slots):
        if pulp.value(charging[bus['id'], t]) == 1:
            percent = soc / battery_capacity
            power = charging_power_full if percent < 0.9 else charging_power_taper
            energy_added = power * slot_duration
            soc += energy_added
            soc = min(soc, battery_capacity)
            schedule_df.at[bus['id'], t] = 1
            demand[t] += power
        soc_matrix.at[bus['id'], t] = soc / battery_capacity

# Print max demand
print("Maximum Demand per Schedule:")
for t in range(num_slots):
    print(f"Time Slot {t} ({start_hour + t//2}:{'00' if t % 2 == 0 else '30'}): {demand[t]:.2f} kW")

# Final SoC Check
final_soc = soc_matrix.iloc[:, -1]
if final_soc.eq(1.0).all():
    print("All buses reach 100% SoC.")
else:
    print("Some buses did not reach 100% SoC.")
    print(final_soc[final_soc < 1.0])

    

# Gantt Chart
plt.figure(figsize=(14, 8))
for i, bus in enumerate(schedule_df.index):
    for t in schedule_df.columns:
        if schedule_df.at[bus, t] == 1:
            plt.barh(i, 1, left=t, color='skyblue')
plt.yticks(range(len(schedule_df.index)), schedule_df.index)
plt.xticks(range(num_slots), [f"{start_hour + t//2}:{'00' if t % 2 == 0 else '30'}" for t in range(num_slots)], rotation=45)
plt.xlabel("Time")
plt.title("Bus Charging Schedule (Gantt Chart)")
plt.tight_layout()
plt.grid(True, axis='x')
plt.show()

# Demand Profile
plt.figure(figsize=(12, 4))
plt.bar(range(num_slots), demand, color='skyblue', alpha=0.7)
plt.plot(demand, marker='o', color='b', label='Total Demand (kW)')
plt.axhline(max_demand, color='r', linestyle='--', label=f'Max Demand = {max_demand} kW')
plt.xticks(range(num_slots), [f"{start_hour + t//2}:{'00' if t % 2 == 0 else '30'}" for t in range(num_slots)])
plt.xlabel("Time Slot")
plt.ylabel("Power (kW)")
plt.title("Power Demand Profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# SoC Plot
plt.figure(figsize=(14, 6))
for bus in soc_matrix.index:
    plt.plot(soc_matrix.columns, soc_matrix.loc[bus], label=bus)
plt.xlabel("Time Slot")
plt.ylabel("State of Charge (fraction)")
plt.title("Bus SoC Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()


