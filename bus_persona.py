import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time
import pulp
import numpy as np
import pandas as pd

random.seed(42)

# Parameters
NUM_BUSES = 20
NUM_SLOTS = 5
SLOT_DURATION_MIN = 30
SLOTS_START = datetime.strptime("22:00", "%H:%M")
BATTERY_RANGE_KM = 253
BATTERY_SIZE_KWH = 230
MIN_SOC = 25
MAX_SOC = 40
EARLIEST_ARRIVAL_TIME = "21:00"
LATEST_ARRIVAL_TIME = "23:50"
EARLIEST_DEPARTURE_TIME = "04:00"
LATEST_DEPARTURE_TIME = "08:00"
charging_power_full = 100  # kW
charging_power_taper = 50  # kW
slot_duration = SLOT_DURATION_MIN / 60  # hours
max_demand = 10400  # kW

# Create charging slots
charging_slots = [(SLOTS_START + timedelta(minutes=i*SLOT_DURATION_MIN),
                   SLOTS_START + timedelta(minutes=(i+1)*SLOT_DURATION_MIN)) for i in range(NUM_SLOTS)]

def random_time(start_str, end_str):
    start = datetime.strptime(start_str, "%H:%M")
    end = datetime.strptime(end_str, "%H:%M")
    random_min = random.randint(0, int((end - start).total_seconds() / 60))
    return start + timedelta(minutes=random_min)

# Generate Bus Personas
buses = []
for i in range(NUM_BUSES):
    bus_id = f"Bus_{i+1:02d}"
    arrival_time = random_time(EARLIEST_ARRIVAL_TIME, LATEST_ARRIVAL_TIME)
    soc = round(random.uniform(MIN_SOC, MAX_SOC), 1)
    km_driven = round((100 - soc) / 100 * BATTERY_RANGE_KM, 1)
    departure_time = random_time(EARLIEST_DEPARTURE_TIME, LATEST_DEPARTURE_TIME)
    needed_kwh = round((100 - soc) / 100 * BATTERY_SIZE_KWH, 1)
    buses.append({
        "id": bus_id,
        "arrival": arrival_time,
        "soc": soc,
        "km_driven": km_driven,
        "departure": departure_time,
        "needed_kwh": needed_kwh
    })

# Assign each bus to the earliest possible slot
bus_schedule_map = {}
for bus in buses:
    for idx, (start, end) in enumerate(charging_slots):
        if bus["arrival"] <= start - timedelta(minutes=10):
            bus_schedule_map[bus["id"]] = idx
            break

# Prepare bus data for MILP
milp_buses = []
for bus in buses:
    initial_soc = bus['soc'] / 100
    energy_needed = BATTERY_SIZE_KWH * (1 - initial_soc)
    schedule = bus_schedule_map[bus['id']]
    milp_buses.append({
        'id': bus['id'],
        'schedule': schedule,
        'initial_soc': initial_soc,
        'energy_needed': energy_needed
    })

# MILP model
model = pulp.LpProblem("EV_Charging_Schedule", pulp.LpMinimize)
charging = pulp.LpVariable.dicts("Charging", ((bus['id'], t) for bus in milp_buses for t in range(bus['schedule'], NUM_SLOTS)), cat='Binary')

model += pulp.lpSum(charging[bus['id'], t] * charging_power_full * slot_duration for bus in milp_buses for t in range(bus['schedule'], NUM_SLOTS))

# Charging requirement
for bus in milp_buses:
    available_slots = list(range(bus['schedule'], NUM_SLOTS))
    model += pulp.lpSum(charging[bus['id'], t] * charging_power_full * slot_duration for t in available_slots) >= bus['energy_needed']

# Demand limit
for t in range(NUM_SLOTS):
    model += pulp.lpSum(charging[bus['id'], t] * charging_power_full for bus in milp_buses if t >= bus['schedule']) <= max_demand

# Solve
model.solve()

# Results matrices
schedule_df = pd.DataFrame(0, index=[bus['id'] for bus in milp_buses], columns=range(NUM_SLOTS))
soc_matrix = pd.DataFrame(0.0, index=[bus['id'] for bus in milp_buses], columns=range(NUM_SLOTS))
demand = np.zeros(NUM_SLOTS)

for bus in milp_buses:
    soc = bus['initial_soc'] * BATTERY_SIZE_KWH
    for t in range(bus['schedule'], NUM_SLOTS):
        if pulp.value(charging[bus['id'], t]) == 1:
            percent = soc / BATTERY_SIZE_KWH
            power = charging_power_full if percent < 0.9 else charging_power_taper
            energy_added = power * slot_duration
            soc += energy_added
            soc = min(soc, BATTERY_SIZE_KWH)
            schedule_df.at[bus['id'], t] = 1
            demand[t] += power
        soc_matrix.at[bus['id'], t] = soc / BATTERY_SIZE_KWH

# Print max demand
print("Maximum Demand per Schedule:")
for t in range(NUM_SLOTS):
    slot_time = charging_slots[t][0].strftime('%I:%M %p')
    print(f"Time Slot {t} ({slot_time}): {demand[t]:.2f} kW")

# Gantt Chart
plt.figure(figsize=(14, 8))
for i, bus in enumerate(schedule_df.index):
    charged_any = False
    for t in schedule_df.columns:
        if schedule_df.at[bus, t] == 1:
            plt.barh(i, 1, left=t, color='skyblue')
            charged_any = True
    if not charged_any:
        plt.barh(i, 1, left=0, color='lightgray', alpha=0.3)
plt.yticks(range(len(schedule_df.index)), schedule_df.index)
plt.xticks(range(NUM_SLOTS), [charging_slots[t][0].strftime('%I:%M %p') for t in range(NUM_SLOTS)], rotation=45)
plt.xlabel("Time")
plt.title("Bus Charging Schedule (Gantt Chart, MILP Optimized)")
plt.tight_layout()
plt.grid(True, axis='x')
plt.show()

# Demand Profile
plt.figure(figsize=(12, 4))
plt.bar(range(NUM_SLOTS), demand, color='skyblue', alpha=0.7)
plt.plot(demand, marker='o', color='b', label='Total Demand (kW)')
plt.axhline(max_demand, color='r', linestyle='--', label=f'Max Demand = {max_demand} kW')
plt.xticks(range(NUM_SLOTS), [charging_slots[t][0].strftime('%I:%M %p') for t in range(NUM_SLOTS)])
plt.xlabel("Time Slot")
plt.ylabel("Power (kW)")
plt.title("Power Demand Profile (MILP Optimized)")
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
plt.title("Bus SoC Over Time (MILP Optimized)")
plt.grid(True)
plt.tight_layout()
plt.show()

# Final SoC Check
final_soc = soc_matrix.iloc[:, -1]
if final_soc.eq(1.0).all():
    print("All buses reach 100% SoC.")
else:
    print("Some buses did not reach 100% SoC.")
    print(final_soc[final_soc < 1.0])

