import pulp
import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# Parameters
num_buses = 20
battery_capacity = 230  # kWh
min_initial_soc = 0.25
max_initial_soc = 0.4
charging_power_full = 90  # kW
# charging_power_taper = 30  # kW
slot_duration = 0.5  # hours
num_slots = 12  # 10 PM to 4 AM
start_hour = 22  # Start time
max_demand = 900  # Increased from 400 kW to allow more simultaneous charging

# Add these constants after other parameters
BATTERY_RANGE_KM = 253  # km
EARLIEST_ARRIVAL_TIME = "21:00"
LATEST_ARRIVAL_TIME = "23:50"
EARLIEST_DEPARTURE_TIME = "04:00"
LATEST_DEPARTURE_TIME = "08:00"

# CONTINUOUS_CHARGING
CONTINUOUS_CHARGING = True  # Set to False to allow breaks

def random_time(start_str, end_str):
    start = datetime.strptime(start_str, "%H:%M")
    end = datetime.strptime(end_str, "%H:%M")
    random_min = random.randint(0, int((end - start).total_seconds() / 60))
    return start + timedelta(minutes=random_min)

# Fixed schedule assignment
def get_slot_time(slot, start_hour):
    hours_to_add = slot // 2
    minutes_to_add = (slot % 2) * 30
    slot_time = datetime.strptime(f"{start_hour}:00", "%H:%M")
    slot_time += timedelta(hours=hours_to_add, minutes=minutes_to_add)
    # Handle midnight crossing
    if slot_time.hour < start_hour:
        slot_time += timedelta(days=1)
    return slot_time

# Generate bus data
buses = []
for i in range(num_buses):
    bus_id = f"Bus_{i:02d}"
    arrival_time = random_time(EARLIEST_ARRIVAL_TIME, LATEST_ARRIVAL_TIME)
    initial_soc = random.uniform(min_initial_soc, max_initial_soc)
    km_driven = round((1 - initial_soc) * BATTERY_RANGE_KM, 1)
    departure_time = random_time(EARLIEST_DEPARTURE_TIME, LATEST_DEPARTURE_TIME)
    energy_needed = battery_capacity * (1 - initial_soc)
    
    # Improved schedule assignment
    slot_found = False
    for t in range(num_slots):
        slot_time = get_slot_time(t, start_hour)
        if arrival_time <= slot_time:
            schedule = t
            slot_found = True
            break
    if not slot_found:
        schedule = num_slots - 1  # Assign to last slot if arrival is after all slots
    
    buses.append({
        'id': bus_id,
        'arrival': arrival_time,
        'soc': initial_soc * 100,  # Store as percentage
        'km_driven': km_driven,
        'departure': departure_time,
        'schedule': schedule,
        'energy_needed': energy_needed
    })

# Sort buses by arrival time
buses.sort(key=lambda x: x['arrival'])

# Print bus personas
print("\n=== BUS PERSONAS ===")
for bus in buses:
    print(f"{bus['id']} | Arrival: {bus['arrival'].strftime('%I:%M %p')} | "
          f"SOC: {bus['soc']:.1f}% | Km Driven: {bus['km_driven']}km | "
          f"Departure: {bus['departure'].strftime('%I:%M %p')} | "
          f"Schedule: Slot {bus['schedule']}")

# Detailed Bus Information
print("\n=== DETAILED BUS INFORMATION ===")
print("Bus ID | Arrival Time | Initial SOC | Km Driven | Departure Time | Energy Needed")
print("-" * 80)
for bus in buses:
    print(f"{bus['id']:6} | "
          f"{bus['arrival'].strftime('%I:%M %p'):11} | "
          f"{bus['soc']:8.1f}% | "
          f"{bus['km_driven']:9.1f}km | "
          f"{bus['departure'].strftime('%I:%M %p'):13} | "
          f"{bus['energy_needed']:8.1f} kWh")
print("-" * 80)
print(f"Total energy needed for all buses: {sum(bus['energy_needed'] for bus in buses):.1f} kWh")

# MILP model with robust SOC tracking and tapering

model = pulp.LpProblem("EV_Charging_Schedule", pulp.LpMinimize)

# Decision variables
charging_power = pulp.LpVariable.dicts(
    "ChargingPower",
    ((bus['id'], t) for bus in buses for t in range(bus['schedule'], num_slots)),
    lowBound=0, upBound=charging_power_full, cat='Continuous'
)

# Objective: minimize total energy used 
model += pulp.lpSum(
    charging_power[bus['id'], t] * slot_duration
    for bus in buses for t in range(bus['schedule'], num_slots)
)

# SOC tracking variables
soc_vars = {}
for bus in buses:
    for t in range(bus['schedule'], num_slots + 1):  # +1 for initial SOC
        soc_vars[(bus['id'], t)] = pulp.LpVariable(f"soc_{bus['id']}_{t}", lowBound=0, upBound=1, cat='Continuous')

# Initial SOC constraint
for bus in buses:
    soc_init = bus['soc'] / 100
    model += soc_vars[(bus['id'], bus['schedule'])] == soc_init

# SOC progression and tapering constraints
# Piecewise linear tapering breakpoints (SOC as fraction) and power levels (kW)
soc_breakpoints = [0.8, 0.9, 0.95, 1.0]
power_levels = [charging_power_full, 70, 40, 20]  # Example: adjust as needed

for bus in buses:
    for t in range(bus['schedule'], num_slots):
        # SOC progression
        model += soc_vars[(bus['id'], t+1)] == soc_vars[(bus['id'], t)] + (charging_power[bus['id'], t] * slot_duration) / battery_capacity

        # Piecewise linear tapering
        soc_t = soc_vars[(bus['id'], t)]
        # For SOC < 80%
        model += charging_power[bus['id'], t] <= power_levels[0]
        # For SOC 80-90%
        model += charging_power[bus['id'], t] <= power_levels[1] + (power_levels[0] - power_levels[1]) * (soc_breakpoints[1] - soc_t) / (soc_breakpoints[1] - soc_breakpoints[0])
        # For SOC 90-95%
        model += charging_power[bus['id'], t] <= power_levels[2] + (power_levels[1] - power_levels[2]) * (soc_breakpoints[2] - soc_t) / (soc_breakpoints[2] - soc_breakpoints[1])
        # For SOC 95-100%
        model += charging_power[bus['id'], t] <= power_levels[3] + (power_levels[2] - power_levels[3]) * (1.0 - soc_t) / (1.0 - soc_breakpoints[2])

# Final SOC must be 1.0 (100%)
for bus in buses:
    model += soc_vars[(bus['id'], num_slots)] >= 1.0

# Demand limit
for t in range(num_slots):
    model += pulp.lpSum(
        charging_power[bus['id'], t]
        for bus in buses if t >= bus['schedule']
    ) <= max_demand

# Continuous charging constraint (if enabled)
if CONTINUOUS_CHARGING:
    for bus in buses:
        for t in range(bus['schedule'], num_slots - 1):
            # If charging in t+1, must have charged in t
            model += charging_power[bus['id'], t] >= charging_power[bus['id'], t + 1]

# Force each bus to start charging at its assigned schedule slot
for bus in buses:
    model += charging_power[bus['id'], bus['schedule']] >= 1e-3  # Must charge at least a little in the first available slot

# Solve
model.solve()

# Results matrices
schedule_df = pd.DataFrame(0, index=[bus['id'] for bus in buses], columns=range(num_slots))
soc_matrix = pd.DataFrame(0.0, index=[bus['id'] for bus in buses], columns=range(num_slots))
demand = np.zeros(num_slots)

# Update SOC tracking loop (matches MILP solution exactly)
for bus in buses:
    soc = bus['soc'] / 100 * battery_capacity  # SOC in kWh
    for t in range(bus['schedule'], num_slots):
        percent = soc / battery_capacity
        # Piecewise power logic
        if percent < soc_breakpoints[0]:
            power = power_levels[0]
        elif percent < soc_breakpoints[1]:
            power = power_levels[1]
        elif percent < soc_breakpoints[2]:
            power = power_levels[2]
        else:
            power = power_levels[3]
        actual_power = min(power, pulp.value(charging_power[bus['id'], t]))
        if actual_power > 1e-3:
            schedule_df.at[bus['id'], t] = 1
            demand[t] += actual_power
        soc += actual_power * slot_duration
        soc_matrix.at[bus['id'], t] = soc / battery_capacity

# Table: kW consumed by each bus per schedule (time slot) + total
power_consumed_df = pd.DataFrame(0.0, index=[bus['id'] for bus in buses], columns=[f"Slot {t}" for t in range(num_slots)])

for bus in buses:
    for t in range(bus['schedule'], num_slots):
        power = pulp.value(charging_power[bus['id'], t])
        if power is not None and power > 1e-3:
            power_consumed_df.at[bus['id'], f"Slot {t}"] = power

# Add a 'Total' column for each bus
power_consumed_df['Total (kW)'] = power_consumed_df.sum(axis=1)

print("\n=== kW Consumed by Each Bus per Schedule (Time Slot) ===")
print(power_consumed_df.round(2).to_string())

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

# Print charging timeline for each bus
print("\n=== CHARGING TIMELINE ===")
print("Bus ID | Start Slot | Active Charging Slots | Final SOC")
print("-" * 70)
for bus in buses:
    charging_slots = []
    for t in range(num_slots):
        if schedule_df.at[bus['id'], t] == 1:
            charging_slots.append(t)
    
    if charging_slots:
        start_slot = min(charging_slots)
        active_slots = [str(s) for s in charging_slots]
        final_soc = soc_matrix.at[bus['id'], num_slots-1] * 100
        print(f"{bus['id']:6} | {start_slot:^10} | {', '.join(active_slots):^20} | {final_soc:>.1f}%")
    else:
        print(f"{bus['id']:6} | {'N/A':^10} | {'No charging':^20} | {bus['soc']:.1f}%")
print("-" * 70)

# Add slot time reference
print("\nSlot Time Reference:")
for t in range(num_slots):
    slot_start = f"{start_hour + t//2}:{'00' if t % 2 == 0 else '30'}"
    print(f"Slot {t}: {slot_start}")

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

# SoC Plot
plt.figure(figsize=(14, 6))
for bus in soc_matrix.index:
    plt.plot(soc_matrix.columns, soc_matrix.loc[bus], label=bus)
plt.xlabel("Time Slot")
plt.ylabel("State of Charge (fraction)")
plt.title("Bus SoC Over Time")
plt.grid(True)
plt.tight_layout()


# Single Bus Charging Profile (Random Selection)
selected_bus = random.choice(buses)  # Randomly select a bus
plt.figure(figsize=(12, 6))

# Get charging data for selected bus
bus_soc = soc_matrix.loc[selected_bus['id']] * 100  # Convert to percentage
charging_times = [f"{start_hour + t//2}:{'00' if t % 2 == 0 else '30'}" for t in range(num_slots)]

# Plot the charging profile
plt.plot(charging_times, bus_soc, 'b-o', linewidth=2, markersize=8, label='Actual SOC')
plt.axhline(y=selected_bus['soc'], color='r', linestyle='--', 
           label=f'Initial SOC ({selected_bus["soc"]:.1f}%)')
plt.axhline(y=100, color='g', linestyle='--', label='Target SOC (100%)')

# Add charging power annotations (piecewise tapering)
for t in range(num_slots):
    if schedule_df.at[selected_bus['id'], t] == 1:
        soc_value = bus_soc[t] / 100  # Convert to fraction for comparison
        if soc_value < soc_breakpoints[0]:
            power = power_levels[0]
        elif soc_value < soc_breakpoints[1]:
            power = power_levels[1]
        elif soc_value < soc_breakpoints[2]:
            power = power_levels[2]
        else:
            power = power_levels[3]
        plt.annotate(f'{power}kW', 
                    (t, bus_soc[t]),
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center')

# Customize the plot
plt.title(f"Charging Profile for {selected_bus['id']}\n" + 
          f"Arrival: {selected_bus['arrival'].strftime('%I:%M %p')} | Departure: {selected_bus['departure'].strftime('%I:%M %p')}")
plt.xlabel("Time Slot")
plt.ylabel("State of Charge (%)")
plt.legend()
plt.grid(True)
plt.tight_layout()


# Scatter plot: Arrival Time vs SOC
plt.figure(figsize=(10, 6))
arrival_times = [bus['arrival'] for bus in buses]
soc_levels = [bus['soc'] for bus in buses]
bus_ids = [bus['id'] for bus in buses]
plt.scatter(arrival_times, soc_levels, color='green', label='Bus')

for x, y, label in zip(arrival_times, soc_levels, bus_ids):
    plt.annotate(label, (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

plt.title('Bus Arrival Time vs SOC Level')
plt.xlabel('Arrival Time')
plt.ylabel('SOC (%)')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# # Scatter plot: Arrival Time vs Charging Start Time (only for buses that actually charged)
# plt.figure(figsize=(10, 6))
# arrival_times_plot = []
# charging_start_times_plot = []
# bus_ids_plot = []

# for bus in buses:
#     # Find the first slot where the bus actually charges
#     first_charging_slot = None
#     for t in range(num_slots):
#         if schedule_df.at[bus['id'], t] == 1:
#             first_charging_slot = t
#             break
#     if first_charging_slot is not None:
#         arrival_times_plot.append(bus['arrival'])
#         charging_start_times_plot.append(get_slot_time(first_charging_slot, start_hour))
#         bus_ids_plot.append(bus['id'])

# plt.scatter(arrival_times_plot, charging_start_times_plot, color='blue', label='Bus')

# for x, y, label in zip(arrival_times_plot, charging_start_times_plot, bus_ids_plot):
#     plt.annotate(label, (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)

# plt.title('Bus Arrival Time vs Charging Start Time')
# plt.xlabel('Arrival Time')
# plt.ylabel('Charging Start Time (First Charging Slot)')
# plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
# plt.gca().yaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.tight_layout()
# plt.show()


