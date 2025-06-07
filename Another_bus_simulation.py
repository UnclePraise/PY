# Re-run after code execution environment reset

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, LpBinary, LpStatus, value

# Helper function to convert decimal hours to HH:MM format
def hour_to_time_str(hour):
    h = int(hour)
    m = int((hour - h) * 60)
    # Handle hours past midnight
    if h >= 24:
        h -= 24
    return f"{h:02d}:{m:02d}"

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
NUM_BUSES = 20
ENERGY_FULL = 230  # kWh
SOC_MIN, SOC_MAX = 0.25, 0.35
R_MIN, R_MAX = 20, 60  # kW charger limits
GROUPS = [1, 2, 3, 4, 5]
GROUP_START = {1: 22.0, 2: 22.5, 3: 23.0, 4: 23.5, 5: 24.0}
GROUP_TIME = {g: 28.0 - GROUP_START[g] for g in GROUPS}  # Charging until 04:00 AM (28.0)
TIME_SLOTS = [22.0 + 0.5 * i for i in range(13)]  # 22.0 to 28.0 in 0.5 hr steps
END_TIME = 28.0  # 4 AM in decimal hours
MARGIN_MINUTES = 5  # Safety margin in minutes

# Random SoC and energy requirement
soc = np.random.uniform(SOC_MIN, SOC_MAX, NUM_BUSES)
energy_required = (1 - soc) * ENERGY_FULL  # in kWh

# Define the optimization problem
prob = LpProblem("Bus_Charging_Optimization", LpMinimize)

# Variables
x = {(i, g): LpVariable(f"x_{i}_{g}", cat=LpBinary) for i in range(NUM_BUSES) for g in GROUPS}
r = {}
for g in GROUPS:
    # Calculate minimum required rate for this group
    total_energy_in_group = lpSum(x[i, g] * energy_required[i] for i in range(NUM_BUSES))
    prob += total_energy_in_group / (END_TIME - GROUP_START[g] - MARGIN_MINUTES/60) <= r[g]
    
    # Set bounds for charging rate
    r[g] = LpVariable(f"r_{g}", 
                     lowBound=R_MIN,
                     upBound=max(R_MAX, ENERGY_FULL / (END_TIME - GROUP_START[g] - MARGIN_MINUTES/60)))

P_peak = LpVariable("P_peak", lowBound=0)
power_delivered = {g: LpVariable(f"power_delivered_{g}", lowBound=0) for g in GROUPS}

# Objective: minimize peak power
prob += P_peak

# Constraint 1: Each bus assigned to exactly one group
for i in range(NUM_BUSES):
    prob += lpSum(x[i, g] for g in GROUPS) == 1

# Constraint 2: Energy delivered per group >= energy required
for g in GROUPS:
    # Link power_delivered to r[g] and x[i,g]
    prob += power_delivered[g] <= R_MAX * lpSum(x[i, g] for i in range(NUM_BUSES))
    prob += power_delivered[g] >= R_MIN * lpSum(x[i, g] for i in range(NUM_BUSES))
    prob += power_delivered[g] <= r[g] * NUM_BUSES
    # Energy constraint using power_delivered
    prob += power_delivered[g] * GROUP_TIME[g] >= lpSum(x[i, g] * energy_required[i] for i in range(NUM_BUSES))

# Constraint 3: Peak demand at each time slot
for t in TIME_SLOTS:
    active_groups = [g for g in GROUPS if GROUP_START[g] <= t < GROUP_START[g] + GROUP_TIME[g]]
    prob += lpSum(power_delivered[g] for g in active_groups) <= P_peak

# Add group size limits
MAX_BUSES_PER_GROUP = 6  # Adjust as needed
for g in GROUPS:
    prob += lpSum(x[i, g] for i in range(NUM_BUSES)) <= MAX_BUSES_PER_GROUP

# Add minimum group size to ensure distribution
MIN_BUSES_PER_GROUP = 2  # Adjust as needed
for g in GROUPS:
    prob += lpSum(x[i, g] for i in range(NUM_BUSES)) >= MIN_BUSES_PER_GROUP

# Optional: Add SOC-based assignment preference
for i in range(NUM_BUSES):
    for g in GROUPS:
        # Buses with lower SOC prefer earlier groups
        prob += x[i, g] * soc[i] <= 0.35  # Adjust threshold as needed

# Add completion time constraint
for g in GROUPS:
    for i in range(NUM_BUSES):
        prob += (x[i, g] * (GROUP_START[g] + energy_required[i]/r[g])) <= END_TIME - MARGIN_MINUTES/60

# Solve the problem
prob.solve()

# Extract results
bus_group = [None] * NUM_BUSES
for i in range(NUM_BUSES):
    for g in GROUPS:
        if value(x[i, g]) > 0.5:
            bus_group[i] = g
group_sizes = {g: sum(1 for i in range(NUM_BUSES) if bus_group[i] == g) for g in GROUPS}
r_values = {g: value(r[g]) for g in GROUPS}

# Compute load profile for visualization
load_profile = []
for t in TIME_SLOTS:
    load = 0
    for g in GROUPS:
        if GROUP_START[g] <= t < GROUP_START[g] + GROUP_TIME[g]:
            load += r_values[g] * group_sizes[g]
    load_profile.append(load)

# Update TIME_SLOTS labels for plotting
time_labels = [hour_to_time_str(t) for t in TIME_SLOTS]

# Plotting
fig, ax = plt.subplots(figsize=(10, 5))
ax.step(TIME_SLOTS, load_profile, where='post', label="Total Load (kW)")
ax.axhline(value(P_peak), color='r', linestyle='--', label=f"Peak Load: {value(P_peak):.1f} kW")
ax.set_xlabel("Time")
ax.set_ylabel("Power Demand (kW)")
ax.set_title("Bus Fleet Charging Load Profile")
ax.set_xticks(TIME_SLOTS)
ax.set_xticklabels(time_labels, rotation=45)
ax.legend()
plt.grid(True)
plt.tight_layout()
# plt.show()

# Create DataFrame for bus information
bus_data = pd.DataFrame({
    'Bus': [f'Bus_{i}' for i in range(NUM_BUSES)],
    'Initial_SOC': soc * 100,  # Convert to percentage
    'Group': bus_group,
    'Start_Time': [GROUP_START[g] for g in bus_group],
    'Energy_Required': energy_required
})

# Sort by group start time for better visualization
bus_data = bus_data.sort_values('Start_Time')

# After the bus_data DataFrame creation, add charging completion times
def calculate_full_charge_time(row):
    group = row['Group']
    energy_needed = row['Energy_Required']
    charging_rate = r_values[group]
    charging_hours = energy_needed / charging_rate
    start_time = row['Start_Time']
    completion_time = start_time + charging_hours
    return completion_time

# Add completion times to the DataFrame
bus_data['Completion_Time'] = bus_data.apply(calculate_full_charge_time, axis=1)
bus_data['Completion_Time_Str'] = bus_data['Completion_Time'].apply(hour_to_time_str)

# Update the printed information
print("\n=== Bus Charging Assignment Details ===")
print("Bus | Group | Start Time | Completion Time | Initial SOC | Energy Required")
print("-" * 75)
for _, row in bus_data.iterrows():
    print(f"{row['Bus']:6} | {row['Group']:5} | {hour_to_time_str(row['Start_Time']):10} | "
          f"{row['Completion_Time_Str']:14} | {row['Initial_SOC']:10.1f}% | {row['Energy_Required']:10.1f} kWh")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

# Plot 1: SOC and Charging Slot Assignment
colors = plt.cm.Set3(np.linspace(0, 1, len(GROUPS)))
for g, color in zip(GROUPS, colors):
    mask = bus_data['Group'] == g
    ax1.scatter(bus_data[mask]['Start_Time'], 
               bus_data[mask]['Initial_SOC'],
               c=[color], 
               label=f'Group {g}',
               s=100)

# Add bus labels
for _, row in bus_data.iterrows():
    ax1.annotate(row['Bus'], 
                (row['Start_Time'], row['Initial_SOC']),
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8)

# Add completion times to the scatter plot
for _, row in bus_data.iterrows():
    ax1.annotate(f"→{row['Completion_Time_Str']}", 
                (row['Start_Time'], row['Initial_SOC']),
                xytext=(5, -10), 
                textcoords='offset points',
                fontsize=8,
                color='darkred')

ax1.set_xlabel('Charging Start Time')
# Convert Start_Time to formatted strings for scatter plot
bus_data['Start_Time_Str'] = bus_data['Start_Time'].apply(hour_to_time_str)
ax1.set_xticks(sorted(GROUP_START.values()))
ax1.set_xticklabels([hour_to_time_str(t) for t in sorted(GROUP_START.values())], rotation=45)
ax1.set_ylabel('Initial SOC (%)')
ax1.set_title('Bus Initial SOC vs Charging Start Time by Group')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend(title='Charging Groups')

# Plot 2: Group Sizes and Charging Rates
bar_positions = np.arange(len(GROUPS))
bar_width = 0.35

ax2.bar(bar_positions - bar_width/2, 
        [group_sizes[g] for g in GROUPS],
        bar_width,
        label='Buses in Group',
        color='skyblue')

ax2.bar(bar_positions + bar_width/2, 
        [r_values[g] for g in GROUPS],
        bar_width,
        label='Charging Rate (kW)',
        color='lightgreen')

ax2.set_xticks(bar_positions)
ax2.set_xticklabels([f'Group {g}' for g in GROUPS])
ax2.set_ylabel('Count / Power')
ax2.set_title('Group Sizes and Charging Rates')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Print detailed information
print("\n=== Bus Charging Assignment Details ===")
print(bus_data.to_string(index=False))

# After solving, add verification
print("\n=== Charging Schedule Verification ===")
violations = bus_data[bus_data['Completion_Time'] > END_TIME]
if not violations.empty:
    print("Warning: Some buses will not complete charging by 4 AM:")
    for _, row in violations.iterrows():
        print(f"Bus {row['Bus']} in Group {row['Group']} completes at {row['Completion_Time_Str']}")
    print("\nIncreasing charging rates to meet deadline...")
    
    # Recalculate minimum required rates
    def calculate_min_charging_rate(start_time, total_energy):
        remaining_time = END_TIME - start_time
        return total_energy / remaining_time

    for g in GROUPS:
        group_buses = bus_data[bus_data['Group'] == g]
        if not group_buses.empty:
            total_energy = group_buses['Energy_Required'].sum()
            min_rate = calculate_min_charging_rate(GROUP_START[g], total_energy)
            r_values[g] = max(r_values[g], min_rate)
            print(f"Group {g}: Adjusted rate to {r_values[g]:.1f} kW")
else:
    print("All buses will complete charging by 4 AM")

# Return summary info
value(P_peak), group_sizes, r_values
