import pandas as pd
import numpy as np

# ============
# PARAMETERS
# ============
NUM_BUSES = 20
BATTERY_CAPACITY_KWH = 230  # kWh (for information; used to compute km driven)
FULL_RANGE_KM = 230         # km at 100% SOC
# Time boundaries for bus persona generation
arrival_earliest = pd.Timestamp("2025-04-14 21:00")   # 9:00 PM arrival minimum (for early arrivals)
arrival_latest   = pd.Timestamp("2025-04-14 23:30")   # 11:30 PM latest arrival
departure_earliest = pd.Timestamp("2025-04-15 04:00")   # 4:00 AM earliest departure
departure_latest   = pd.Timestamp("2025-04-15 08:00")   # 8:00 AM latest departure

# ============
# BUS PERSONA GENERATION
# ============
# Uncomment the next line for reproducible output; leave commented for true randomness
# np.random.seed(42)

# Generate random arrival times (rounded down to nearest 30min)
arrival_times = pd.to_datetime(np.random.uniform(arrival_earliest.value, arrival_latest.value, NUM_BUSES)).floor('30min')

# Generate random State-of-Charge (SOC) between 25% and 40%
arrival_socs = np.round(np.random.uniform(0.25, 0.40, NUM_BUSES), 2)

# Estimate KM driven based on SOC left: 100% => 230km
kms_driven = np.round((1 - arrival_socs) * FULL_RANGE_KM, 1)

# Generate random departure times (rounded to nearest 30min)
departure_times = pd.to_datetime(np.random.uniform(departure_earliest.value, departure_latest.value, NUM_BUSES)).floor('30min')

# Assemble DataFrame for Bus Personas
bus_personas = pd.DataFrame({
    "Bus ID": range(1, NUM_BUSES + 1),
    "Arrival Time": arrival_times,
    "SOC at Arrival (%)": (arrival_socs * 100).round(1),
    "KM Driven": kms_driven,
    "Departure Time": departure_times
})

print("🚌 EV Bus Charging Personas:\n")
print(bus_personas.to_string(index=False))

# ============
# SCHEDULE ALLOCATION
# ============
# We have 5 slots:
#   Slot 1: 10:00 PM – 10:30 PM
#   Slot 2: 10:30 PM – 11:00 PM
#   Slot 3: 11:00 PM – 11:30 PM
#   Slot 4: 11:30 PM – 12:00 AM
#   Slot 5: 12:00 AM – 12:30 AM
#
# And exactly 4 buses per slot (in this good case scenario).

# Define the slot labels with time intervals
slot_labels = {
    1: "10:00 PM – 10:30 PM",
    2: "10:30 PM – 11:00 PM",
    3: "11:00 PM – 11:30 PM",
    4: "11:30 PM – 12:00 AM",
    5: "12:00 AM – 12:30 AM"
}

# Shuffle the bus IDs (we'll use the bus_personas DataFrame index order)
shuffled_personas = bus_personas.sample(frac=1, random_state=42).reset_index(drop=True)

# Prepare an empty list for assigned slots
assigned_slots = []

# Since we have 20 buses and 5 slots, assign every 4 buses to each successive slot.
buses_per_slot = 4
for i in range(NUM_BUSES):
    # Determine the slot: integer division gives slot index starting at 0, add 1 for slot number
    slot_number = (i // buses_per_slot) + 1  
    assigned_slots.append(slot_number)

# Attach the assigned slot to the DataFrame
shuffled_personas["Assigned Slot"] = assigned_slots

# Replace slot numbers with slot time intervals for readability
shuffled_personas["Assigned Slot"] = shuffled_personas["Assigned Slot"].map(slot_labels)

print("\n📅 Bus Schedule Allocation:\n")
print(shuffled_personas[["Bus ID", "Assigned Slot"]].to_string(index=False))

# Optional: Save the schedule to CSV
# shuffled_personas.to_csv("bus_schedule_allocation.csv", index=False)
# print("\n✅ Schedule allocation saved to 'bus_schedule_allocation.csv'")
