import random
from datetime import datetime, timedelta

# Constants
NUM_BUSES = 20
SLOT_DURATION_MIN = 30
SLOTS_START = datetime.strptime("22:00", "%H:%M")
NUM_SLOTS = 5
BATTERY_RANGE_KM = 253  # km
BATTERY_SIZE_KWH = 230  # Battery size at 100% SOC in kWh
MIN_SOC = 25
MAX_SOC = 40
EARLIEST_ARRIVAL_TIME = "21:00"
LATEST_ARRIVAL_TIME = "23:50"
EARLIEST_DEPARTURE_TIME = "04:00"
LATEST_DEPARTURE_TIME = "08:00"

# Create charging slots
charging_slots = [(SLOTS_START + timedelta(minutes=i*SLOT_DURATION_MIN),
                   SLOTS_START + timedelta(minutes=(i+1)*SLOT_DURATION_MIN)) for i in range(NUM_SLOTS)]

# Helper to generate random time between two times
def random_time(start_str, end_str):
    start = datetime.strptime(start_str, "%H:%M")
    end = datetime.strptime(end_str, "%H:%M")
    random_min = random.randint(0, int((end - start).total_seconds() / 60))
    return start + timedelta(minutes=random_min)

# Generate Bus Personas
buses = []
for i in range(NUM_BUSES):
    bus_id = f"Bus_{i+1:02d}"
    # Generate random arrival time, state of charge (SOC), kilometers driven, departure time and needed kWh
    arrival_time = random_time(EARLIEST_ARRIVAL_TIME, LATEST_ARRIVAL_TIME)
    soc = round(random.uniform(MIN_SOC, MAX_SOC), 1)  # in percentage
    km_driven = round((100 - soc) / 100 * BATTERY_RANGE_KM, 1)
    departure_time = random_time(EARLIEST_DEPARTURE_TIME, LATEST_DEPARTURE_TIME)
    needed_kwh = round((100 - soc) / 100 * BATTERY_SIZE_KWH, 1)  # kWh needed to reach 100% SOC

    buses.append({
        "id": bus_id,
        "arrival": arrival_time,
        "soc": soc,
        "km_driven": km_driven,
        "departure": departure_time,
        "needed_kwh": needed_kwh
    })

# Shuffle buses
random.shuffle(buses)

# Assign buses to slots based on departure time
slot_assignments = []
available_slots = list(range(NUM_SLOTS))  # Keep track of available slots

for bus in buses:
    best_slot = None
    best_slot_idx = None
    min_diff = float('inf')

    for idx in available_slots:
        start, end = charging_slots[idx]
        # Calculate the time difference between the slot end and the bus departure time
        diff = (bus["departure"] - end).total_seconds()
        # Assign the bus to the slot that minimizes the time difference
        if 0 <= diff < min_diff and bus["arrival"] <= start - timedelta(minutes=10):
            min_diff = diff
            best_slot = (start, end)
            best_slot_idx = idx

    if best_slot:
        slot_assignments.append({
            "bus_id": bus["id"],
            "slot": f"{best_slot[0].strftime('%I:%M %p')} – {best_slot[1].strftime('%I:%M %p')}"
        })
        if best_slot_idx is not None:
            available_slots.remove(best_slot_idx)  # Mark slot as unavailable

# Sort buses by arrival time
buses.sort(key=lambda bus: bus["arrival"])

# Sort slot assignments by slot start time in ascending order
slot_assignments.sort(key=lambda assignment: datetime.strptime(assignment["slot"].split(" – ")[0], "%I:%M %p"))

# Print bus personas
print("=== BUS PERSONAS ===")
for bus in buses:
    print(f"{bus['id']} | Arrival: {bus['arrival'].strftime('%I:%M %p')} | SOC: {bus['soc']}% | "
          f"Km Driven: {bus['km_driven']}km | Departure: {bus['departure'].strftime('%I:%M %p')} | "
          f"Needed kWh: {bus['needed_kwh']} kWh")

# Print slot assignments
print("\n=== SLOT ASSIGNMENTS ===")
for assignment in slot_assignments:
    print(f"{assignment['bus_id']} → {assignment['slot']}")

# Calculate total kWh needed for each slot
slot_totals = {}
for assignment in slot_assignments:
    slot = assignment["slot"]
    bus_id = assignment["bus_id"]
    bus = next(bus for bus in buses if bus["id"] == bus_id)
    if slot not in slot_totals:
        slot_totals[slot] = 0
    slot_totals[slot] += bus["needed_kwh"]

# Print slot assignments with total kWh needed
print("\n=== SLOT ASSIGNMENTS WITH TOTAL kWh ===")
for slot, total_kwh in slot_totals.items():
    print(f"{slot} → Total kWh Needed: {total_kwh} kWh")
