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
CHARGING_POWER_KW = 150  # Charging power in kW

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
    charging_time_needed = round(needed_kwh / CHARGING_POWER_KW * 60)  # Convert to minutes

    buses.append({
        "id": bus_id,
        "arrival": arrival_time,
        "soc": soc,
        "km_driven": km_driven,
        "departure": departure_time,
        "needed_kwh": needed_kwh,
        "charging_time": charging_time_needed
    })

# Shuffle buses and assign to available slots
random.shuffle(buses)
slot_assignments = []

for bus in buses:
    for idx, (start, end) in enumerate(charging_slots):
        if bus["arrival"] <= start - timedelta(minutes=10):
            slot_assignments.append({
                "bus_id": bus["id"],
                "slot": f"{start.strftime('%I:%M %p')} – {end.strftime('%I:%M %p')}",
                "charging_time": bus["charging_time"]
            })
            break

buses.sort(key=lambda bus: bus["arrival"])
slot_assignments.sort(key=lambda assignment: datetime.strptime(assignment["slot"].split(" – ")[0], "%I:%M %p"))

print("=== BUS PERSONAS ===")
for bus in buses:
    print(f"{bus['id']} | Arrival: {bus['arrival'].strftime('%I:%M %p')} | SOC: {bus['soc']}% | "
          f"Km Driven: {bus['km_driven']}km | Departure: {bus['departure'].strftime('%I:%M %p')} | "
          f"Needed kWh: {bus['needed_kwh']} kWh | Charging Time: {bus['charging_time']} min")

print("\n=== SLOT ASSIGNMENTS ===")
for assignment in slot_assignments:
    print(f"{assignment['bus_id']} → {assignment['slot']} (Required charging time: {assignment['charging_time']} min)")

slot_totals = {}
for assignment in slot_assignments:
    slot = assignment["slot"]
    bus_id = assignment["bus_id"]
    bus = next(bus for bus in buses if bus["id"] == bus_id)
    if slot not in slot_totals:
        slot_totals[slot] = {"kwh": 0, "charging_time": 0}
    slot_totals[slot]["kwh"] += bus["needed_kwh"]
    slot_totals[slot]["charging_time"] += bus["charging_time"]

print("\n=== SLOT ASSIGNMENTS WITH TOTAL kWh AND CHARGING TIME ===")
for slot, totals in slot_totals.items():
    print(f"{slot} → Total kWh Needed: {totals['kwh']} kWh | Total Charging Time: {totals['charging_time']} min")