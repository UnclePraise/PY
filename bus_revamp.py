import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta, time
random.seed(42) 



NUM_BUSES = 20
NUM_SLOTS = 5

# Constants
SLOT_DURATION_MIN = 30
SLOTS_START = datetime.strptime("22:00", "%H:%M")
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

# Shuffle buses and assign to available slots
random.shuffle(buses)
slot_assignments = []

for bus in buses:
    for idx, (start, end) in enumerate(charging_slots):
        if bus["arrival"] <= start - timedelta(minutes=10):
            slot_assignments.append({
                "bus_id": bus["id"],
                "slot": f"{start.strftime('%I:%M %p')} – {end.strftime('%I:%M %p')}"
            })
            break  # Assign only to first possible slot

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

# Plot 1: Arrival Time vs SOC Level
fig_arrival, ax_arrival = plt.subplots(figsize=(10, 5))
fig_arrival.canvas.manager.set_window_title('Figure 1: Arrival Time vs SOC Level')
arrival_times = [bus['arrival'] for bus in buses]
soc_levels = [bus['soc'] for bus in buses]
bus_ids = [bus['id'] for bus in buses]
ax_arrival.scatter(arrival_times, soc_levels, color='green', label='Bus')
for x, y, label in zip(arrival_times, soc_levels, bus_ids):
    ax_arrival.annotate(label, (x, y), textcoords="offset points", xytext=(5,5), ha='left', fontsize=8)
ax_arrival.set_title('Bus Arrival Time vs SOC Level')
ax_arrival.set_xlabel('Arrival Time')
ax_arrival.set_ylabel('SOC (%)')
# ax_arrival.set_ylim(0, 100)  # Set SOC range from 0 to 100
ax_arrival.xaxis.set_major_formatter(mdates.DateFormatter('%I:%M %p'))
ax_arrival.legend()
fig_arrival.autofmt_xdate()
fig_arrival.tight_layout()


# Plot 2: Bus SOC Distribution
fig_soc, ax_soc = plt.subplots(figsize=(10, 5))
fig_soc.canvas.manager.set_window_title('Figure 2: Bus SOC Distribution')
soc_values = [bus['soc'] for bus in buses]
bus_ids = [bus['id'] for bus in buses]
ax_soc.bar(bus_ids, soc_values, label='SOC (%)')
ax_soc.set_title('Bus State of Charge (SOC) Distribution')
ax_soc.set_xlabel('Bus ID')
ax_soc.set_ylabel('SOC (%)')
ax_soc.set_ylim(0, 100)  # Set SOC range from 0 to 100
ax_soc.tick_params(axis='x', rotation=45)
ax_soc.legend()
fig_soc.tight_layout()


# Plot 3: Charging Slot Assignments and Total kWh
def slot_sort_key(slot_label):
    # Extract start time from slot label (e.g., "10:00 PM – 10:30 PM")
    start_str = slot_label.split('–')[0].strip()
    dt = datetime.strptime(start_str, "%I:%M %p")
    # If midnight, set a high value to sort it last
    if dt.time() == time(0, 0):
        return time(23, 59)
    return dt.time()

# Sort slots so that "12:00 AM – 12:30 AM" comes last
sorted_slots = sorted(slot_totals.keys(), key=slot_sort_key)
sorted_kwh_values = [slot_totals[slot] for slot in sorted_slots]

fig_slot, ax_slot = plt.subplots(figsize=(10, 5))
fig_slot.canvas.manager.set_window_title('Figure 3: Charging Slot Assignments')
ax_slot.bar(sorted_slots, sorted_kwh_values, label='Total kWh Needed')
ax_slot.set_title('Total kWh Needed per Charging Slot')
ax_slot.set_xlabel('Time Slot')
ax_slot.set_ylabel('Total kWh Needed')
ax_slot.tick_params(axis='x', rotation=45)
ax_slot.legend()
fig_slot.tight_layout()

plt.show()

