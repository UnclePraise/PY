import pandas as pd
import matplotlib.pyplot as plt

excel_file = 'Bus SOC Load Profile Worked on.xlsx'
sheet_names = ['12tha']

for sheet in sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet)
    # If 'Time' is only in HH:MM, add a date
    if df['Time'].dtype == object or pd.api.types.is_string_dtype(df['Time']):
        df['Time'] = '2025-06-12 ' + df['Time'].astype(str)
    df['Time'] = pd.to_datetime(df['Time'])
    time_str = df['Time'].dt.strftime('%H:%M')
    soc = df['Soc']
    plt.figure(figsize=(10, 6))
    plt.plot(time_str, soc, marker='o')
    plt.xlabel('Time')
    plt.ylabel('State of Charge (SOC)')
    plt.title(f'Bus SOC vs Time ({sheet})')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()