import csv
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as animation

AXIS_LENGTH = 81  # in minutes
CSV_FILE = "../DB_epoch_time.csv"

def parse_csv(file_path):
    next_epoch = []
    next_epoch_times = []
    timestamps = []

    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            try:
                timestamp_str = row['Timestamp']
                next_epoch_time_str = row['Next Epoch Time'].strip()
                next_epoch_number = row['Next Epoch'].strip()

                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %I:%M:%S %p')
                next_time = datetime.strptime(next_epoch_time_str, '%I:%M:%S %p').time()
                next_epoch_datetime = datetime.combine(timestamp.date(), next_time)

                timestamps.append(timestamp)
                next_epoch_times.append(next_epoch_datetime)
                next_epoch.append(next_epoch_number)

            except Exception as e:
                print(f"Skipping row due to error: {e}")
    return timestamps, next_epoch_times, next_epoch

# Initialize figure
fig, ax = plt.subplots(figsize=(16, 7))
plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15)  # Add width control

def update(frame):
    ax.clear()

    # Re-style plot after clearing
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    ax.tick_params(bottom=False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(True)

    # Parse updated data
    timestamps, next_epoch_times, next_epoch_labels = parse_csv(CSV_FILE)
    if not timestamps or not next_epoch_times:
        return

    end_time = datetime.now()
    start_time = end_time - timedelta(minutes=AXIS_LENGTH)

    # X ticks every 30 mins
    time_ticks = []
    tick = start_time.replace(minute=(start_time.minute // 30) * 30, second=0, microsecond=0)
    while tick <= end_time:
        time_ticks.append(tick)
        tick += timedelta(minutes=30)

    for i, time in enumerate(next_epoch_times):
        if start_time <= time <= end_time:
            ax.axvline(x=time, color='blue', linestyle='--', linewidth=1)
            label_text = f"{next_epoch_labels[i]}\n{time.strftime('%I:%M:%S %p')}"
            ax.text(time, -0.03, label_text, rotation=90,
                    verticalalignment='top', horizontalalignment='center',
                    fontsize=8, transform=ax.get_xaxis_transform())

    ax.set_xticks(time_ticks)
    ax.set_xticklabels([t.strftime('%I:%M %p') for t in time_ticks], rotation=0)
    ax.set_xlim(start_time, end_time)

# Animate every 5 seconds
ani = animation.FuncAnimation(fig, update, interval=5000, cache_frame_data=False)

plt.show()
