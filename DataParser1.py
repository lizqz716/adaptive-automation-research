import pandas as pd
from datetime import datetime
import re
import os

parent_output_folder = "/Users/erick/OneDrive/Documents/Airsim/PythonClient/Parsed_csv_Files"

BASELINE_DURATION = 15.0    # seconds

def clean_value(val):

    if pd.isna(val):
        return None

    val = str(val).replace("(", "").replace(")", "")

    val = val.strip()

    parts = val.split(",")

    for p in parts:
        p = p.strip()
        if p:
            p = re.sub(r"[^0-9.+-eE]", "", p)
            try:
                return float(p)
            except:
                return None
    return None

def process_csv(filepath):

    df = pd.read_csv(filepath)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    start_time = df["timestamp"].iloc[0]
    df["time_seconds"] = (df["timestamp"] - start_time).dt.total_seconds()

    file_name = os.path.basename(filepath)

    os.makedirs(parent_output_folder, exist_ok=True)

    output_folder = os.path.join(parent_output_folder, f"run_{file_name}")
    os.makedirs(output_folder, exist_ok=True)



    df["clean_value"] = df["value"].apply(clean_value)

    hr_df = df[df["variable"] == "HR"][["time_seconds", "clean_value"]]
    eda_df = df[df["variable"] == "EDA"][["time_seconds", "clean_value"]]

    hr_path = os.path.join(output_folder, "hr_clean.csv")
    eda_path = os.path.join(output_folder, "eda_clean.csv")

    hr_df.to_csv(hr_path, index=False)
    eda_df.to_csv(eda_path, index=False)

    print("Saved hr_clean.csv and eda_clean.csv")

    return hr_path, eda_path, file_name

user_input = input("Input CSV File Directory:")

clean_text = user_input.strip('"') 

hrpath, edapath, filename = process_csv(clean_text)


output_folder = os.path.join(parent_output_folder, f"run_{filename}")

output_log_path = os.path.join(output_folder, "marker_events.csv")

hr_df = pd.read_csv(hrpath)
eda_df = pd.read_csv(edapath)

hr_baseline = hr_df[hr_df["time_seconds"] <= BASELINE_DURATION]["clean_value"].mean()
eda_baseline = eda_df[eda_df["time_seconds"] <= BASELINE_DURATION]["clean_value"].mean()

hr_threshold = hr_baseline + 9
eda_threshold = eda_baseline + 0.007

events = []

for _, row in hr_df.iterrows():
    events.append(("HR", row["time_seconds"], row["clean_value"]))

for _, row in eda_df.iterrows():
    events.append(("EDA", row["time_seconds"], row["clean_value"]))

events.sort(key=lambda x: x[1])

hr_active = False
eda_active = False

log_rows = []

for signal, time_s, value in events:

    if signal == "HR":
        if not hr_active and value >= hr_threshold:
            hr_active = True
            print(f"[{time_s:.2f}s] HR MARKER ACTIVATED")

            log_rows.append({
                "signal": "HR",
                "event": "ACTIVATED",
                "time_seconds": time_s,
                "value": value,
                "baseline": hr_baseline,
                "threshold": hr_threshold
            })

        elif hr_active and value < hr_threshold:
            hr_active = False
            print(f"[{time_s:.2f}s] HR MARKER DEACTIVATED")

            log_rows.append({
                "signal": "HR",
                "event": "DEACTIVATED",
                "time_seconds": time_s,
                "value": value,
                "baseline": hr_baseline,
                "threshold": hr_threshold
            })

    elif signal == "EDA":
        if not eda_active and value >= eda_threshold:
            eda_active = True
            print(f"[{time_s:.2f}s] EDA MARKER ACTIVATED")

            log_rows.append({
                "signal": "EDA",
                "event": "ACTIVATED",
                "time_seconds": time_s,
                "value": value,
                "baseline": eda_baseline,
                "threshold": eda_threshold
            })

        elif eda_active and value < eda_threshold:
            eda_active = False
            print(f"[{time_s:.2f}s] EDA MARKER DEACTIVATED")

            log_rows.append({
                "signal": "EDA",
                "event": "DEACTIVATED",
                "time_seconds": time_s,
                "value": value,
                "baseline": eda_baseline,
                "threshold": eda_threshold
            })

log_df = pd.DataFrame(log_rows)
log_df.to_csv(output_log_path, index=False)

print(f"\nMarker events saved to {output_log_path}")