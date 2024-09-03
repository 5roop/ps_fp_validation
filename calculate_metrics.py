from sklearn.metrics import classification_report
import krippendorff
from utils import is_overlapping
import pandas as pd
import numpy as np
from utils import intervals_to_frames

df = pd.read_json("y_pred_y_true.jsonl", lines=True)

print(
    f"Laura annotated {df.laura.notna().sum()} instances, Lara annotated {df.lara.notna().sum()} instances."
)
# Inter-annotator agreement
iaa = df[df.lara.notna() & df.laura.notna()].copy()
iaa["lara_frames"] = iaa.apply(
    lambda row: intervals_to_frames(row["lara"], duration_ms=row["duration"]), axis=1
)
iaa["laura_frames"] = iaa.apply(
    lambda row: intervals_to_frames(row["laura"], duration_ms=row["duration"]), axis=1
)
iaa["y_pred_frames"] = iaa.apply(
    lambda row: intervals_to_frames(row["y_pred"], duration_ms=row["duration"]), axis=1
)
lara_frames = [i for j in iaa.lara_frames.values for i in j]
laura_frames = [i for j in iaa.laura_frames.values for i in j]

data = np.array([lara_frames, laura_frames])


print("Krippendorff alpha on frame-by-frame level: ", krippendorff.alpha(data))

lara_events, laura_events = [], []
for i, row in iaa.iterrows():
    lara = row["lara"]
    laura = row["laura"]
    if (lara == []) and (laura == []):
        lara_events.append(0)
        laura_events.append(0)
    for l in lara + laura:
        if any([is_overlapping(l, x) for x in lara + laura if x != l]):
            lara_events.append(1)
            laura_events.append(1)
        else:
            if l in lara:
                lara_events.append(1)
                laura_events.append(0)
            else:
                laura_events.append(1)
                lara_events.append(0)
print(
    "Krippendorff alpha on event level: ",
    krippendorff.alpha(np.array([lara_events, laura_events])),
)

# Performance of our classifier on Laura:

subset = df[df.laura.notna()].copy()
subset["laura_frames"] = subset.apply(
    lambda row: intervals_to_frames(row["laura"], duration_ms=row["duration"]), axis=1
)
subset["y_pred_frames"] = subset.apply(
    lambda row: intervals_to_frames(row["y_pred"], duration_ms=row["duration"]), axis=1
)
y_pred_frames = [i for j in subset.y_pred_frames.values for i in j]
laura_frames = [i for j in subset.laura_frames.values for i in j]


print(
    "Classification report for Laura vs y_pred on frame-by-frame level:",
    classification_report(laura_frames, y_pred_frames),
    sep="\n",
)

y_pred_events, laura_events = [], []
for i, row in subset.iterrows():
    y_pred = row["y_pred"]
    laura = row["laura"]
    if (y_pred == []) and (laura == []):
        y_pred_events.append(0)
        laura_events.append(0)
    for l in y_pred + laura:
        if any([is_overlapping(l, x) for x in y_pred + laura if x != l]):
            y_pred_events.append(1)
            laura_events.append(1)
        else:
            if l in y_pred:
                y_pred_events.append(1)
                laura_events.append(0)
            else:
                laura_events.append(1)
                y_pred_events.append(0)
print(
    "Classification report for Laura vs y_pred on event level: ",
    classification_report(laura_events, y_pred_events),
    sep="\n",
)



subset = df[df.lara.notna()].copy()
subset["lara_frames"] = subset.apply(
    lambda row: intervals_to_frames(row["lara"], duration_ms=row["duration"]), axis=1
)
subset["y_pred_frames"] = subset.apply(
    lambda row: intervals_to_frames(row["y_pred"], duration_ms=row["duration"]), axis=1
)
y_pred_frames = [i for j in subset.y_pred_frames.values for i in j]
lara_frames = [i for j in subset.lara_frames.values for i in j]


print(
    "Classification report for Lara vs y_pred on frame-by-frame level:",
    classification_report(lara_frames, y_pred_frames),
    sep="\n",
)

y_pred_events, lara_events = [], []
for i, row in subset.iterrows():
    y_pred = row["y_pred"]
    lara = row["lara"]
    if (y_pred == []) and (lara == []):
        y_pred_events.append(0)
        lara_events.append(0)
    for l in y_pred + lara:
        if any([is_overlapping(l, x) for x in y_pred + lara if x != l]):
            y_pred_events.append(1)
            lara_events.append(1)
        else:
            if l in y_pred:
                y_pred_events.append(1)
                lara_events.append(0)
            else:
                lara_events.append(1)
                y_pred_events.append(0)
print(
    "Classification report for Lara vs y_pred on event level: ",
    classification_report(lara_events, y_pred_events),
    sep="\n",
)

# Export the TP, FP, FN:
from pydub import AudioSegment
from pathlib import Path

TP, FN, FP = [], [], []
for i, row in subset.iterrows():
    y_pred = row["y_pred"]
    y_true = row["laura"]
    found_audio = list(Path(".").glob(f"**/{row['name']}.wav"))[0]
    AS = AudioSegment.from_file(found_audio)
    while y_pred + y_true:
        event = (y_pred + y_true)[0]
        if any([is_overlapping(event, i) for i in y_pred + y_true if i != event]):
            TP.append(
                AS[event[0] - 400 : event[1] + 400] + AudioSegment.silent(duration=300)
            )

            y_pred = [
                i for i in y_pred if (i != event) and (not is_overlapping(event, i))
            ]
            y_true = [
                i for i in y_true if (i != event) and (not is_overlapping(event, i))
            ]
            continue
        else:
            if event in y_pred:
                FP.append(
                    AS[event[0] - 400 : event[1] + 400]
                    + AudioSegment.silent(duration=300)
                )
                y_pred = [i for i in y_pred if i != event]
                continue
            else:  # Event is in y_true
                FN.append(
                    AS[event[0] - 400 : event[1] + 400]
                    + AudioSegment.silent(duration=300)
                )
                y_true = [i for i in y_true if i != event]
                continue

if TP:
    sum(TP).export("TP.mp3", format="mp3")
if FN:
    sum(FN).export("FN.mp3", format="mp3")
if FP:
    sum(FP).export("FP.mp3", format="mp3")
2 + 2
