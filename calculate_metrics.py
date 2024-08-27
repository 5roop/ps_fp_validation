import pandas as pd
import numpy as np
from utils import intervals_to_frames

df = pd.read_json("y_pred_y_true.jsonl", lines=True)


# Inter-annotator agreement
df = df[df.lara.notna() & df.laura.notna()]
df["lara_frames"] = df.apply(
    lambda row: intervals_to_frames(row["lara"], duration_ms=row["duration"]), axis=1
)
df["laura_frames"] = df.apply(
    lambda row: intervals_to_frames(row["laura"], duration_ms=row["duration"]), axis=1
)
df["y_pred_frames"] = df.apply(
    lambda row: intervals_to_frames(row["y_pred"], duration_ms=row["duration"]), axis=1
)
lara_frames = [i for j in df.lara_frames.values for i in j]
laura_frames = [i for j in df.laura_frames.values for i in j]

data = np.array([lara_frames, laura_frames])

import krippendorff

print("Krippendorff alpha on frame-by-frame level: ", krippendorff.alpha(data))
from utils import is_overlapping

lara_events, laura_events = [], []
for i, row in df.iterrows():
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
                lara.append(1)
                laura.append(0)
            else:
                laura.append(1)
                lara.append(0)
print(
    "Krippendorff alpha on event level: ",
    krippendorff.alpha(np.array([lara_events, laura_events])),
)

# Performance of our classifier:

from sklearn.metrics import classification_report

# df = pd.read_json("y_pred_y_true.jsonl", lines=True)
# df["lara_frames"] = df.apply(
#     lambda row: intervals_to_frames(row["lara"], duration_ms=row["duration"]), axis=1
# )
# df["laura_frames"] = df.apply(
#     lambda row: intervals_to_frames(row["laura"], duration_ms=row["duration"]), axis=1
# )
# df["y_pred_frames"] = df.apply(
#     lambda row: intervals_to_frames(row["y_pred"], duration_ms=row["duration"]), axis=1
# )


2 + 2
