import pandas as pd
from pathlib import Path
import soundfile as sf

df = pd.read_json("y_pred.jsonl", lines=True)
df["name"] = df.audio.apply(lambda s: Path(s).with_suffix("").name)


def find_and_extract(name: str, annotator: str):
    found_eaf = list(Path(f"{annotator}/results").glob(f"**/{name}.eaf"))
    if len(found_eaf) == 0:
        return pd.NA

    from utils import extract_annotations

    return extract_annotations(found_eaf[0])


df["lara"] = df.name.apply(lambda name: find_and_extract(name, annotator="lara"))
df["laura"] = df.name.apply(lambda name: find_and_extract(name, annotator="laura"))


def get_audio_length(s: str):
    array, sampling_rate = sf.read(s)
    return int(len(array) / sampling_rate * 1000)


df["duration"] = df.audio.apply(get_audio_length)
df = df.drop(columns="audio")

df.to_json("y_pred_y_true.jsonl", orient="records", lines=True)
2 + 2
