outfile = "y_pred.jsonl"

from transformers import AutoFeatureExtractor, Wav2Vec2BertForAudioFrameClassification
from datasets import load_dataset, Dataset, Audio
import torch
import numpy as np
import soundfile as sf
import tqdm
import os
import numpy as np
import os
import pandas as pd
import datasets
from datasets import load_dataset, load_metric, Audio
from itertools import zip_longest, pairwise
from pathlib import Path

os.environ["WANDB_DISABLED"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")
TARGET = "filledPause"
checkpoint = Path(
    f"../mezzanine_resources/filled_pauses/model_{TARGET}_3e-5_20_4/checkpoint-900"
)
feature_extractor = AutoFeatureExtractor.from_pretrained(str(checkpoint))
model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(str(checkpoint)).to(
    device
)
wavs = Path(".").glob("**/*.wav")

df = pd.DataFrame(data={"audio": [i for i in wavs]})
df["name"] = df.audio.apply(lambda p: p.name)
df = df.drop_duplicates(subset="name").reset_index()
df["audio"] = df.audio.apply(str)
df = df.drop(columns="name")

ds = datasets.Dataset.from_pandas(df)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000, mono=True))

from utils import frames_to_intervals


def evaluator(chunks):
    sampling_rate = chunks["audio"][0]["sampling_rate"]
    with torch.no_grad():
        inputs = feature_extractor(
            [i["array"] for i in chunks["audio"]],
            return_tensors="pt",
            sampling_rate=sampling_rate,
        ).to(device)
        logits = model(**inputs).logits
    y_pred = np.array(logits.cpu()).argmax(axis=-1)
    return {"y_pred": [frames_to_intervals(i) for i in y_pred]}


ds = ds.map(evaluator, batch_size=30, batched=True, desc="Running inference")
df["y_pred"] = [i for i in ds["y_pred"]]


df.to_json(outfile, orient="records", lines=True)
