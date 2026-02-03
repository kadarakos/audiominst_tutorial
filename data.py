import json
import random
import torch
import soundfile

import torchaudio

from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Callable

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


@dataclass
class AudioMNISTSample:
    wav: torch.Tensor
    sr: int
    digit: int
    gender: Literal[0, 1]
    age: int
    accent: str
    native: bool
    speaker: str

    def __len__(self):
        return len(self.wav)


def collate_fn_digit(batch: list[AudioMNISTSample]):
    wavs, labels = zip(*[(item.wav.squeeze(), item.digit) for item in batch])
    return pad_sequence(wavs, batch_first=True), labels


def collate_fn_gender(batch: list[AudioMNISTSample]):
    wavs, labels = zip(*[(item.wav.squeeze(), item.gender) for item in batch])
    targets = torch.tensor(labels, dtype=torch.float32)
    return pad_sequence(wavs, batch_first=True), targets


def load_audio(path):
    wav, sr = soundfile.read(path, dtype="float32")

    # (T,) or (T, C) → (1, T)
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav.T

    return torch.from_numpy(wav), sr


def read_audio_mnist(
    path="AudioMNIST",
    n_test_speakers: int = 5,
    resample: Literal[32000, 16000, 8000] | None = None,
    batch_size: int = 32,
    collator: Callable = collate_fn_gender,
) -> tuple[DataLoader, DataLoader]:
    home_dir = Path(path)
    data_dir = home_dir / "data"
    meta_dir = data_dir / "audioMNIST_meta.txt"
    speaker_meta = json.loads((data_dir / "audioMNIST_meta.txt").read_text())
    n_speakers = len(speaker_meta)
    test_speakers = random.sample(
        list(speaker_meta.keys()),
        k=n_test_speakers
    )
    train_data = []
    test_data = []
    desc = "Loading speaker data"
    pbar = tqdm(
        data_dir.iterdir(),
        desc=desc,
        total=n_speakers,
        colour="green",
        ascii="<O",
    )
    for speaker in pbar:
        if speaker == meta_dir:
            continue
        speaker_id = speaker.name
        speaker_data = speaker_meta[speaker_id]
        for audio_path in speaker.iterdir():
            digit, _, _ = audio_path.name.split("_")
            wav, sr = load_audio(audio_path)
            if resample is not None:
                wav = torchaudio.functional.resample(
                    wav, orig_freq=sr, new_freq=resample
                )
            sample = AudioMNISTSample(
                wav=wav,
                sr=sr,
                speaker=speaker_id,
                digit=int(digit),
                accent=speaker_data["accent"],
                native=speaker_data["native speaker"] == "yes",
                age=int(speaker_data["age"]),
                gender=int(speaker_data["gender"] == "female"),
            )
            if speaker_id in test_speakers:
                test_data.append(sample)
            else:
                train_data.append(sample)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=collator,
    )
    return train_loader, test_loader
