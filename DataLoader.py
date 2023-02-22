import torch
import torchaudio
import torchaudio.functional as F
from torch.utils.data import Dataset

import sqlite3

import os
import math
import pandas as pd
from IPython.display import Audio
import matplotlib.pyplot as plt


class InstrumentsDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transform, target_sample_rate, num_samples, device):
        conn = sqlite3.connect(annotations_file)
        self.annotations = pd.read_sql_query('SELECT * from takes', conn)
        conn.close()

        self.audio_dir = audio_dir
        self.device = device
        self.transform = transform.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, item):
        audio_pth = self._get_audio_sample_path(item)
        label = self._get_audio_sample_label(item)
        signal, sr = torchaudio.load(audio_pth)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)  # Zero padding-method
        signal = self.transform(signal)
        return signal, label

    def _get_audio_sample_path(self, item):
        # fold = f"fold{self.annotations.iloc[item, ]}"

        # path = os.path.join(self.audio_dir, fold, self.annotations.iloc[item, 13])

        #   INSTRUMENT_PERSON_RECORDINGS
        path = os.path.join(self.audio_dir, self.annotations.iloc[item, 2])
        return path

    def _get_audio_sample_label(self, item):
        return self.annotations.iloc[item, 5]

    #   Make a 1-dim signal(num_chann, sr)
    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_miss_sampl = self.num_samples - length_signal
            last_dim_padding = (0, num_miss_sampl)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _cut_if_necessary(self, signal):
        # signal -> Tensor -> (1, num_samples)
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal


if __name__ == "__main__":
    ANNOTATIONS_FILE = "/home/pablo/Documents/good-sounds/database.sqlite"
    AUDIO_DIR = "/home/pablo/Documents/good-sounds"
    SAMPLE_RATE = 48000
    NUM_SAMPLES = 48000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)

    isd = InstrumentsDataset(ANNOTATIONS_FILE, AUDIO_DIR, mel_spec, SAMPLE_RATE, NUM_SAMPLES, device)

    print(f"There are {len(isd)} samples in the dataset.")

    signal, label = isd[0]
