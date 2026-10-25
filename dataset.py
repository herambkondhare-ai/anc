import os
import random
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NoiseReductionDataset(Dataset):

    def __init__(
        self,
        clean_dir,
        noise_dir,
        sample_rate=16000,
        segment_length=32000
    ):
        self.clean_dir = clean_dir
        self.noise_dir = noise_dir
        self.sample_rate = sample_rate
        self.segment_length = segment_length

        self.clean_files = [
            os.path.join(clean_dir, f)
            for f in os.listdir(clean_dir)
            if f.lower().endswith(".wav")
        ]

        self.noise_files = [
            os.path.join(noise_dir, f)
            for f in os.listdir(noise_dir)
            if f.lower().endswith(".wav")
        ]

        if len(self.clean_files) == 0:
            raise RuntimeError("No clean WAV files found.")

        if len(self.noise_files) == 0:
            raise RuntimeError("No noise WAV files found.")

        print(f"Clean files: {len(self.clean_files)}")
        print(f"Noise files: {len(self.noise_files)}")

    def __len__(self):
        return max(len(self.clean_files), len(self.noise_files))

    def load_audio(self, path):

        audio, sr = sf.read(path, dtype="float32")

        # Convert stereo/multichannel to mono
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Resample if necessary
        if sr != self.sample_rate:
            audio = resample_poly(
                audio,
                self.sample_rate,
                sr
            ).astype(np.float32)

        return torch.from_numpy(audio)

    def fix_length(self, audio):

        length = audio.shape[0]

        if length < self.segment_length:

            pad_amount = self.segment_length - length

            audio = F.pad(
                audio,
                (0, pad_amount)
            )

        elif length > self.segment_length:

            start = random.randint(
                0,
                length - self.segment_length
            )

            audio = audio[
                start:start + self.segment_length
            ]

        return audio

    def __getitem__(self, idx):

        # -------------------------
        # CLEAN SPEECH
        # -------------------------

        clean_path = self.clean_files[
            idx % len(self.clean_files)
        ]

        clean = self.load_audio(clean_path)
        clean = self.fix_length(clean)

        # -------------------------
        # NOISE
        # -------------------------

        noise_path = random.choice(
            self.noise_files
        )

        noise = self.load_audio(noise_path)
        noise = self.fix_length(noise)

        # -------------------------
        # RANDOM SNR
        # -------------------------

        snr_db = random.uniform(-5.0, 15.0)

        clean_power = torch.mean(clean ** 2)
        noise_power = torch.mean(noise ** 2)

        # Prevent division by zero
        if noise_power < 1e-10:
            noise = torch.randn_like(clean)
            noise_power = torch.mean(noise ** 2)

        scale = torch.sqrt(
            clean_power /
            (noise_power * (10 ** (snr_db / 10)))
        )

        noise = noise * scale

        # -------------------------
        # MIX
        # -------------------------

        noisy = clean + noise

        # Prevent clipping
        max_value = torch.max(torch.abs(noisy))

        if max_value > 0.99:
            noisy = noisy / max_value * 0.99

        return noisy.unsqueeze(0), clean.unsqueeze(0)