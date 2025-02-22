import logging

import numpy as np
import torch
import torchaudio
import torchvision
from torchvision.transforms import transforms
from torch.nn import functional as F

torchaudio.set_audio_backend("soundfile")

def torchaudio_loader(path):
    return torchaudio.load(path)

# def int16_to_float32_torch(x):
#     return (x / 32767.0).type(torch.float32)

# def float32_to_int16_torch(x):
#     x = torch.clamp(x, min=-1., max=1.)
#     return (x * 32767.).type(torch.int16)



class AudioTransform:
    def __init__(self, args):
        self.sample_rate = 16000
        self.num_mel_bins = 128
        self.target_length = args.target_length
        # self.audio_mean = args.audio_mean
        # self.audio_std = args.audio_std
        # self.mean = []
        # self.std = []

    def __call__(self, audio_data_and_origin_sr):
        audio_data, origin_sr = audio_data_and_origin_sr
        if self.sample_rate != origin_sr:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=origin_sr, new_freq=self.sample_rate)
        waveform_melspec = self.waveform2melspec(audio_data)
        return waveform_melspec


    def waveform2melspec(self, audio_data):
        mel = self.get_mel(audio_data)
        if mel.shape[0] > self.target_length:
            # split to three parts
            chunk_frames = self.target_length
            total_frames = mel.shape[0]
            ranges = np.array_split(list(range(0, total_frames - chunk_frames + 1)), 3)
            # print('total_frames-chunk_frames:', total_frames-chunk_frames,
            #       'len(audio_data):', len(audio_data),
            #       'chunk_frames:', chunk_frames,
            #       'total_frames:', total_frames)
            if len(ranges[1]) == 0:  # if the audio is too short, we just use the first chunk
                ranges[1] = [0]
            if len(ranges[2]) == 0:  # if the audio is too short, we just use the first chunk
                ranges[2] = [0]
            # randomly choose index for each part
            idx_front = np.random.choice(ranges[0])
            idx_middle = np.random.choice(ranges[1])
            idx_back = np.random.choice(ranges[2])
            # idx_front = ranges[0][0]  # fixed
            # idx_middle = ranges[1][0]
            # idx_back = ranges[2][0]
            # select mel
            mel_chunk_front = mel[idx_front:idx_front + chunk_frames, :]
            mel_chunk_middle = mel[idx_middle:idx_middle + chunk_frames, :]
            mel_chunk_back = mel[idx_back:idx_back + chunk_frames, :]
            # print(total_frames, idx_front, idx_front + chunk_frames, idx_middle, idx_middle + chunk_frames, idx_back, idx_back + chunk_frames)
            # stack
            mel_fusion = torch.stack([mel_chunk_front, mel_chunk_middle, mel_chunk_back], dim=0)
        elif mel.shape[0] < self.target_length:  # padding if too short
            n_repeat = int(self.target_length / mel.shape[0]) + 1
            # print(self.target_length, mel.shape[0], n_repeat)
            mel = mel.repeat(n_repeat, 1)[:self.target_length, :]
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        else:  # if equal
            mel_fusion = torch.stack([mel, mel, mel], dim=0)
        mel_fusion = mel_fusion.transpose(1, 2)  # [3, target_length, mel_bins] -> [3, mel_bins, target_length]

        # self.mean.append(mel_fusion.mean())
        # self.std.append(mel_fusion.std())
        # mel_fusion = (mel_fusion - self.audio_mean) / (self.audio_std * 2)
        return mel_fusion

    def get_mel(self, audio_data):
        # mel shape: (n_mels, T)
        audio_data -= audio_data.mean()
        mel = torchaudio.compliance.kaldi.fbank(
            audio_data,
            htk_compat=True,
            sample_frequency=self.sample_rate,
            use_energy=False,
            window_type="hanning",
            num_mel_bins=self.num_mel_bins,
            dither=0.0,
            frame_length=25,
            frame_shift=10,
        )
        return mel  # (T, n_mels)



def get_audio_transform(args):
    return AudioTransform(args)

def load_and_transform_audio(
    audio_path,
    transform,
):
    waveform_and_sr = torchaudio_loader(audio_path)
    audio_outputs = transform(waveform_and_sr)

    return audio_outputs