import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torchaudio



class AudioLoader(Dataset):
    """Pytorch audio loader."""
    def __init__(
        self,
        path,
        output_path,
        sample_rate,
        num_mel_bins,
        frame_shift,
        target_length,
        audio_mean,
        audio_std,
    ):
        self.audio_path = path
        self.audio_list = os.listdir(self.audio_path)
        self.output = output_path
        self.sample_rate = sample_rate
        self.target_length = target_length
        self.num_mel_bins = num_mel_bins
        self.frame_shift = frame_shift
        self.audio_mean = audio_mean
        self.audio_std = audio_std

    def __len__(self):
        return len(self.audio_list)
  
    def waveform2melspec(self, audio_data):
        mel = self.get_mel(audio_data)
        #################################
        if mel.shape[0] < self.target_length:# padding if too short
            n_repeat = int(self.target_length / mel.shape[0]) + 1
            mel = mel.repeat(n_repeat, 1)[:self.target_length, :]

        chunk_size = self.target_length//10
        mel_chunks = torch.split(mel, chunk_size)
        if len(mel_chunks[-1])<chunk_size:#avoid smaller chunks
            mel_chunks = mel_chunks[:-1]
        #################################
        mel_selected = torch.stack([mel_chunks[i] for i in range(len(mel_chunks))])
        mel_selected = torch.unsqueeze(mel_selected,1)
        mel_fusion = torch.cat((mel_selected,mel_selected,mel_selected), 1)#[T, 3, 224, mel_bins]TxCxWxH
        #################################
        # #################################
        # if len(mel_chunks) == 10:#the audio lenght is 10 seconds
        #     mel_selected = torch.stack([mel_chunks[i] for i in range(len(mel_chunks))])
        #     mel_selected = torch.unsqueeze(mel_selected,1)
        #     mel_fusion = torch.cat((mel_selected,mel_selected,mel_selected), 1)#[10, 3, 224, mel_bins]TxCxWxH
        # else:
        #     if len(mel_chunks) >= 30:##the audio lenght is more than 30 seconds
        #         idxs = sorted(np.random.choice(len(mel_chunks),30,replace=False))
        #         mel_selected = torch.stack([mel_chunks[i] for i in idxs])
        #         mel_fusion = torch.reshape(mel_selected,[10,3, chunk_size, self.num_mel_bins])# [10, 3, 224, mel_bins]
        #     else: 
        #         idxs = sorted(np.random.choice(len(mel_chunks),10,replace=False))
        #         mel_selected = torch.stack([mel_chunks[i] for i in idxs])
        #         mel_selected = torch.unsqueeze(mel_selected,1)
        #         mel_fusion = torch.cat((mel_selected,mel_selected,mel_selected), 1)# [10, 3, 224, mel_bins] 

        mel_fusion = mel_fusion.transpose(2, 3) # [T, 3, target_length, mel_bins] -> [T, 3, mel_bins, target_length]

        mel_fusion = (mel_fusion - self.audio_mean) / (self.audio_std * 2)    

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
            frame_shift=self.frame_shift,
        )
        return mel  # (T, n_mels)

    def __getitem__(self, idx):
        audio_file = self.audio_list[idx]
        output_file = self.output

        audio_data, origin_sr = torchaudio.load(os.path.join(self.audio_path,audio_file))
        audio = self.waveform2melspec(audio_data)

        return {"audio": audio, "input": os.path.join(self.audio_path,audio_file), "output": os.path.join(output_file,audio_file[:-4])}
