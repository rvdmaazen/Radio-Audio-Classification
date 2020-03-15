import torch
from torch.utils.data import Dataset
from torchaudio import load, transforms
import glob
import os


class AudioDataset(Dataset):
    def __init__(
        self,
        path,
        sample_rate=22050,
        n_fft=2048,
        n_mels=128,
        win_length=1024,
        hop_length=1024,
        log_mel=True,
        augment=False
    ):
        """
        A custom dataset class to load audio snippets and create
        mel spectrograms.

        Args:
            path (string): path to folder with audio files
            sample_rate (integer): sample rate of audio signal
            n_fft (integer): number of Fourier transforms to use for the mel spectrogram
            n_mels (integer): number of mel bins to use for the mel spectrogram
            log_mel (boolean): whether to use log-mel spectrograms instead of db-scaled
        """
        self.path = path
        self.sr = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.win_length = win_length
        self.hop_length = hop_length
        self.log_mel = log_mel
        self.augment = augment
        self.file_paths = glob.glob(os.path.join(self.path, "**", f"*wav"), recursive=True)
        self.labels = [x.split("/")[-2] for x in self.file_paths]
        self.mapping = {"ads_other": 0, "music": 1}
        for i, label in enumerate(self.labels):
            self.labels[i] = self.mapping[label]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        audio, sr = load(self.file_paths[index])
        audio = torch.mean(audio, dim=0, keepdim=True)
        if self.sr != sr:
            audio = transforms.Resample(sr, self.sr)(audio)
        mel_spectrogram = transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            f_max=self.sr / 2
        )(audio)
        if self.log_mel:
            offset = 1e-6
            mel_spectrogram = torch.log(mel_spectrogram + offset)
        else:
            mel_spectrogram = transforms.AmplitudeToDB(stype="power", top_db=80)(mel_spectrogram)
        if self.augment:
            audio = transforms.FrequencyMasking(freq_mask_param=20)(audio)
            audio = transforms.TimeMasking(time_mask_param=10)(audio)
        label = self.labels[index]
        return mel_spectrogram, label
