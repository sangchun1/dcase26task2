import torch
import random
import numpy as np
import pandas as pd
import soundfile as sf
import torchaudio
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

def custom_collate_fn(batch):
    b = {
        "source": [i[0] for i in batch], 
        "padding_mask": [i[1] for i in batch], 
        "audio_path": [i[2] for i in batch], 
        "machine": [i[3] for i in batch], 
        "year": [i[4] for i in batch], 
        "domain": [i[5] for i in batch], 
        "attribute": [i[6] for i in batch], 
        "label": [i[7] for i in batch]
    }
    return b

def custom_collate_fn_eval(batch):
    b = {
        "source": [i[0] for i in batch], 
        "padding_mask": [i[1] for i in batch], 
        "audio_path": [i[2] for i in batch], 
        "machine": [i[3] for i in batch]
    }
    return b

def pad_or_truncate_to_seconds(wav, sr, target_seconds=10):
    target_length = target_seconds * sr
    current_length = wav.shape[0]
    
    padding_mask = np.zeros(target_length)
    
    if current_length < target_length:
        pad_size = target_length - current_length
        wav = np.pad(wav, (0, pad_size), 'constant')
        padding_mask[current_length:] = 1
    elif current_length > target_length:
        max_start = current_length - target_length
        start_idx = np.random.randint(0, max_start + 1)
        wav = wav[start_idx:start_idx + target_length]
    else:
        pass
    
    return wav, padding_mask

class CustomDataset(Dataset):
    def __init__(self, data_path, augment, freqm, timem, audio_model='eat', test=False):
        le = LabelEncoder()
        self.df = pd.read_csv(data_path)
        if 'train' in data_path:
            self.df['label'] = le.fit_transform(self.df['label'])

        self.augment = augment
        self.freqm = freqm
        self.timem = timem
        
        self.norm_mean = -5.1731
        self.norm_std = 2.9974

        self.audio_model = audio_model
        
        self.test = test
        
    def __len__(self):
        return len(self.df)
    
    def spectrogram_augment(self,spec):
        freq_masking = torchaudio.transforms.FrequencyMasking(self.freqm, iid_masks=True)
        time_masking = torchaudio.transforms.TimeMasking(self.timem, iid_masks=True)
        spec_ = spec.transpose(1, 2)
        input_with_freq_mask = freq_masking(spec_)
        input_with_time_freq_mask = time_masking(input_with_freq_mask)
        input_with_time_freq_mask = torch.transpose(input_with_time_freq_mask, 1, 2)
        return input_with_time_freq_mask
    
    def __getitem__(self, index):
        row = self.df.iloc[index, :]
        source_file = row['audio_path']
        wav, sr = sf.read(source_file)
        channel = sf.info(source_file).channels
        assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)

        # # add random noise with random snr
        # if self.add_noise and row['is_clean']:
        #     noise_idx = random.choice(range(len(self.noise_df)))
        #     noise_sample = self.noise_df.loc[noise_idx, 'audio_path']
        #     noise_sample, _ = sf.read(noise_sample)
        #     snr_db = np.random.uniform(0, 15)
        #     wav = add_noise(wav, noise_sample, snr_db)

        wav = wav - wav.mean()  # Zero-mean normalization
        wav, padding_mask = pad_or_truncate_to_seconds(wav, sr, target_seconds=10)
        
        wav = torch.from_numpy(wav).float()
        wav = wav.unsqueeze(dim=0)

        padding_mask = torch.from_numpy(padding_mask).float()
        padding_mask = padding_mask.unsqueeze(dim=0)

        if self.audio_model=='beats':
            source = wav
        else:
            mel = torchaudio.compliance.kaldi.fbank(wav, htk_compat=True, sample_frequency=16000, 
                                                    use_energy=False, window_type='hanning', num_mel_bins=128, 
                                                    dither=0.0, frame_shift=10).unsqueeze(dim=0)

            source = (mel - self.norm_mean) / (self.norm_std)
            
            if self.augment:
                source = self.spectrogram_augment(source)

            assert not (torch.isnan(source).any() or torch.isinf(source).any()), \
                print("NaN or Inf detected after normalization")

        if not self.test:
            return source, padding_mask, row['audio_path'], row['machine'], row['year'], row['domain'], row['attribute'], row['label']
        else:
            return source, padding_mask, row['audio_path'], row['machine']