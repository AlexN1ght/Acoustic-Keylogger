# %%
import os
from torch.utils.data import Dataset
import pandas as pd
import torchaudio
import json
import torch
import matplotlib.pyplot as plt
from librosa.feature import mfcc
import umap

from bisect import bisect_left


# %%
class KeySoundDataset(Dataset):
    def __init__(
            self,
            dataset_path, 
            shape=(10, 100),
            hop_length=110,
            n_fft=1024,
            mode='mfcc',
            reduse_dims=32,
            pwr_trashhold=2,
            time_drift=0.015,
            start_time_shift=0.193,
            press_frame_window=100,
            label_detection_range=80,
            min_bin_for_power=15):
        assert mode == 'mfcc' or mode == 'mel_spec'

        self.mode = mode
        self.reduse_dims = reduse_dims
        self.shape = shape
        self.spec_shape = shape
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.pwr_trashhold = pwr_trashhold
        self.one_hot = False
        self.time_drift = time_drift
        self.start_time_shift = start_time_shift
        self.press_frame_window = press_frame_window
        self.label_detection_range = label_detection_range
        self.min_bin_for_power = min_bin_for_power
        self.annotations = pd.read_csv(os.path.join(dataset_path, 'inputs.csv'), sep='\t')

        self.source_audio, self.source_audio_sample_rate =\
            torchaudio.load(os.path.join(dataset_path, 'keyboard_audio.wav'))

        audio_metadata = open(os.path.join(dataset_path, 'audio_metadata.json'), 'r')
        self.audio_start_ts = json.load(audio_metadata)['start_time'] # - 0.05 #shift fix
        audio_metadata.close()

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.source_audio_sample_rate,
            n_fft=self.n_fft,
            hop_length=110,
            f_min = 150,
            f_max = 20000,
            n_mels=self.spec_shape[1],
        )

        if self.reduse_dims is not None:
            self.umap_reducer = umap.UMAP(n_components=reduse_dims, n_neighbors=15, metric='manhattan', n_epochs=1000, low_memory=False)

        self.mel_spectrogram = self.mel_transform(self.source_audio)

        self.make_power()
        self.make_keys_start_frame()
        self.make_mfcc_features()
        self.convert_annotations()

        self.make_cache()
        self.make_lables_cache()

        self.original_text = ''
        for i in range(self.cache.shape[0] - 1):
            self.original_text += self.get_label_item(i)


    def __len__(self):
        return self.cache.shape[0] - 1

    def make_cache(self):

        self.cache = torch.zeros(
            (len(self.keys_start_frame) - 1, self.shape[1] * self.shape[0]), dtype=torch.float)
        
        for i in range(len(self.keys_start_frame) - 1):
            self.cache[i] = torch.flatten(self.get_raw_item(i))

        if self.reduse_dims is not None:
            self.shape = tuple([self.reduse_dims])
            self.umap_reducer.fit(self.cache)
            self.cache = torch.tensor(self.umap_reducer.transform(self.cache))

        self.cache = self.cache -  torch.min(self.cache)
        self.cache = self.cache / torch.max(self.cache)

        print("Max value:", torch.max(self.cache))
        print("Min value:", torch.min(self.cache))

    def make_lables_cache(self):
        self.lables_cache = ['' for _ in range(len(self.keys_start_frame) - 1)]

        for i in range(len(self.keys_start_frame) - 1):
            self.lables_cache[i] = self.get_label_item(i)


    def __getitem__(self, index):
        lbl = self.lables_cache[index]
        if self.one_hot:
            lbl = torch.nn.functional.one_hot(torch.tensor(self.id_m[lbl]), len(self.id_m)).float()
        return self.cache[index], lbl

    def get_raw_item(self, index):
        if self.mode == 'mfcc':
            feature = self.get_mfcc_item(index)
        elif self.mode == 'mel_spec':
            feature = self.get_mel_item(index)
        return feature

    def get_mfcc_item(self, index):
        frame = self.index_to_frame(index)
        return self.mfcc_features[:,:, frame:frame+self.spec_shape[0]]

    def get_mel_item(self, index):
        frame = self.index_to_frame(index)
        return self.mel_spectrogram[:,:, frame:frame+self.spec_shape[0]]
    
    def get_power_item(self, index):
        frame = self.index_to_frame(index)
        return self.power[frame:frame+self.spec_shape[0]]

    def get_signal_item(self, index):
        frame = self.index_to_frame(index)
        return self.source_audio[:, self.frame_to_sample(frame):self.frame_to_sample(frame+self.spec_shape[0])]
    
    def get_label_item(self, index):
        frame = self.index_to_frame(index)
        sample = self.frame_to_sample(frame)
        lbl = self.take_closest_label(sample)
        if lbl == 'space':
            lbl = ' '
        if len(lbl) > 1:
            print(lbl)
            lbl = ' '
        return lbl

    def frame_to_sample(self, frame):
        return frame * 110

    def index_to_frame(self, index):
        return self.keys_start_frame[index] - 2

    def make_power(self):
        self.power = torch.squeeze(torch.sum(self.mel_spectrogram[:,self.min_bin_for_power:,:], dim=1))

    def make_keys_start_frame(self):
        self.keys_start_frame = []
        push_start = False
        for i in range(1, len(self.power) - 1):
            if self.power[i] > self.pwr_trashhold and \
                    self.power[i - 1] < self.pwr_trashhold and \
                    (i - (self.keys_start_frame[-1] if self.keys_start_frame else 0)) > self.press_frame_window :
                push_start = True
            if  push_start and \
                    self.power[i - 1] < self.power[i] and \
                    self.power[i + 1] < self.power[i]:
                self.keys_start_frame.append(i)
                push_start = False

    def make_mfcc_features(self):
        self.mfcc_features = torch.tensor(
                mfcc(y=self.source_audio.numpy(),
                sr=self.source_audio_sample_rate,
                n_mfcc= self.spec_shape[1],
                n_mels= self.spec_shape[1],
                fmin = 150,
                fmax = 20000,
                n_fft=self.n_fft, # n_fft=220 for a 10ms window
                hop_length=self.hop_length, # hop_length=110 for ~2.5ms
            )
        )

    def convert_annotations(self):
        self.annotations_samples = []
        self.annotations_lbls = []
        for i in range(len(self.annotations)):
            lbl = self.annotations.iloc[i, 0]
            start_time = self.annotations.iloc[i, 1] - self.audio_start_ts - self.start_time_shift 
            start_time = start_time - (self.time_drift / 1000) * start_time  # fix time drift 
            start_sample = int(start_time * self.source_audio_sample_rate)
            self.annotations_samples.append(start_sample)
            self.annotations_lbls.append(lbl)

    def take_closest_label(self, sample):
        pos = bisect_left(self.annotations_samples, sample)
        if pos == 0:
            ann_sample = self.annotations_samples[0]
            lbl = self.annotations_lbls[0]
        elif pos == len(self.annotations_samples):
            ann_sample = self.annotations_samples[-1]
            lbl = self.annotations_lbls[-1]
        else:
            before = self.annotations_samples[pos - 1]
            after = self.annotations_samples[pos]
            if after - sample < sample - before:
                ann_sample = self.annotations_samples[pos]
                lbl = self.annotations_lbls[pos]
            else:
                ann_sample = self.annotations_samples[pos - 1]
                lbl = self.annotations_lbls[pos - 1]
        if abs(ann_sample - sample) > self.hop_length * self.label_detection_range:
            lbl = '$'
        return lbl

    def score(self, pred_text, exclude_spaces=False):
        cnt = 0
        spaces_count = 0
        for i, char in enumerate(pred_text):
            true_lbl = self.get_label_item(i)
            if true_lbl == ' ' and exclude_spaces:
                spaces_count += 1
                continue
            if char == true_lbl:
                cnt += 1
        return cnt / (len(self) - spaces_count)

    def get_full_data(self):
        data = torch.zeros((len(self) - 1, self[0][0].shape[0]), dtype=torch.float32)
        lbls = []
        true_lbls = []

        for i in range(len(self) -1):
            features, lbl = self[i]
            data[i] = torch.flatten(features)
            true_lbls.append(lbl)
            lbls.append(ord(lbl))

        mapping = dict([ (a, b) for b, a in enumerate(set(lbls))])
        for i in range(len(lbls)):
            lbls[i] = mapping[lbls[i]]
        return data, lbls, true_lbls

    def rebuild_from_indexes(self, indexes, new_lbls=None):
        cache_copy = self.cache
        self.cache = torch.zeros((len(indexes), self.cache.shape[1]), dtype=torch.float)

        
        lbls_copy = self.lables_cache
        self.lables_cache = [ '' for i in range(len(indexes))]
        
        for i, idx in enumerate(indexes):
            self.cache[i] = cache_copy[idx]
            self.lables_cache[i] = lbls_copy[idx]

        if new_lbls is not None:
            for i, lbl in enumerate(new_lbls):
                self.lables_cache[i] = lbl

    def enable_one_hot(self, id_m):
        self.one_hot = True
        self.id_m = id_m

    def disable_one_hot(self):
        self.one_hot = False


    # deprecated 
    def get_sound_from_annotation(self, index):
        start_sample, end_sample = self.sample_range(index)
        signal = self.source_audio[:,start_sample:end_sample]
        return signal.numpy()


    # deprecated 
    def sample_range_from_annotation(self, index):
        start_time = self.annotations.iloc[index, 1] - self.audio_start_ts - self.start_time_shift
        start_time = start_time - (self.time_drift / 1000) * start_time  # fix time drift 
        start_sample = int(start_time * self.source_audio_sample_rate)
        end_sample = start_sample + (self.spec_shape[0] - 1) * self.hop_length
        return start_sample, end_sample


class BagOfKeysDataset(Dataset):
    def __init__(self, key_sound_dataset, context_size=1):
        self.key_sound_dataset = key_sound_dataset
        self.context_size = context_size

    def __len__(self):
        return max(len(self.key_sound_dataset.keys_start_frame) - 1 - self.context_size * 2, 0)

    def __getitem__(self, index):
        shifted_index = index + self.context_size
        return \
          torch.cat(
              (self.key_sound_dataset.cache[shifted_index-self.context_size:shifted_index],
               self.key_sound_dataset.cache[shifted_index+1:shifted_index + self.context_size+ 1])), \
          self.key_sound_dataset.cache[shifted_index]

def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Pic')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec[0], origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
