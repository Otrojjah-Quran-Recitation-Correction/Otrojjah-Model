from tensorflow.python.keras.utils.data_utils import Sequence
from scipy.io.wavfile import read, write
import numpy as np
import pandas as pd
from numpy import array, int16, float16
from scipy import signal
from scipy.signal import lfilter, butter
import librosa
import librosa.display
import cv2
import glob
import random
import os
from scipy.io import wavfile
import IPython.display as ipd
from matplotlib import pyplot as plt


class AudioProcessing(object):

    __slots__ = ('audio_data', 'sample_freq')

    def __init__(self, input_audio_path):
        self.sample_freq, self.audio_data = read(input_audio_path)
        self.audio_data = AudioProcessing.convert_to_mono_audio(
            self.audio_data)

    def save_to_file(self, output_path):
        '''Writes a WAV file representation of the processed audio data'''
        write(output_path, self.sample_freq,
              array(self.audio_data, dtype=int16))

    def set_audio_speed(self, speed_factor):
        '''Sets the speed of the audio by a floating-point factor'''
        sound_index = np.round(
            np.arange(0, len(self.audio_data), speed_factor))
        self.audio_data = self.audio_data[sound_index[sound_index < len(
            self.audio_data)].astype(int)]

    def set_echo(self, delay):
        '''Applies an echo that is 0...<input audio duration in seconds> seconds from the beginning'''
        output_audio = np.zeros(len(self.audio_data))
        output_delay = delay * self.sample_freq

        for count, e in enumerate(self.audio_data):
            output_audio[count] = e + \
                self.audio_data[count - int(output_delay)]

        self.audio_data = output_audio

    def set_volume(self, level):
        '''Sets the overall volume of the data via floating-point factor'''
        output_audio = np.zeros(len(self.audio_data))

        for count, e in enumerate(self.audio_data):
            output_audio[count] = (e * level)

        self.audio_data = output_audio

    def set_reverse(self):
        '''Reverses the audio'''
        self.audio_data = self.audio_data[::-1]

    def set_audio_pitch(self, n, window_size=2**13, h=2**11):
        '''Sets the pitch of the audio to a certain threshold'''
        factor = 2 ** (1.0 * n / 12.0)
        self._set_stretch(1.0 / factor, window_size, h)
        self.audio_data = self.audio_data[window_size:]
        self.set_audio_speed(factor)

    def _set_stretch(self, factor, window_size, h):
        phase = np.zeros(window_size)
        hanning_window = np.hanning(window_size)
        result = np.zeros(int(len(self.audio_data) / factor + window_size))

        for i in np.arange(0, len(self.audio_data) - (window_size + h), h*factor):
            # Two potentially overlapping subarrays
            a1 = self.audio_data[int(i): int(i + window_size)]
            a2 = self.audio_data[int(i + h): int(i + window_size + h)]

            # The spectra of these arrays
            s1 = np.fft.fft(hanning_window * a1)
            s2 = np.fft.fft(hanning_window * a2)

            # Rephase all frequencies
            phase = (phase + np.angle(s2/s1)) % 2*np.pi

            a2_rephased = np.fft.ifft(np.abs(s2)*np.exp(1j*phase))
            i2 = int(i / factor)
            result[i2: i2 + window_size] += hanning_window*a2_rephased.real

        # normalize (16bit)
        result = ((2 ** (16 - 4)) * result/result.max())
        self.audio_data = result.astype('int16')

    def set_lowpass(self, cutoff_low, order=5):
        '''Applies a low pass filter'''
        nyquist = self.sample_freq / 2.0
        cutoff = cutoff_low / nyquist
        x, y = signal.butter(order, cutoff, btype='lowpass', analog=False)
        self.audio_data = signal.filtfilt(x, y, self.audio_data)

    def set_highpass(self, cutoff_high, order=5):
        '''Applies a high pass filter'''
        nyquist = self.sample_freq / 2.0
        cutoff = cutoff_high / nyquist
        x, y = signal.butter(order, cutoff, btype='highpass', analog=False)
        self.audio_data = signal.filtfilt(x, y, self.audio_data)

    def set_bandpass(self, cutoff_low, cutoff_high, order=5):
        '''Applies a band pass filter'''
        cutoff = np.zeros(2)
        nyquist = self.sample_freq / 2.0
        cutoff[0] = cutoff_low / nyquist
        cutoff[1] = cutoff_high / nyquist
        x, y = signal.butter(order, cutoff, btype='bandpass', analog=False)
        self.audio_data = signal.filtfilt(x, y, self.audio_data)

    @staticmethod
    def convert_to_mono_audio(input_audio):
        '''Returns a numpy array that represents the mono version of a stereo input'''
        output_audio = []
        temp_audio = input_audio.astype(float)

        for e in temp_audio:
            output_audio.append((e[0] / 2) + (e[1] / 2))

        return np.array(output_audio, dtype='int16')


class AudioEffect(object):

    __slots__ = ()

    @staticmethod
    def darth_vader(input_path, output_path):
        '''Applies a Darth Vader effect to a given input audio file'''
        sound = AudioProcessing(input_path)
        sound.set_audio_speed(.8)
        sound.set_echo(0.02)
        sound.set_lowpass(2500)
        sound.save_to_file(output_path)

    @staticmethod
    def echo(input_path, output_path):
        '''Applies an echo effect to a given input audio file'''
        sound = AudioProcessing(input_path)
        sound.set_echo(0.09)
        sound.save_to_file(output_path)

    @staticmethod
    def radio(input_path, output_path):
        '''Applies a radio effect to a given input audio file'''
        sound = AudioProcessing(input_path)
        sound.set_highpass(2000)
        sound.set_volume(4)
        sound.set_bandpass(50, 2600)
        sound.set_volume(2)
        sound.save_to_file(output_path)

    @staticmethod
    def robotic(input_path, output_path):
        '''Applies a robotic effect to a given input audio file'''
        sound = AudioProcessing(input_path)
        sound.set_volume(1.5)
        sound.set_echo(0.01)
        sound.set_bandpass(300, 4000)
        sound.save_to_file(output_path)

    @staticmethod
    def ghost(input_path, output_path):
        '''Applies a ghostly halloween effect to a given input audio file'''
        sound = AudioProcessing(input_path)
        sound.set_reverse()
        sound.set_echo(0.05)
        sound.set_reverse()
        sound.set_audio_speed(.70)
        sound.set_audio_pitch(2)
        sound.set_volume(8.0)
        sound.set_bandpass(50, 3200)
        sound.save_to_file(output_path)


class DataGenerator(Sequence):
    def __init__(self, sh_paths, pos_paths, neg_paths, augment_for,
                 augmentation_methods, combine_augmentation, train, swap_axis=False,
                 noises_path='./new_data/noise', mfccs_number=20, max_pad_len=250,
                 batch_size=1, shuffle=True):
        self.augment_for = augment_for
        self.augmentation_methods = augmentation_methods
        self.combine_augmentation = combine_augmentation
        self.train = train
        self.swap_axis = swap_axis
        self.noises_path = noises_path
        self.mfccs_number = mfccs_number
        self.max_pad_len = max_pad_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        # self.sh_clips = np.empty(((augment_for * len(sh_paths) + len(sh_paths)), 20, self.max_pad_len))
        self.sh_clips = None
        self.sh_paths = self.__get_labeled_sh_paths(sh_paths)
        self.pos_and_neg_paths = self.__get_labeled_pos_and_neg_paths(
            pos_paths, neg_paths)
        self.__get_sh_clips()
        self.on_epoch_end()

    def __get_labeled_pos_and_neg_paths(self, pos_paths, neg_paths):
        pos = [(pos_path, 1) for pos_path in pos_paths]
        neg = [(neg_path, 0) for neg_path in neg_paths]
        return pos + neg

    def __get_labeled_sh_paths(self, sh_paths):
        return [(sh_path, 2) for sh_path in sh_paths]

    def __get_sh_clips(self):
        self.sh_clips, _ = self.get_clips(self.sh_paths)
        if self.train and self.augment_for != 0:
            augmented_sh_clips, __ = self.get_augmented_clips(self.sh_paths)
            self.sh_clips = np.vstack((self.sh_clips, augmented_sh_clips))

    def __len__(self):
        return len(self.pos_and_neg_paths)//self.batch_size

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size: (index+1) * self.batch_size]

        pos_and_neg_temp_paths = [self.pos_and_neg_paths[k] for k in indexes]

        x, y = self.__load_data(pos_and_neg_temp_paths)

        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.pos_and_neg_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __load_data(self, temp_clips):
        clips, labels = self.get_clips(temp_clips)

        if self.augment_for:
            augmented_clips, augmented_labels = self.get_augmented_clips(
                temp_clips)
            clips = np.vstack((clips, augmented_clips))
            labels = np.concatenate([labels, augmented_labels])

        x, y = self.__get_combinations(clips, labels)

        return x, y

    def __get_combinations(self, clips, labels):
        sh = []
        cl = []
        y = []
        for sh_clip in self.sh_clips:
            for i, clip in enumerate(clips):
                sh.append(sh_clip)
                cl.append(clip)
                y.append(labels[i])
        return [np.array(sh), np.array(cl)], np.array(y)

    def get_clips(self, clips_paths):
        clips = []
        labels = []
        np.random.shuffle(clips_paths)
        for path, label in clips_paths:
            clips.append(self.get_clip(path))
            labels.append(label)
        return np.array(clips), np.array(labels)

    def get_clip(self, file_path):
        wave, sr = librosa.load(file_path, mono=True)
        mfcc = librosa.feature.mfcc(np.asfortranarray(wave), sr=sr,
                                    n_mfcc=self.mfccs_number)
        if self.swap_axis:
            mfcc = np.swapaxes(mfcc, 0, 1)
        else:
            pad_width = self.max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=(
                (0, 0), (0, pad_width)), mode='constant')
        mfccx = max(mfcc.min(), mfcc.max(), key=abs)
        mfcc = mfcc/mfccx
        return mfcc

    def get_augmented_clips(self, clips_paths):
        augmented_clips = []
        labels = []
        np.random.shuffle(clips_paths)
        for i in range(self.augment_for):
            for path, label in clips_paths:
                if self.combine_augmentation:
                    augmented_clips.append(self.get_augmented_clip(path, 0))
                    labels.append(label)
                else:
                    for augmentation_method in self.augmentation_methods:
                        augmented_clips.append(
                            self.get_augmented_clip(path, augmentation_method))
                        labels.append(label)
        return np.array(augmented_clips), np.array(labels)

    def get_augmented_clip(self, file_path, augmentation_method):
        wave, sr = librosa.load(file_path, mono=True)
        if self.combine_augmentation:
            wave = self.Pitch_scaling(wave)
            wave = self.Add_noise(wave)
        else:
            if augmentation_method == 1:
                wave = self.Pitch_scaling(wave)
            if augmentation_method == 2:
                wave = self.Add_noise(wave)
            if augmentation_method == 3:
                path = self.Add_effect(file_path, 3)
                wave, sr = librosa.load(path, mono=True)
            if augmentation_method == 4:
                path = self.Add_effect(file_path, 4)
                wave, sr = librosa.load(path, mono=True)
            if augmentation_method == 5:
                path = self.Add_effect(file_path, 5)
                wave, sr = librosa.load(path, mono=True)
            if augmentation_method == 6:
                path = self.Add_effect(file_path, 6)
                wave, sr = librosa.load(path, mono=True)
            if augmentation_method == 7:
                path = self.Add_effect(file_path, 7)
                wave, sr = librosa.load(path, mono=True)
        mfcc = librosa.feature.mfcc(np.asfortranarray(wave), sr=sr)
        if self.swap_axis:
            mfcc = np.swapaxes(mfcc, 0, 1)
        else:
            pad_width = self.max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=(
                (0, 0), (0, pad_width)), mode='constant')
        mfccx = max(mfcc.min(), mfcc.max(), key=abs)
        mfcc = mfcc/mfccx
        return mfcc

    def Pitch_scaling(self, wave):
        speed_rate = np.random.uniform(0.7, 1.3)
        wave_speed_tune = cv2.resize(
            wave, (1, int(len(wave) * speed_rate))).squeeze()
        if len(wave_speed_tune) < 16000:
            pad_len = 16000 - len(wave_speed_tune)
            wave_speed_tune = np.r_[np.random.uniform(-0.001, 0.001, int(pad_len/2)),
                                    wave_speed_tune,
                                    np.random.uniform(-0.001, 0.001, int(np.ceil(pad_len/2)))]
        return wave_speed_tune

    def Add_noise(self, wave):
        wave = np.interp(wave, (wave.min(), wave.max()), (-1, 1))
        noise_files = glob.glob(os.path.join(self.noises_path, '*.wav'))
        noise_audio = random.choice(noise_files)
        noise, noise_sr = librosa.load(noise_audio, mono=True)
        noise = np.interp(noise, (noise.min(), noise.max()), (-1, 1))
        if(len(noise) > len(wave)):
            noise = noise[0:len(wave)]
        else:
            noise = np.pad(noise, (0, (len(wave)-len(noise))), 'wrap')
        wave_with_noise = wave + noise
        return wave_with_noise

    def Add_effect(self, file_path, augmentation_method):
        input_file_path = file_path
        output_filename = './new_data/output/'
        if augmentation_method == 3:
            AudioEffect.ghost(
                input_file_path, output_filename + 'ghost/_ghost.wav')
            path = glob.glob(os.path.join(
                './new_data' + '/output/ghost', '*.wav'))
        if augmentation_method == 4:
            AudioEffect.robotic(
                input_file_path, output_filename + 'robotic/_robotic.wav')
            path = glob.glob(os.path.join(
                './new_data' + '/output/robotic', '*.wav'))
        if augmentation_method == 5:
            AudioEffect.echo(
                input_file_path, output_filename + 'echo/_echo.wav')
            path = glob.glob(os.path.join(
                './new_data' + '/output/echo', '*.wav'))
        if augmentation_method == 6:
            AudioEffect.radio(
                input_file_path, output_filename + 'radio/_radio.wav')
            path = glob.glob(os.path.join(
                './new_data' + '/output/radio', '*.wav'))
        if augmentation_method == 7:
            AudioEffect.darth_vader(
                input_file_path, output_filename + 'darth_vader/_vader.wav')
            path = glob.glob(os.path.join(
                './new_data' + '/output/darth_vader', '*.wav'))

        return path[0]

    @staticmethod
    def display_audio(path):
        ipd.display(ipd.Audio(path))

    @staticmethod
    def get_false_positives(preds_neg, actual_neg, neg_clip_paths):
        for i in range(preds_neg.shape[0]):
            if preds_neg[i] == 1:
                ipd.display(ipd.Audio(neg_clip_paths[i]))
                print(actual_neg[i])

    @staticmethod
    def get_false_negatives(preds_pos, actual_pos, pos_clip_paths):
        for i in range(preds_pos.shape[0]):
            if preds_pos[i] == 0:
                ipd.display(ipd.Audio(pos_clip_paths[i]))
                print(actual_pos[i])

    @staticmethod
    def split_paths(data_dir, split_at=.3):
        pos_paths = glob.glob(os.path.join(data_dir + '/positive', '*.wav'))
        neg_paths = glob.glob(os.path.join(data_dir + '/negative', '*.wav'))
        sh_paths = glob.glob(os.path.join(data_dir + '/shaikh', '*.wav'))

        pos_paths = sorted(pos_paths)
        neg_paths = sorted(neg_paths)
        sh_paths = sorted(sh_paths)

        min_length = min(len(pos_paths), len(neg_paths))
        split_point = int(min_length*(1-split_at))
        print("Minimum # of(Positive, Negative) is: ", min_length)
        print("Split point is: ", split_point)

        pos_train_paths = pos_paths[:split_point]
        pos_val_paths = pos_paths[split_point:]
        neg_train_paths = neg_paths[:split_point]
        neg_val_paths = neg_paths[split_point:]

        return pos_train_paths, neg_train_paths, pos_val_paths, neg_val_paths, sh_paths

    @staticmethod
    def graph_spectrogram(wav_file):
        rate, data = DataGenerator.get_wav_info(wav_file)
        nfft = 200  # Length of each window segment
        fs = 8000  # Sampling frequencies
        noverlap = 120  # Overlap between windows
        nchannels = data.ndim
        if nchannels == 1:
            pxx, freqs, bins, im = plt.specgram(
                data, nfft, fs, noverlap=noverlap)
        elif nchannels == 2:
            pxx, freqs, bins, im = plt.specgram(
                data[:, 0], nfft, fs, noverlap=noverlap)
        return pxx

    @staticmethod
    def get_wav_info(wav_file):
        rate, data = wavfile.read(wav_file)
        return rate, data

    @staticmethod
    def plot_mfcc(path):
        # Hard coded
        wave, sr = librosa.load(path, mono=True)
        mfcc = librosa.feature.mfcc(np.asfortranarray(wave), sr=sr)
        pad_width = 250 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=(
            (0, 0), (0, pad_width)), mode='constant')
        ipd.display(librosa.display.specshow(mfcc))
