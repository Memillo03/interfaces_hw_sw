"""This module contains the AudioSignal class to make audio signal processing."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from scipy.fft import fft
from scipy.io import wavfile
from scipy.signal import filtfilt, firwin, iirfilter, kaiserord, lfilter


class AudioSignal:
    """A class that represent the audio signal to be manipulated."""

    def __init__(self, filepath):
        """Initialize the AudioSignal object."""
        self.file = filepath

        # check extension
        if self.file.endswith(".wav"):
            self.sampFreq, self.signal = wavfile.read(self.file)
        elif self.file.endswith(".mp3"):
            temp = AudioSegment.from_mp3(self.file)
            self.signal = np.array(temp.get_array_of_samples())
            self.sampFreq = temp.frame_rate
        elif self.file.endswith(".aac"):
            temp = AudioSegment.from_file(self.file, format="aac")
            self.signal = np.array(temp.get_array_of_samples())
            self.sampFreq = temp.frame_rate
        else:
            print("Invalid file format")
            return

        if (
            len(self.signal.shape) > 1 and self.signal.shape[1] > 1
        ):  # if stereo, convert to mono
            self.signal = self.signal[:, 0]  # only one channel

        self.normSignal = self.signal / 2.0**15
        self.nSamples = self.signal.shape[0]
        self.duration = self.nSamples / self.sampFreq
        self.processedSignal = np.zeros(self.signal.shape)

    def get_signal(self, signal):
        """Plot original audio signal."""
        time_vector = np.linspace(0, self.duration, self.nSamples)
        if signal == "original":
            return time_vector, self.normSignal
        elif signal == "processed":
            return time_vector, self.processedSignal
        else:
            print("Invalid signal type")
            return

    def generate_filter(
        self, filter_type, band_type, cutoff_freqs, order=4, ftype="butter", rp=1, rs=40
    ):
        """Generate a filter to be applied to the audio signal."""
        if filter_type == "iir":
            if ftype == "butter" or ftype == "bessel":
                b, a = iirfilter(
                    N=order,
                    Wn=cutoff_freqs,
                    fs=self.sampFreq,
                    btype=band_type,
                    ftype=ftype,
                )
            else:
                b, a = iirfilter(
                    N=order,
                    Wn=cutoff_freqs,
                    fs=self.sampFreq,
                    btype=band_type,
                    ftype=ftype,
                    rp=rp,
                    rs=rs,
                )
            return b, a

        elif filter_type == "fir":
            attenuation_db = 65
            transition_width = 24 / (self.sampFreq * 0.5)
            N, beta = kaiserord(ripple=attenuation_db, width=transition_width)
            coeffs = firwin(
                numtaps=N,
                cutoff=cutoff_freqs,
                window=("kaiser", beta),
                pass_zero=band_type,
                fs=self.sampFreq,
            )
            return coeffs
        else:
            print("Invalid filter type")
            return

    def apply_filter(
        self, filter_type, band_type, cutoff_freqs, order=4, ftype="butter"
    ):
        """Apply a filter to the audio signal."""
        if filter_type == "iir":
            b, a = self.generate_filter(
                filter_type, band_type, cutoff_freqs, order, ftype
            )
            self.processedSignal = filtfilt(b, a, self.normSignal)

        elif filter_type == "fir":
            coeffs = self.generate_filter(filter_type, band_type, cutoff_freqs)
            self.processedSignal = lfilter(coeffs, 1.0, self.normSignal)

        else:
            print("Invalid filter type")
            return

    def fourier_transform(self, sig):
        """Compute the Fourier Transform of a signal."""
        if sig == "original":
            signal = self.normSignal
        elif sig == "processed":
            signal = self.processedSignal
        else:
            print("Invalid signal type")
            return

        ft = fft(signal)
        freq = np.linspace(0, self.sampFreq, len(ft))
        # freq = fftfreq(len(ft), 1/self.sampFreq)
        # ft = fftshift(ft)
        return freq[: len(freq) // 2], ft[: len(ft) // 2]

    def plot_fft(self, sig):
        """Plot the Fourier Transform of the signal."""
        freq, ft = self.fourier_transform(sig)
        freq = freq[: len(freq) // 2]
        ft = ft[: len(ft) // 2]

        plt.plot(freq, np.abs(ft))
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Amplitude")
        plt.show()

    def save_signal(self, file_path, file_name, format):
        """Save the processed signal to a file."""
        # add file extension
        file_name = file_name + format
        path = Path(file_path) / file_name
        data = np.int16(
            self.processedSignal * 32767 / np.max(np.abs(self.processedSignal))
        )

        if format == ".wav":
            wavfile.write(str(path), self.sampFreq, data)
        elif format == ".mp3":
            song = AudioSegment(
                data.tobytes(), frame_rate=self.sampFreq, sample_width=2, channels=1
            )
            song.export(path, format="mp3")
        elif format == ".aac":
            song = AudioSegment(
                data.tobytes(), frame_rate=self.sampFreq, sample_width=2, channels=1
            )
            song.export(path, format="adts")
