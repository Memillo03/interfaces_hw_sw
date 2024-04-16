"""Filtering audio signal using Fourier Transform."""

# Libraries
import matplotlib.pyplot as plt
import numpy as np
from playsound import playsound
from scipy.io import wavfile


def audio_filter():
    """Filtering audio signal using Fourier Transform."""
    # Play audio
    playsound("MB_song.wav")

    # Read audio file
    sampFreq, sound = wavfile.read("MB_song.wav")
    print(sound.dtype, sampFreq)

    # Normalice audio to b between - 1 to 1
    sound = sound / 2.0**15

    # Just one channel
    sound = sound[:, 0]

    # Measure in seconds
    length_in_s = sound.shape[0] / sampFreq
    print("Audio length in seconds: ", length_in_s)

    # Audio plot
    plt.plot(sound[:], "r")
    plt.xlabel("Sound signal")
    plt.tight_layout
    plt.show()

    # Time vector
    time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s
    plt.plot(time, sound[:], "r")
    plt.xlabel("Time signal [s]")
    plt.tight_layout
    plt.show()

    # Add noise to signal
    yerr = (
        0.005 * np.sin(2 * np.pi * 6000.0 * time)
        + 0.008 * np.sin(2 * np.pi * 8000.0 * time)
        + 0.006 * np.sin(2 * np.pi * 2500.0 * time)
    )
    signal = sound + yerr

    # Zoom
    plt.plot(time[6000:7000], signal[6000:7000])
    plt.ylabel("Time [s]")
    plt.xlabel("Amplitude")
    plt.show()

    # Fourier Transform
    fft_spectrum = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(signal.size, d=1.0 / sampFreq)
    print("Fourier Transform: ", fft_spectrum)
    fft_spectrum_abs = np.abs(fft_spectrum)

    # Plot FFT
    plt.plot(freq, fft_spectrum_abs)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.show()

    # Filter working on FFT domain
    for i, f in enumerate(freq):
        if f > 5900 and f < 6100:
            fft_spectrum[i] = 0.0

    noiseless_signal = np.fft.irfft(fft_spectrum)

    # Audio plot
    plt.plot(time, noiseless_signal, "r")
    plt.xlabel("Time signal [s]")
    plt.tight_layout
    plt.show()

    file_path1 = r"C:\Users\Guilermo\Documents\inter_hw_sw\filters\noisy_audio.wav"
    file_path_2 = r"C:\Users\Guilermo\Documents\inter_hw_sw\filters\noiseless_audio.wav"

    wavfile.write(file_path1, sampFreq, signal)
    wavfile.write(file_path_2, sampFreq, noiseless_signal)
    playsound("noisy_audio.wav")
    playsound("noiseless_audio.wav")


if __name__ == "__main__":
    """Filtering audio signal using Fourier Transform."""
    audio_filter()
