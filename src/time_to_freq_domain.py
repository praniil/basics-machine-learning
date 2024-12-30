import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

file_path = "/home/pranil/python_projects/goolge_collab_projects/learning-google-collab/basics-machine-learning/data/sample_wav.wav"
sample_rate, audio_data = wavfile.read(file_path)

if audio_data.ndim > 1:
    audio_data = audio_data[:, 0]

n = len(audio_data)
frequency_data = np.fft.fft(audio_data)
frequencies = np.fft.fftfreq(n, d = 1/sample_rate)

magnitude_spectrum = np.abs(frequency_data)

time_axis = np.linspace(0, n / sample_rate, n)
print(time_axis)

# Plot time-domain signal
plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis, audio_data, color='blue')
plt.title('Time-Domain Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.savefig('time_domain_audio_plot.png') 
plt.clf()

# Plot frequency-domain signal
plt.subplot(2, 1, 2)
plt.plot(frequencies[:n // 2], magnitude_spectrum[:n // 2], color='red')  # Positive frequencies only
plt.title('Frequency-Domain Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.savefig('frequency_domain_audio_plot.png') 
plt.clf()

print(magnitude_spectrum)

print(audio_data)