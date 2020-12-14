# REPET-Python

## repet Python module

repet Functions:
- [original - REPET (original)](#repet-original-1)
- [extended - REPET extended](#repet-extended-1)
- [adaptive - Adaptive REPET](#adaptive-repet-1)
- [sim - REPET-SIM](#repet-sim-1)
- [simonline - Online REPET-SIM](#online-repet-sim-1)

### REPET (original)
`background_signal = repet.original(audio_signal, sample_rate)`

Arguments:
```
audio_signal: audio signal [number_samples, number_channels]
sample_rate: sample rate in Hz
background_signal: background signal [number_samples, number_channels]
```

Example: Estimate the background and foreground signals, and display their spectrograms
```
# Import modules
import scipy.io.wavfile
import repet
import numpy as np
import matplotlib.pyplot as plt

# Audio signal (normalized) and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))

# Estimate the background signal and infer the foreground signal
background_signal = repet.original(audio_signal, sample_rate);
foreground_signal = audio_signal-background_signal;

# Write the background and foreground signals (un-normalized)
scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)

# Compute the audio, background, and foreground spectrograms
window_length = repet.windowlength(sample_rate)
window_function = repet.windowfunction(window_length)
step_length = repet.steplength(window_length)
audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])

# Display the audio, background, and foreground spectrograms (up to 5kHz)
plt.rc('font', size=30)
plt.subplot(3, 1, 1)
plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Audio Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
           np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
           np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 2)
plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Background Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
           np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
           np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 3)
plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Foreground Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
           np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
           np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.show()
```

<img src="images/repet_python/repet_original.png" width="1000">

### REPET extended
`background_signal = repet.extended(audio_signal, sample_rate)`

Arguments:
```
audio_signal: audio signal [number_samples, number_channels]
sample_rate: sample rate in Hz
background_signal: background signal [number_samples, number_channels]
```

Example: Estimate the background and foreground signals, and display their spectrograms
```
import scipy.io.wavfile
import repet
import numpy as np
import matplotlib.pyplot as plt

# Audio signal (normalized) and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))

# Estimate the background signal and infer the foreground signal
background_signal = repet.extended(audio_signal, sample_rate);
foreground_signal = audio_signal-background_signal;

# Write the background and foreground signals (un-normalized)
scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)

# Compute the audio, background, and foreground spectrograms
window_length = repet.windowlength(sample_rate)
window_function = repet.windowfunction(window_length)
step_length = repet.steplength(window_length)
audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])

# Display the audio, background, and foreground spectrograms (up to 5kHz)
plt.rc('font', size=30)
plt.subplot(3, 1, 1)
plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Audio Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 2)
plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Background Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 3)
plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Foreground Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.show()
```

<img src="images/repet_python/repet_extended.png" width="1000">

### Adaptive REPET
`background_signal = repet.adaptive(audio_signal, sample_rate)`

Arguments:
```
audio_signal: audio signal [number_samples, number_channels]
sample_rate: sample rate in Hz
background_signal: background signal [number_samples, number_channels]
```

Example: Estimate the background and foreground signals, and display their spectrograms
```
import scipy.io.wavfile
import repet
import numpy as np
import matplotlib.pyplot as plt

# Audio signal (normalized) and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))

# Estimate the background signal and infer the foreground signal
background_signal = repet.adaptive(audio_signal, sample_rate);
foreground_signal = audio_signal-background_signal;

# Write the background and foreground signals (un-normalized)
scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)

# Compute the audio, background, and foreground spectrograms
window_length = repet.windowlength(sample_rate)
window_function = repet.windowfunction(window_length)
step_length = repet.steplength(window_length)
audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])

# Display the audio, background, and foreground spectrograms (up to 5kHz)
plt.rc('font', size=30)
plt.subplot(3, 1, 1)
plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Audio Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 2)
plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Background Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 3)
plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Foreground Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.show()
```

<img src="images/repet_python/repet_adaptive.png" width="1000">

### REPET-SIM
`background_signal = repet.sim(audio_signal, sample_rate)`

Arguments:
```
audio_signal: audio signal [number_samples, number_channels]
sample_rate: sample rate in Hz
background_signal: background signal [number_samples, number_channels]
```

Example: Estimate the background and foreground signals, and display their spectrograms
```
import scipy.io.wavfile
import repet
import numpy as np
import matplotlib.pyplot as plt

# Audio signal (normalized) and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))

# Estimate the background signal and infer the foreground signal
background_signal = repet.sim(audio_signal, sample_rate);
foreground_signal = audio_signal-background_signal;

# Write the background and foreground signals (un-normalized)
scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)

# Compute the audio, background, and foreground spectrograms
window_length = repet.windowlength(sample_rate)
window_function = repet.windowfunction(window_length)
step_length = repet.steplength(window_length)
audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])

# Display the audio, background, and foreground spectrograms (up to 5kHz)
plt.rc('font', size=30)
plt.subplot(3, 1, 1)
plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Audio Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 2)
plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Background Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 3)
plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Foreground Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.show()
```

<img src="images/repet_python/repet_sim.png" width="1000">

### Online REPET-SIM
`background_signal = repet.simonline(audio_signal, sample_rate)`

Arguments:
```
audio_signal: audio signal [number_samples, number_channels]
sample_rate: sample rate in Hz
background_signal: background signal [number_samples, number_channels]
```

Example: Estimate the background and foreground signals, and display their spectrograms
```
import scipy.io.wavfile
import repet
import numpy as np
import matplotlib.pyplot as plt

# Audio signal (normalized) and sample rate in Hz
sample_rate, audio_signal = scipy.io.wavfile.read('audio_file.wav')
audio_signal = audio_signal / (2.0**(audio_signal.itemsize*8-1))

# Estimate the background signal and infer the foreground signal
background_signal = repet.simonline(audio_signal, sample_rate);
foreground_signal = audio_signal-background_signal;

# Write the background and foreground signals (un-normalized)
scipy.io.wavfile.write('background_signal.wav', sample_rate, background_signal)
scipy.io.wavfile.write('foreground_signal.wav', sample_rate, foreground_signal)

# Compute the audio, background, and foreground spectrograms
window_length = repet.windowlength(sample_rate)
window_function = repet.windowfunction(window_length)
step_length = repet.steplength(window_length)
audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])
foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:int(window_length/2)+1, :])

# Display the audio, background, and foreground spectrograms (up to 5kHz)
plt.rc('font', size=30)
plt.subplot(3, 1, 1)
plt.imshow(20*np.log10(audio_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Audio Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 2)
plt.imshow(20*np.log10(background_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Background Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.subplot(3, 1, 3)
plt.imshow(20*np.log10(foreground_spectrogram[1:int(window_length/8), :]), aspect='auto', cmap='jet', origin='lower')
plt.title('Foreground Spectrogram (dB)')
plt.xticks(np.round(np.arange(1, np.floor(len(audio_signal)/sample_rate)+1)*sample_rate/step_length),
        np.arange(1, int(np.floor(len(audio_signal)/sample_rate))+1))
plt.xlabel('Time (s)')
plt.yticks(np.round(np.arange(1e3, int(sample_rate/8)+1, 1e3)/sample_rate*window_length),
        np.arange(1, int(sample_rate/8*1e3)+1))
plt.ylabel('Frequency (kHz)')
plt.show()
```

<img src="images/repet_python/repet_simonline.png" width="1000">


## References

- Zafar Rafii, Antoine Liutkus, and Bryan Pardo. "REPET for Background/Foreground Separation in Audio," *Blind Source Separation*, Springer, Berlin, Heidelberg, 2014. [[article](http://zafarrafii.com/Publications/Rafii-Liutkus-Pardo%20-%20REPET%20for%20Background-Foreground%20Separation%20in%20Audio%20-%202014.pdf)]

- Zafar Rafii and Bryan Pardo. "Online REPET-SIM for Real-time Speech Enhancement," *38th International Conference on Acoustics, Speech and Signal Processing*, Vancouver, BC, Canada, May 26-31, 2013. [[article](http://zafarrafii.com/Publications/Rafii-Pardo%20-%20Online%20REPET-SIM%20for%20Real-time%20Speech%20Enhancement%20-%202013.pdf)][[poster](http://zafarrafii.com/Publications/Rafii-Pardo%20-%20Online%20REPET-SIM%20for%20Real-time%20Speech%20Enhancement%20-%202013%20(poster).pdf)]

- Zafar Rafii and Bryan Pardo. "Audio Separation System and Method," US20130064379 A1, US 13/612,413, March 14, 2013. [[URL](https://www.google.com/patents/US20130064379)]

- Zafar Rafii and Bryan Pardo. "REpeating Pattern Extraction Technique (REPET): A Simple Method for Music/Voice Separation," *IEEE Transactions on Audio, Speech, and Language Processing*, vol. 21, no. 1, January 2013. [[article](http://zafarrafii.com/Publications/Rafii-Pardo%20-%20REpeating%20Pattern%20Extraction%20Technique%20(REPET)%20A%20Simple%20Method%20for%20Music-Voice%20Separation%20-%202013.pdf)]

- Zafar Rafii and Bryan Pardo. "Music/Voice Separation using the Similarity Matrix," *13th International Society on Music Information Retrieval*, Porto, Portugal, October 8-12, 2012. [[article](http://zafarrafii.com/Publications/Rafii-Pardo%20-%20Music-Voice%20Separation%20using%20the%20Similarity%20Matrix%20-%202012.pdf)][[slides](http://zafarrafii.com/Publications/Rafii-Pardo%20-%20Music-Voice%20Separation%20using%20the%20Similarity%20Matrix%20-%202012%20(slides).pdf)]

- Antoine Liutkus, Zafar Rafii, Roland Badeau, Bryan Pardo, and Gaël Richard. "Adaptive Filtering for Music/Voice Separation Exploiting the Repeating Musical Structure," *37th International Conference on Acoustics, Speech and Signal Processing*, Kyoto, Japan, March 25-30, 2012. [[article](http://zafarrafii.com/Publications/Liutkus-Rafii-Badeau-Pardo-Richard%20-%20Adaptive%20Filtering%20for%20Music-Voice%20Separation%20Exploiting%20the%20Repeating%20Musical%20Structure%20-%202012.pdf)][[slides](http://zafarrafii.com/Publications/Liutkus-Rafii-Badeau-Pardo-Richard%20-%20Adaptive%20Filtering%20for%20Music-Voice%20Separation%20Exploiting%20the%20Repeating%20Musical%20Structure%20-%202012%20(slides).pdf)]

- Zafar Rafii and Bryan Pardo. "A Simple Music/Voice Separation Method based on the Extraction of the Repeating Musical Structure," *36th International Conference on Acoustics, Speech and Signal Processing*, Prague, Czech Republic, May 22-27, 2011. [[article](http://zafarrafii.com/Publications/Rafii-Pardo%20-%20A%20Simple%20Music-Voice%20Separation%20Method%20based%20on%20the%20Extraction%20of%20the%20Repeating%20Musical%20Structure%20-%202011.pdf)][[poster](http://zafarrafii.com/Publications/Rafii-Pardo%20-%20A%20Simple%20Music-Voice%20Separation%20Method%20based%20on%20the%20Extraction%20of%20the%20Repeating%20Musical%20Structure%20-%202011%20(poster).pdf)]


## Author

- Zafar Rafii
- zafarrafii@gmail.com
- [Website](http://zafarrafii.com/)
- [CV](http://zafarrafii.com/Zafar%20Rafii%20-%20C.V..pdf)
- [Google Scholar](https://scholar.google.com/citations?user=8wbS2EsAAAAJ&hl=en)
- [LinkedIn](https://www.linkedin.com/in/zafarrafii/)
