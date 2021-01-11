"""
This Python module implements a number of functions for the REpeating Pattern Extraction Technique (REPET).
    Repetition is a fundamental element in generating and perceiving structure. In audio, mixtures are 
    often composed of structures where a repeating background signal is superimposed with a varying 
    foreground signal (e.g., a singer overlaying varying vocals on a repeating accompaniment or a varying 
    speech signal mixed up with a repeating background noise). On this basis, we present the REpeating 
    Pattern Extraction Technique (REPET), a simple approach for separating the repeating background from 
    the non-repeating foreground in an audio mixture. The basic idea is to find the repeating elements in 
    the mixture, derive the underlying repeating models, and extract the repeating background by comparing 
    the models to the mixture. Unlike other separation approaches, REPET does not depend on special 
    parameterizations, does not rely on complex frameworks, and does not require external information. 
    Because it is only based on repetition, it has the advantage of being simple, fast, blind, and 
    therefore completely and easily automatable.

Functions:
    original - Compute the original REPET.
    extended - Compute REPET extended.
    adaptive - Compute the adaptive REPET.
    sim - Compute REPET-SIM.
    simonline - Compute the online REPET-SIM.

Other:
    wavread - Read a WAVE file (using SciPy).
    wavwrite - Write a WAVE file (using SciPy).
    specshow - Display an spectrogram in dB, seconds, and Hz.

Author:
    Zafar Rafii
    zafarrafii@gmail.com
    http://zafarrafii.com
    https://github.com/zafarrafii
    https://www.linkedin.com/in/zafarrafii/
    01/11/21
"""

import numpy as np
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt


# Public variables
# Define the cutoff frequency in Hz for the dual high-pass filter of the foreground (vocals are rarely below 100 Hz)
cutoff_frequency = 100

# Define the period range in seconds for the beat spectrum (for REPET, REPET extented, and the adaptive REPET)
period_range = [1, 10]

# Define the segment length and step in seconds (for REPET extented and the adaptive REPET)
segment_length = 10
segment_step = 5

# Define the filter order for the median filter (for the adaptive REPET)
filter_order = 5

# Define the minimal threshold for two similar frames in [0,1], minimal distance between two similar frames in seconds,
# and maximal number of similar frames for every frame (for REPET-SIM and the online REPET-SIM)
similarity_threshold = 0
similarity_distance = 1
similarity_number = 100

# Define the buffer length in seconds (for the online REPET-SIM)
buffer_length = 10


# Public functions
def original(audio_signal, sampling_frequency):
    """
    Compute the original REPET.
        The original REPET aims at identifying and extracting the repeating patterns in an audio mixture, by estimating
        a period of the underlying repeating structure and modeling a segment of the periodically repeating background.

    Inputs:
        audio_signal: audio signal (number_samples, number_channels)
        sampling_frequency: sampling frequency in Hz
    Output
        background_signal: background signal (number_samples, number_channels)

    Example: Estimate the background and foreground signals, and display their spectrograms.
        # Import the modules
        import numpy as np
        import scipy.signal
        import repet
        import matplotlib.pyplot as plt

        # Read the audio signal (normalized) with its sampling frequency in Hz
        audio_signal, sampling_frequency = repet.wavread("audio_file.wav")

        # Estimate the background signal, and the foreground signal
        background_signal = repet.original(audio_signal, sampling_frequency)
        foreground_signal = audio_signal-background_signal

        # Write the background and foreground signals
        repet.wavwrite(background_signal, sampling_frequency, "background_signal.wav")
        repet.wavwrite(foreground_signal, sampling_frequency, "foreground_signal.wav")

        # Compute the mixture, background, and foreground spectrograms
        window_length = pow(2, int(np.ceil(np.log2(0.04*sampling_frequency))))
        window_function = scipy.signal.hamming(window_length, sym=False)
        step_length = int(window_length/2)
        number_frequencies = int(window_length/2)+1
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:number_frequencies, :])

        # Display the mixture, background, and foreground spectrograms in dB, seconds, and Hz
        time_duration = len(audio_signal)/sampling_frequency
        maximum_frequency = sampling_frequency/8
        xtick_step = 1
        ytick_step = 1000
        plt.figure(figsize=(17, 10))
        plt.subplot(3,1,1)
        repet.specshow(audio_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Audio spectrogram (dB)")
        plt.subplot(3,1,2)
        repet.specshow(background_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Background spectrogram (dB)")
        plt.subplot(3,1,3)
        repet.specshow(foreground_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Foreground spectrogram (dB)")
        plt.show()
    """

    # Get the number of samples and channels in the audio signal
    number_samples, number_channels = np.shape(audio_signal)

    # Set the parameters for the STFT
    # (audio stationary around 40 ms, power of 2 for fast FFT and constant overlap-add (COLA),
    # periodic Hamming window for COLA, and step equal to half the window length for COLA)
    window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
    window_function = scipy.signal.hamming(window_length, sym=False)
    step_length = int(window_length / 2)

    # Derive the number of time frames (given the zero-padding at the start and the end of the signal)
    number_times = (
        int(
            np.ceil(
                (
                    (number_samples + 2 * int(np.floor(window_length / 2)))
                    - window_length
                )
                / step_length
            )
        )
        + 1
    )

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times, number_channels), dtype=complex)

    # Loop over the channels
    for i in range(0, number_channels):

        # Compute the STFT of the current channel
        audio_stft[:, :, i] = _stft(audio_signal[:, i], window_function, step_length)

    # Derive the magnitude spectrogram (with the DC component and without the mirrored frequencies)
    audio_spectrogram = abs(audio_stft[0 : int(window_length / 2) + 1, :, :])

    # Compute the beat spectrum of the spectrograms averaged over the channels
    # (take the square to emphasize peaks of periodicitiy)
    beat_spectrum = _beatspectrum(np.power(np.mean(audio_spectrogram, axis=2), 2))

    # Get the period range in time frames for the beat spectrum
    period_range2 = np.round(
        np.array(period_range) * sampling_frequency / step_length
    ).astype(int)

    # Estimate the repeating period in time frames given the period range
    repeating_period = _periods(beat_spectrum, period_range2)

    # Get the cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = int(
        np.ceil(cutoff_frequency * window_length / sampling_frequency)
    )

    # Initialize the background signal
    background_signal = np.zeros((number_samples, number_channels))

    # Loop over the channels
    for i in range(number_channels):

        # Compute the repeating mask for the current channel given the repeating period
        repeating_mask = _mask(audio_spectrogram[:, :, i], repeating_period)

        # Perform a high-pass filtering of the dual foreground
        repeating_mask[1 : cutoff_frequency2 + 2, :] = 1

        # Recover the mirrored frequencies
        repeating_mask = np.concatenate(
            (repeating_mask, repeating_mask[-2:0:-1, :]), axis=0
        )

        # Synthesize the repeating background for the current channel
        background_signal1 = _istft(
            repeating_mask * audio_stft[:, :, i],
            window_function,
            step_length,
        )

        # Truncate to the original number of samples
        background_signal[:, i] = background_signal1[0:number_samples]

    return background_signal


def extended(audio_signal, sampling_frequency):
    """
    Compute REPET extended.
        The original REPET can be easily extended to handle varying repeating structures, by simply applying the method
        along time, on individual segments or via a sliding window.

    Inputs:
        audio_signal: audio signal (number_samples, number_channels)
        sampling_frequency: sampling frequency in Hz
    Output
        background_signal: background signal (number_samples, number_channels)

    Example: Estimate the background and foreground signals, and display their spectrograms.
        # Import the modules
        import numpy as np
        import scipy.signal
        import repet
        import matplotlib.pyplot as plt

        # Read the audio signal (normalized) with its sampling frequency in Hz
        audio_signal, sampling_frequency = repet.wavread("audio_file.wav")

        # Estimate the background signal, and the foreground signal
        background_signal = repet.extended(audio_signal, sampling_frequency)
        foreground_signal = audio_signal-background_signal

        # Write the background and foreground signals
        repet.wavwrite(background_signal, sampling_frequency, "background_signal.wav")
        repet.wavwrite(foreground_signal, sampling_frequency, "foreground_signal.wav")

        # Compute the mixture, background, and foreground spectrograms
        window_length = pow(2, int(np.ceil(np.log2(0.04*sampling_frequency))))
        window_function = scipy.signal.hamming(window_length, sym=False)
        step_length = int(window_length/2)
        number_frequencies = int(window_length/2)+1
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:number_frequencies, :])

        # Display the mixture, background, and foreground spectrograms in dB, seconds, and Hz
        time_duration = len(audio_signal)/sampling_frequency
        maximum_frequency = sampling_frequency/8
        xtick_step = 1
        ytick_step = 1000
        plt.figure(figsize=(17, 10))
        plt.subplot(3,1,1)
        repet.specshow(audio_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Audio spectrogram (dB)")
        plt.subplot(3,1,2)
        repet.specshow(background_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Background spectrogram (dB)")
        plt.subplot(3,1,3)
        repet.specshow(foreground_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Foreground spectrogram (dB)")
        plt.show()
    """

    # Get the number of samples and channels in the audio signal
    number_samples, number_channels = np.shape(audio_signal)

    # Get the segment length, step, and overlap in samples
    segment_length2 = round(segment_length * sampling_frequency)
    segment_step2 = round(segment_step * sampling_frequency)
    segment_overlap2 = segment_length2 - segment_step2

    # Get the number of segments
    if number_samples < segment_length2 + segment_step2:

        # Use a single segment if the signal is too short
        number_segments = 1

    else:

        # Use multiple segments if the signal is long enough (the last segment could be longer)
        number_segments = 1 + int(
            np.floor((number_samples - segment_length2) / segment_step2)
        )

        # Use a triangular window for the overlapping parts
        segment_window = scipy.signal.triang(2 * segment_overlap2)

    # Set the parameters for the STFT
    # (audio stationary around 40 ms, power of 2 for fast FFT and constant overlap-add (COLA),
    # periodic Hamming window for COLA, and step equal half the window length for COLA)
    window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
    window_function = scipy.signal.hamming(window_length, sym=False)
    step_length = int(window_length / 2)

    # Get the period range in time frames for the beat spectrum
    period_range2 = np.round(
        np.array(period_range) * sampling_frequency / step_length
    ).astype(int)

    # Get the cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = int(
        np.ceil(cutoff_frequency * window_length / sampling_frequency)
    )

    # Initialize background signal
    background_signal = np.zeros((number_samples, number_channels))

    # Loop over the segments
    k = 0
    for j in range(number_segments):

        # Check if there is a single segment or multiple ones
        if number_segments == 1:

            # Use the whole signal as the segment
            audio_segment = audio_signal
            segment_length2 = number_samples

        else:

            # Check if it is one of the first segments (same length) or the last one (could be longer)
            if j < number_segments - 1:
                audio_segment = audio_signal[k : k + segment_length2, :]
            elif j == number_segments - 1:
                audio_segment = audio_signal[k:number_samples, :]
                segment_length2 = len(audio_segment)

        # Get the number of time frames
        number_times = int(
            np.ceil((window_length - step_length + segment_length2) / step_length)
        )

        # Initialize the STFT
        audio_stft = np.zeros(
            (window_length, number_times, number_channels), dtype=complex
        )

        # Loop over the channels
        for i in range(number_channels):

            # Compute the STFT for the current channel
            audio_stft[:, :, i] = _stft(
                audio_segment[:, i], window_function, step_length
            )

        # Derive the magnitude spectrogram (with the DC component and without the mirrored frequencies)
        audio_spectrogram = abs(audio_stft[0 : int(window_length / 2) + 1, :, :])

        # Compute the beat spectrum of the spectrograms averaged over the channels
        # (take the square to emphasize peaks of periodicitiy)
        beat_spectrum = _beatspectrum(np.power(np.mean(audio_spectrogram, axis=2), 2))

        # Estimate the repeating period in time frames given the period range
        repeating_period = _periods(beat_spectrum, period_range2)

        # Initialize the background segment
        background_segment = np.zeros((segment_length2, number_channels))

        # Loop over the channels
        for i in range(number_channels):

            # Compute the repeating mask for the current channel given the repeating period
            repeating_mask = _mask(audio_spectrogram[:, :, i], repeating_period)

            # Perform a high-pass filtering of the dual foreground
            repeating_mask[1 : cutoff_frequency2 + 2, :] = 1

            # Recover the mirrored frequencies
            repeating_mask = np.concatenate(
                (repeating_mask, repeating_mask[-2:0:-1, :])
            )

            # Synthesize the repeating background for the current channel
            background_segment1 = _istft(
                repeating_mask * audio_stft[:, :, i],
                window_function,
                step_length,
            )

            # Truncate to the original number of samples
            background_segment[:, i] = background_segment1[0:segment_length2]

        # Check again if there is a single segment or multiple ones
        if number_segments == 1:

            # Use the segment as the whole signal
            background_signal = background_segment

        else:

            # Check if it is one of the first segments or the last one
            if j == 0:

                # Add the segment to the signal
                background_signal[0:segment_length2, :] = (
                    background_signal[0:segment_length2, :] + background_segment
                )

            elif j <= number_segments - 1:

                # Perform a half windowing of the overlap part of the background signal on the right
                background_signal[k : k + segment_overlap2, :] = (
                    background_signal[k : k + segment_overlap2, :]
                    * segment_window[
                        segment_overlap2 : 2 * segment_overlap2, np.newaxis
                    ]
                )

                # Perform a half windowing of the overlap part of the background segment on the left
                background_segment[0:segment_overlap2, :] = (
                    background_segment[0:segment_overlap2, :]
                    * segment_window[0:segment_overlap2, np.newaxis]
                )

                # Add the segment to the signal
                background_signal[k : k + segment_length2, :] = (
                    background_signal[k : k + segment_length2, :] + background_segment
                )

            # Update the index
            k = k + segment_step2

    return background_signal


def adaptive(audio_signal, sampling_frequency):
    """
    Compute the adaptive REPET.
        The original REPET works well when the repeating background is relatively stable (e.g., a verse or the chorus in
        a song); however, the repeating background can also vary over time (e.g., a verse followed by the chorus in the
        song). The adaptive REPET is an extension of the original REPET that can handle varying repeating structures, by
        estimating the time-varying repeating periods and extracting the repeating background locally, without the need
        for segmentation or windowing.

    Inputs:
        audio_signal: audio signal (number_samples, number_channels)
        sampling_frequency: sampling frequency in Hz
    Output
        background_signal: background signal (number_samples, number_channels)

    Example: Estimate the background and foreground signals, and display their spectrograms.
        # Import the modules
        import numpy as np
        import scipy.signal
        import repet
        import matplotlib.pyplot as plt

        # Read the audio signal (normalized) with its sampling frequency in Hz
        audio_signal, sampling_frequency = repet.wavread("audio_file.wav")

        # Estimate the background signal, and the foreground signal
        background_signal = repet.adaptive(audio_signal, sampling_frequency)
        foreground_signal = audio_signal-background_signal

        # Write the background and foreground signals
        repet.wavwrite(background_signal, sampling_frequency, "background_signal.wav")
        repet.wavwrite(foreground_signal, sampling_frequency, "foreground_signal.wav")

        # Compute the mixture, background, and foreground spectrograms
        window_length = pow(2, int(np.ceil(np.log2(0.04*sampling_frequency))))
        window_function = scipy.signal.hamming(window_length, sym=False)
        step_length = int(window_length/2)
        number_frequencies = int(window_length/2)+1
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:number_frequencies, :])

        # Display the mixture, background, and foreground spectrograms in dB, seconds, and Hz
        time_duration = len(audio_signal)/sampling_frequency
        maximum_frequency = sampling_frequency/8
        xtick_step = 1
        ytick_step = 1000
        plt.figure(figsize=(17, 10))
        plt.subplot(3,1,1)
        repet.specshow(audio_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Audio spectrogram (dB)")
        plt.subplot(3,1,2)
        repet.specshow(background_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Background spectrogram (dB)")
        plt.subplot(3,1,3)
        repet.specshow(foreground_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Foreground spectrogram (dB)")
        plt.show()
    """

    # Get the number of samples and channels in the audio signal
    number_samples, number_channels = np.shape(audio_signal)

    # Set the parameters for the STFT
    # (audio stationary around 40 ms, power of 2 for fast FFT and constant overlap-add (COLA),
    # periodic Hamming window for COLA, and step equal to half the window length for COLA)
    window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
    window_function = scipy.signal.hamming(window_length, sym=False)
    step_length = int(window_length / 2)

    # Derive the number of time frames (given the zero-padding at the start and the end of the signal)
    number_times = (
        int(
            np.ceil(
                (
                    (number_samples + 2 * int(np.floor(window_length / 2)))
                    - window_length
                )
                / step_length
            )
        )
        + 1
    )

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times, number_channels), dtype=complex)

    # Loop over the channels
    for i in range(0, number_channels):

        # Compute the STFT of the current channel
        audio_stft[:, :, i] = _stft(audio_signal[:, i], window_function, step_length)

    # Derive the magnitude spectrogram (with the DC component and without the mirrored frequencies)
    audio_spectrogram = abs(audio_stft[0 : int(window_length / 2) + 1, :, :])

    # Get the segment length and step in time frames for the beat spectrogram
    segment_length2 = int(round(segment_length * sampling_frequency / step_length))
    segment_step2 = int(round(segment_step * sampling_frequency / step_length))

    # Compute the beat spectrogram of the spectrograms averaged over the channels
    # (take the square to emphasize peaks of periodicitiy)
    beat_spectrogram = _beatspectrogram(
        np.power(np.mean(audio_spectrogram, axis=2), 2), segment_length2, segment_step2
    )

    # Get the period range in time frames
    period_range2 = np.round(
        np.array(period_range) * sampling_frequency / step_length
    ).astype(int)

    # Estimate the repeating periods in time frames given the period range
    repeating_periods = _periods(beat_spectrogram, period_range2)

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = (
        int(np.ceil(cutoff_frequency * (window_length - 1) / sampling_frequency)) - 1
    )

    # Initialize the background signal
    background_signal = np.zeros((number_samples, number_channels))

    # Loop over the channels
    for i in range(0, number_channels):

        # Compute the repeating mask for the current channel given the repeating periods
        repeating_mask = _adaptivemask(
            audio_spectrogram[:, :, i], repeating_periods, filter_order
        )

        # Perform a high-pass filtering of the dual foreground
        repeating_mask[1 : cutoff_frequency2 + 2, :] = 1

        # Recover the mirrored frequencies
        repeating_mask = np.concatenate(
            (repeating_mask, repeating_mask[-2:0:-1, :]), axis=0
        )

        # Synthesize the repeating background for the current channel
        background_signal1 = _istft(
            repeating_mask * audio_stft[:, :, i],
            window_function,
            step_length,
        )

        # Truncate to the original number of samples
        background_signal[:, i] = background_signal1[0:number_samples]

    return background_signal


def sim(audio_signal, sampling_frequency):
    """
    Compute REPET-SIM.
        The REPET methods work well when the repeating background has periodically repeating patterns (e.g., jackhammer
        noise); however, the repeating patterns can also happen intermittently or without a global or local periodicity
        (e.g., frogs by a pond). REPET-SIM is a generalization of REPET that can also handle non-periodically repeating
        structures, by using a similarity matrix to identify the repeating elements.

    Inputs:
        audio_signal: audio signal (number_samples, number_channels)
        sampling_frequency: sampling frequency in Hz
    Output
        background_signal: background signal (number_samples, number_channels)

    Example: Estimate the background and foreground signals, and display their spectrograms.
        # Import the modules
        import numpy as np
        import scipy.signal
        import repet
        import matplotlib.pyplot as plt

        # Read the audio signal (normalized) with its sampling frequency in Hz
        audio_signal, sampling_frequency = repet.wavread("audio_file.wav")

        # Estimate the background signal, and the foreground signal
        background_signal = repet.sim(audio_signal, sampling_frequency)
        foreground_signal = audio_signal-background_signal

        # Write the background and foreground signals
        repet.wavwrite(background_signal, sampling_frequency, "background_signal.wav")
        repet.wavwrite(foreground_signal, sampling_frequency, "foreground_signal.wav")

        # Compute the mixture, background, and foreground spectrograms
        window_length = pow(2, int(np.ceil(np.log2(0.04*sampling_frequency))))
        window_function = scipy.signal.hamming(window_length, sym=False)
        step_length = int(window_length/2)
        number_frequencies = int(window_length/2)+1
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:number_frequencies, :])

        # Display the mixture, background, and foreground spectrograms in dB, seconds, and Hz
        time_duration = len(audio_signal)/sampling_frequency
        maximum_frequency = sampling_frequency/8
        xtick_step = 1
        ytick_step = 1000
        plt.figure(figsize=(17, 10))
        plt.subplot(3,1,1)
        repet.specshow(audio_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Audio spectrogram (dB)")
        plt.subplot(3,1,2)
        repet.specshow(background_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Background spectrogram (dB)")
        plt.subplot(3,1,3)
        repet.specshow(foreground_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Foreground spectrogram (dB)")
        plt.show()
    """

    # Number of samples and channels
    number_samples, number_channels = np.shape(audio_signal)

    # Set the parameters for the STFT
    # (audio stationary around 40 ms, power of 2 for fast FFT and constant overlap-add (COLA),
    # periodic Hamming window for COLA, and step equal half the window length for COLA)
    window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
    window_function = scipy.signal.hamming(window_length, sym=False)
    step_length = int(window_length / 2)

    # Number of time frames
    number_times = int(
        np.ceil((window_length - step_length + number_samples) / step_length)
    )

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times, number_channels), dtype=complex)

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # STFT of the current channel
        audio_stft[:, :, channel_index] = _stft(
            audio_signal[:, channel_index], window_function, step_length
        )

    # Magnitude spectrogram (with DC component and without mirrored frequencies)
    audio_spectrogram = abs(audio_stft[0 : int(window_length / 2) + 1, :, :])

    # Self-similarity of the spectrograms averaged over the channels
    similarity_matrix = _selfsimilaritymatrix(np.mean(audio_spectrogram, axis=2))

    # Similarity distance in time frames
    similarity_distance2 = int(
        round(similarity_distance * sampling_frequency / step_length)
    )

    # Similarity indices for all the frames
    similarity_indices = _indices(
        similarity_matrix, similarity_threshold, similarity_distance2, similarity_number
    )

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = (
        int(np.ceil(cutoff_frequency * (window_length - 1) / sampling_frequency)) - 1
    )

    # Initialize the background signal
    background_signal = np.zeros((number_samples, number_channels))

    # Loop over the channels
    for channel_index in range(0, number_channels):

        # Repeating mask for the current channel
        repeating_mask = _simmask(
            audio_spectrogram[:, :, channel_index], similarity_indices
        )

        # High-pass filtering of the dual foreground
        repeating_mask[1 : cutoff_frequency2 + 2, :] = 1

        # Mirror the frequency channels
        repeating_mask = np.concatenate((repeating_mask, repeating_mask[-2:0:-1, :]))

        # Estimated repeating background for the current channel
        background_signal1 = _istft(
            repeating_mask * audio_stft[:, :, channel_index],
            window_function,
            step_length,
        )

        # Truncate to the original number of samples
        background_signal[:, channel_index] = background_signal1[0:number_samples]

    return background_signal


def simonline(audio_signal, sampling_frequency):
    """
    Compute the online REPET-SIM.
        REPET-SIM can be easily implemented online to handle real-time computing, particularly for real-time speech
        enhancement. The online REPET-SIM simply processes the time frames of the mixture one after the other given a
        buffer that temporally stores past frames.

     Inputs:
        audio_signal: audio signal (number_samples, number_channels)
        sampling_frequency: sampling frequency in Hz
    Output
        background_signal: background signal (number_samples, number_channels)

    Example: Estimate the background and foreground signals, and display their spectrograms.
        # Import the modules
        import numpy as np
        import scipy.signal
        import repet
        import matplotlib.pyplot as plt

        # Read the audio signal (normalized) with its sampling frequency in Hz
        audio_signal, sampling_frequency = repet.wavread("audio_file.wav")

        # Estimate the background signal, and the foreground signal
        background_signal = repet.simonline(audio_signal, sampling_frequency)
        foreground_signal = audio_signal-background_signal

        # Write the background and foreground signals
        repet.wavwrite(background_signal, sampling_frequency, "background_signal.wav")
        repet.wavwrite(foreground_signal, sampling_frequency, "foreground_signal.wav")

        # Compute the mixture, background, and foreground spectrograms
        window_length = pow(2, int(np.ceil(np.log2(0.04*sampling_frequency))))
        window_function = scipy.signal.hamming(window_length, sym=False)
        step_length = int(window_length/2)
        number_frequencies = int(window_length/2)+1
        audio_spectrogram = abs(repet._stft(np.mean(audio_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        background_spectrogram = abs(repet._stft(np.mean(background_signal, axis=1), window_function, step_length)[0:number_frequencies, :])
        foreground_spectrogram = abs(repet._stft(np.mean(foreground_signal, axis=1), window_function, step_length)[0:number_frequencies, :])

        # Display the mixture, background, and foreground spectrograms in dB, seconds, and Hz
        time_duration = len(audio_signal)/sampling_frequency
        maximum_frequency = sampling_frequency/8
        xtick_step = 1
        ytick_step = 1000
        plt.figure(figsize=(17, 10))
        plt.subplot(3,1,1)
        repet.specshow(audio_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Audio spectrogram (dB)")
        plt.subplot(3,1,2)
        repet.specshow(background_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Background spectrogram (dB)")
        plt.subplot(3,1,3)
        repet.specshow(foreground_spectrogram[0:int(window_length/8), :], time_duration, maximum_frequency, xtick_step, ytick_step)
        plt.title("Foreground spectrogram (dB)")
        plt.show()
    """

    # Number of samples and channels
    number_samples, number_channels = np.shape(audio_signal)

    # Set the parameters for the STFT
    # (audio stationary around 40 ms, power of 2 for fast FFT and constant overlap-add (COLA),
    # periodic Hamming window for COLA, and step equal half the window length for COLA)
    window_length = pow(2, int(np.ceil(np.log2(0.04 * sampling_frequency))))
    window_function = scipy.signal.hamming(window_length, sym=False)
    step_length = int(window_length / 2)

    # Number of time frames
    number_times = int(np.ceil((number_samples - window_length) / step_length + 1))

    # Buffer length in time frames
    buffer_length2 = int(
        round((buffer_length * sampling_frequency - window_length) / step_length + 1)
    )

    # Initialize the buffer spectrogram
    buffer_spectrogram = np.zeros(
        (int(window_length / 2 + 1), buffer_length2, number_channels)
    )

    # Loop over the time frames to compute the buffer spectrogram (the last frame will be the frame to be processed)
    for time_index in range(0, buffer_length2 - 1):

        # Sample index in the signal
        sample_index = step_length * time_index

        # Loop over the channels
        for channel_index in range(0, number_channels):

            # Compute the FT of the segment
            buffer_ft = np.fft.fft(
                audio_signal[sample_index : window_length + sample_index, channel_index]
                * window_function,
                axis=0,
            )

            # Derive the spectrum of the frame
            buffer_spectrogram[:, time_index, channel_index] = abs(
                buffer_ft[0 : int(window_length / 2 + 1)]
            )

    # Zero-pad the audio signal at the end
    audio_signal = np.pad(
        audio_signal,
        (0, (number_times - 1) * step_length + window_length - number_samples),
        "constant",
        constant_values=0,
    )

    # Similarity distance in time frames
    similarity_distance2 = int(round(similarity_distance * sample_rate / step_length))

    # Cutoff frequency in frequency channels for the dual high-pass filter of the foreground
    cutoff_frequency2 = (
        int(np.ceil(cutoff_frequency * (window_length - 1) / sample_rate)) - 1
    )

    # Initialize the background signal
    background_signal = np.zeros(
        ((number_times - 1) * step_length + window_length, number_channels)
    )

    # Loop over the time frames to compute the background signal
    for time_index in range(buffer_length2 - 1, number_times):

        # Sample index in the signal
        sample_index = step_length * time_index

        # Time index of the current frame
        current_index = time_index % buffer_length2

        # Initialize the FT of the current segment
        current_ft = np.zeros((window_length, number_channels), dtype=complex)

        # Loop over the channels
        for channel_index in range(0, number_channels):

            # Compute the FT of the current segment
            current_ft[:, channel_index] = np.fft.fft(
                audio_signal[sample_index : window_length + sample_index, channel_index]
                * window_function,
                axis=0,
            )

            # Derive the spectrum of the current frame and update the buffer spectrogram
            buffer_spectrogram[:, current_index, channel_index] = np.abs(
                current_ft[0 : int(window_length / 2 + 1), channel_index]
            )

        # Cosine similarity between the spectrum of the current frame and the past frames, for all the channels
        similarity_vector = _similaritymatrix(
            np.mean(buffer_spectrogram, axis=2),
            np.mean(
                buffer_spectrogram[:, current_index : current_index + 1, :], axis=2
            ),
        )

        # Indices of the similar frames
        _, similarity_indices = _localmaxima(
            similarity_vector[:, 0],
            similarity_threshold,
            similarity_distance2,
            similarity_number,
        )

        # Loop over the channels
        for channel_index in range(0, number_channels):

            # Compute the repeating spectrum for the current frame
            repeating_spectrum = np.median(
                buffer_spectrogram[:, similarity_indices, channel_index], axis=1
            )

            # Refine the repeating spectrum
            repeating_spectrum = np.minimum(
                repeating_spectrum, buffer_spectrogram[:, current_index, channel_index]
            )

            # Derive the repeating mask for the current frame
            repeating_mask = (repeating_spectrum + np.finfo(float).eps) / (
                buffer_spectrogram[:, current_index, channel_index]
                + np.finfo(float).eps
            )

            # High-pass filtering of the dual foreground
            repeating_mask[1 : cutoff_frequency2 + 2] = 1

            # Mirror the frequency channels
            repeating_mask = np.concatenate((repeating_mask, repeating_mask[-2:0:-1]))

            # Apply the mask to the FT of the current segment
            background_ft = repeating_mask * current_ft[:, channel_index]

            # Inverse FT of the current segment
            background_signal[
                sample_index : window_length + sample_index, channel_index
            ] = background_signal[
                sample_index : window_length + sample_index, channel_index
            ] + np.real(
                np.fft.ifft(background_ft, axis=0)
            )

    # Truncate the signal to the original length
    background_signal = background_signal[0:number_samples, :]

    # Un-window the signal (just in case)
    background_signal = background_signal / sum(
        window_function[0:window_length:step_length]
    )

    return background_signal


def wavread(audio_file):
    """
    Read a WAVE file (using SciPy).

    Input:
        audio_file: path to an audio file
    Outputs:
        audio_signal: audio signal (number_samples, number_channels)
        sampling_frequency: sampling frequency in Hz
    """

    # Read the audio file and return the sampling frequency in Hz and the non-normalized signal using SciPy
    sampling_frequency, audio_signal = scipy.io.wavfile.read(audio_file)

    # Normalize the signal by the data range given the size of an item in bytes
    audio_signal = audio_signal / pow(2, audio_signal.itemsize * 8 - 1)

    return audio_signal, sampling_frequency


def wavwrite(audio_signal, sampling_frequency, audio_file):
    """
    Write a WAVE file (using Scipy).

    Inputs:
        audio_signal: audio signal (number_samples, number_channels)
        sampling_frequency: sampling frequency in Hz
    Output:
        audio_file: path to an audio file
    """

    # Write the audio signal using SciPy
    scipy.io.wavfile.write(audio_file, sampling_frequency, audio_signal)


def specshow(
    audio_spectrogram,
    time_duration,
    maximum_frequency,
    xtick_step=1,
    ytick_step=1000,
):
    """
    Display a spectrogram in dB, seconds, and Hz.

    Inputs:
        audio_spectrogram: audio spectrogram (without DC and mirrored frequencies) (number_frequencies, number_times)
        time_duration: time duration of the spectrogram in seconds
        maximum_frequency: maximum frequency in the spectrogram in Hz
        xtick_step: step for the x-axis ticks in seconds (default: 1 second)
        ytick_step: step for the y-axis ticks in Hz (default: 1000 Hz)
    """

    # Get the number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Derive the number of time frames per second and the number of frequency channels per Hz
    time_resolution = number_times / time_duration
    frequency_resolution = number_frequencies / maximum_frequency

    # Prepare the tick locations and labels for the x-axis
    xtick_locations = np.arange(
        xtick_step * time_resolution,
        number_times,
        xtick_step * time_resolution,
    )
    xtick_labels = np.arange(xtick_step, time_duration, xtick_step).astype(int)

    # Prepare the tick locations and labels for the y-axis
    ytick_locations = np.arange(
        ytick_step * frequency_resolution,
        number_frequencies,
        ytick_step * frequency_resolution,
    )
    ytick_labels = np.arange(ytick_step, maximum_frequency, ytick_step).astype(int)

    # Display the spectrogram in dB, seconds, and Hz
    plt.imshow(
        20 * np.log10(audio_spectrogram), aspect="auto", cmap="jet", origin="lower"
    )
    plt.xticks(ticks=xtick_locations, labels=xtick_labels)
    plt.yticks(ticks=ytick_locations, labels=ytick_labels)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")


# Private functions
def _stft(audio_signal, window_function, step_length):
    """
    Compute the short-time Fourier transform (STFT)

    Inputs:
        audio_signal: audio signal (number_samples,)
        window_function: window function (window_length,)
        step_length: step length in samples
    Output:
        audio_stft: audio STFT (window_length, number_frames)

    """
    # Get the number of samples and the window length in samples
    number_samples = len(audio_signal)
    window_length = len(window_function)

    # Derive the zero-padding length at the start and at the end of the signal to center the windows
    padding_length = int(np.floor(window_length / 2))

    # Compute the number of time frames given the zero-padding at the start and at the end of the signal
    number_times = (
        int(
            np.ceil(
                ((number_samples + 2 * padding_length) - window_length) / step_length
            )
        )
        + 1
    )

    # Zero-pad the start and the end of the signal to center the windows
    audio_signal = np.pad(
        audio_signal,
        (
            padding_length,
            (
                number_times * step_length
                + (window_length - step_length)
                - padding_length
            )
            - number_samples,
        ),
        "constant",
        constant_values=0,
    )

    # Initialize the STFT
    audio_stft = np.zeros((window_length, number_times))

    # Loop over the time frames
    i = 0
    for j in range(number_times):

        # Window the signal
        audio_stft[:, j] = audio_signal[i : i + window_length] * window_function
        i = i + step_length

    # Compute the Fourier transform of the frames using the FFT
    audio_stft = np.fft.fft(audio_stft, axis=0)

    return audio_stft


def _istft(audio_stft, window_function, step_length):
    """
    Compute the inverse short-time Fourier transform (STFT).

    Inputs:
        audio_stft: audio STFT (window_length, number_frames)
        window_function: window function (window_length,)
        step_length: step length in samples
    Output:
        audio_signal: audio signal (number_samples,)
    """

    # Get the window length in samples and the number of time frames
    window_length, number_times = np.shape(audio_stft)

    # Compute the number of samples for the signal
    number_samples = number_times * step_length + (window_length - step_length)

    # Initialize the signal
    audio_signal = np.zeros(number_samples)

    # Compute the inverse Fourier transform of the frames and take the real part to ensure real values
    audio_stft = np.real(np.fft.ifft(audio_stft, axis=0))

    # Loop over the time frames
    i = 0
    for j in range(number_times):

        # Perform a constant overlap-add (COLA) of the signal (with proper window function and step length)
        audio_signal[i : i + window_length] = (
            audio_signal[i : i + window_length] + audio_stft[:, j]
        )
        i = i + step_length

    # Remove the zero-padding at the start and at the end of the signal
    audio_signal = audio_signal[
        window_length - step_length : number_samples - (window_length - step_length)
    ]

    # Normalize the signal by the gain introduced by the COLA (if any)
    audio_signal = audio_signal / sum(window_function[0:window_length:step_length])

    return audio_signal


def _acorr(data_matrix):
    """
    Compute the autocorrelation of the columns in a matrix using the Wiener–Khinchin theorem.

    Input:
        data_matrix: data matrix (number_points, number_columns)
    Output:
        autocorrelation_matrix: autocorrelation matrix (number_lags, number_columns)
    """

    # Get the number of points in each column
    number_points = data_matrix.shape[0]

    # Compute the power spectral density (PSD) of the columns
    # (with zero-padding for proper autocorrelation)
    data_matrix = np.power(
        np.abs(np.fft.fft(data_matrix, n=2 * number_points, axis=0)), 2
    )

    # Compute the autocorrelation using the Wiener–Khinchin theorem
    # (the PSD equals the Fourier transform of the autocorrelation)
    autocorrelation_matrix = np.real(np.fft.ifft(data_matrix, axis=0))

    # Discard the symmetric part
    autocorrelation_matrix = autocorrelation_matrix[0:number_points, :]

    # Derive the unbiased autocorrelation
    autocorrelation_matrix = np.divide(
        autocorrelation_matrix, np.arange(number_points, 0, -1)[:, np.newaxis]
    )

    return autocorrelation_matrix


def _beatspectrum(audio_spectrogram):
    """
    Compute the beat spectrum using autocorrelation.

    Input:
        audio_spectrogram: audio spectrogram (number_frequencies, number_times)
    Output:
        beat_spectrum: beat spectrum (number_lags,)
    """

    # Compute the autocorrelation over times for every frequency channel
    beat_spectrum = _acorr(audio_spectrogram.T)

    # Take the mean over the frequency channels
    beat_spectrum = np.mean(beat_spectrum, axis=1)

    return beat_spectrum


def _beatspectrogram(audio_spectrogram, segment_length, segment_step):
    """
    Compute the beat spectrogram using the beat sectrum.

    Input:
        audio_spectrogram: audio spectrogram (number_frequencies, number_times)
        segment_length: segment length in seconds for the segmentation
        segment_step: step length in seconds for the segmentation
    Output:
        beat_spectrogram: beat spectrogram (number_lags, number_times)
    """

    # Get the number of time frames
    number_times = np.shape(audio_spectrogram)[1]

    # Zero-pad the audio spectrogram to center the segments
    audio_spectrogram = np.pad(
        audio_spectrogram,
        (
            (0, 0),
            (
                int(np.ceil((segment_length - 1) / 2)),
                int(np.floor((segment_length - 1) / 2)),
            ),
        ),
        "constant",
        constant_values=0,
    )

    # Initialize the beat spectrogram
    beat_spectrogram = np.zeros((segment_length, number_times))

    # Loop over the time frames every segment step (including the last one)
    for i in range(0, number_times, segment_step):

        # Compute the beat spectrum of the centered audio spectrogram segment
        beat_spectrogram[:, i] = _beatspectrum(
            audio_spectrogram[:, i : i + segment_length]
        )

        # Duplicate the values between segment steps
        # (for display only; they are actually not needed for the adaptive REPET)
        beat_spectrogram[
            :, i : min(i + segment_step - 1, number_times)
        ] = beat_spectrogram[:, i : i + 1]

    return beat_spectrogram


def _selfsimilaritymatrix(data_matrix):
    """Self-similarity matrix using the cosine similarity"""

    # Divide each column by its Euclidean norm
    data_matrix = data_matrix / np.sqrt(sum(np.power(data_matrix, 2), 0))

    # Multiply each normalized columns with each other
    similarity_matrix = np.matmul(data_matrix.T, data_matrix)

    return similarity_matrix


def _similaritymatrix(data_matrix1, data_matrix2):
    """Similarity matrix using the cosine similarity"""

    # Divide each column by its Euclidean norm
    data_matrix1 = data_matrix1 / np.sqrt(sum(np.power(data_matrix1, 2), 0))
    data_matrix2 = data_matrix2 / np.sqrt(sum(np.power(data_matrix2, 2), 0))

    # Multiply each normalized columns with each other
    similarity_matrix = np.matmul(data_matrix1.T, data_matrix2)

    return similarity_matrix


def _periods(beat_spectrogram, period_range):
    """
    Compute the repeating period(s) from the beat spectrogram(spectrum) given a period range.

    Input:
        beat_spectrogram: beat spectrogram (or spectrum) (number_frequencies, number_lags) (or (number_lags, ))
    Output:
        repeating_periods: repeating period(s) in lags (number_periods,) (or scalar)
    """

    # If beat spectrum, compute the repeating period as its argmax given the period range
    # (should be less than a third of the length to have at least 3 segments for the median filter)
    if beat_spectrogram.ndim == 1:
        repeating_periods = (
            np.argmax(
                beat_spectrogram[
                    period_range[0] : min(
                        period_range[1], int(np.floor(beat_spectrogram.shape[0] / 3))
                    )
                ]
            )
            + 1
        )

    # Else, compute the repeating periods as the argmax of the frequency channels given the period range
    else:
        repeating_periods = (
            np.argmax(
                beat_spectrogram[
                    period_range[0] : min(
                        period_range[1], int(np.floor(beat_spectrogram.shape[0] / 3))
                    ),
                    :,
                ],
                axis=0,
            )
            + 1
        )

    # Re-adjust the indices
    repeating_periods = repeating_periods + period_range[0]

    return repeating_periods


def _localmaxima(data_vector, minimum_value, minimum_distance, number_values):
    """Local maxima, values and indices"""

    # Number of data points
    number_data = len(data_vector)

    # Initialize maximum indices
    maximum_indices = np.array([], dtype=int)

    # Loop over the data points
    for data_index in range(0, number_data):

        # The local maximum should be greater than the maximum value
        if data_vector[data_index] >= minimum_value:

            # The local maximum should be strictly greater than the neighboring data points within +- minimum distance
            if all(
                data_vector[data_index]
                > data_vector[max(data_index - minimum_distance, 0) : data_index]
            ) and all(
                data_vector[data_index]
                > data_vector[
                    data_index + 1 : min(data_index + minimum_distance + 1, number_data)
                ]
            ):

                # Save the maximum index
                maximum_indices = np.append(maximum_indices, data_index)

    # Sort the maximum values in descending order
    maximum_values = data_vector[maximum_indices]
    sort_indices = np.argsort(maximum_values)[::-1]

    # Keep only the top maximum values and indices
    number_values = min(number_values, len(maximum_values))
    maximum_values = maximum_values[0:number_values]
    maximum_indices = maximum_indices[sort_indices[0:number_values]].astype(int)

    return maximum_values, maximum_indices


def _indices(
    similarity_matrix, similarity_threshold, similarity_distance, similarity_number
):
    """Similarity indices from the similarity matrix"""

    # Number of time frames
    number_times = similarity_matrix.shape[0]

    # Initialize the similarity indices
    similarity_indices = [None] * number_times

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Indices of the local maxima
        _, maximum_indices = _localmaxima(
            similarity_matrix[:, time_index],
            similarity_threshold,
            similarity_distance,
            similarity_number,
        )

        # Similarity indices for the current time frame
        similarity_indices[time_index] = maximum_indices

    return similarity_indices


def _mask(audio_spectrogram, repeating_period):
    """
    Compute the repeating mask for REPET.

    Input:
        audio_spectrogram: audio spectrogram (number_frequencies, number_times)
    Output:
        repeating_period: repeating period in lag
    """

    # Get the number of frequency channels and time frames in the spectrogram
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Estimate the number of segments (including the last partial one)
    number_segments = int(np.ceil(number_times / repeating_period))

    # Zero-pad the end of the spectrogram to have a full last segment
    audio_spectrogram = np.pad(
        audio_spectrogram,
        ((0, 0), (0, number_segments * repeating_period - number_times)),
        "constant",
        constant_values=0,
    )

    # Reshape the zero-padded spectrogram to a tensor of size (number_frequencies, number_times, number_segments)
    audio_spectrogram = np.reshape(
        audio_spectrogram,
        (number_frequencies, repeating_period, number_segments),
        order="F",
    )

    # Compute the repeating segment by taking the median over the segments, not accounting for the last zeros
    repeating_segment = np.concatenate(
        (
            np.median(
                audio_spectrogram[
                    :, 0 : number_times - (number_segments - 1) * repeating_period, :
                ],
                2,
            ),
            np.median(
                audio_spectrogram[
                    :,
                    number_times
                    - (number_segments - 1) * repeating_period : repeating_period,
                    0 : number_segments - 1,
                ],
                2,
            ),
        ),
        1,
    )

    # Derive the repeating spectrogram by ensuring it has less energy than the original spectrogram
    repeating_spectrogram = np.minimum(
        audio_spectrogram, repeating_segment[:, :, np.newaxis]
    )

    # Derive the repeating mask by normalizing the repeating spectrogram by the original spectrogram
    repeating_mask = (repeating_spectrogram + np.finfo(float).eps) / (
        audio_spectrogram + np.finfo(float).eps
    )

    # Reshape the repeating mask into (number_frequencies, number_times) and truncate to the original number of time frames
    repeating_mask = np.reshape(
        repeating_mask,
        (number_frequencies, number_segments * repeating_period),
        order="F",
    )
    repeating_mask = repeating_mask[:, 0:number_times]

    return repeating_mask


def _adaptivemask(audio_spectrogram, repeating_periods, filter_order):
    """
    Compute the repeating mask for the adaptive REPET.

    Input:
        audio_spectrogram: audio spectrogram (number_frequencies, number_times)
    Output:
        repeating_periods: repeating periods in lag
        filter_order: filter order for the median filter in number of time frames
    """

    # Get the number of frequency channels and time frames in the spectrogram
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Derive the indices of the center frames for the median filter given the filter order
    # (3 => [-1, 0, 1], 4 => [-1, 0, 1, 2], etc.)
    center_indices = np.arange(1, filter_order + 1) - int(np.ceil(filter_order / 2))

    # Initialize the repeating spectrogram
    repeating_spectrogram = np.zeros((number_frequencies, number_times))

    # Loop over the time frames
    for i in range(0, number_times):

        # Derive the indices of all the frames for the median filter
        # given the repeating period for the current frame in the spectrogram
        all_indices = i + center_indices * repeating_periods[i]

        # Discard the indices that are out-of-range
        all_indices = all_indices[
            np.logical_and(all_indices >= 0, all_indices < number_times)
        ]

        # Compute the median filter for the current frame in the spectrogram
        repeating_spectrogram[:, i] = np.median(
            audio_spectrogram[:, all_indices], axis=1
        )

    # Refine the repeating spectrogram by ensuring it has less energy than the original spectrogram
    repeating_spectrogram = np.minimum(audio_spectrogram, repeating_spectrogram)

    # Derive the repeating mask by normalizing the repeating spectrogram by the original spectrogram
    repeating_mask = (repeating_spectrogram + np.finfo(float).eps) / (
        audio_spectrogram + np.finfo(float).eps
    )

    return repeating_mask


def _simmask(audio_spectrogram, similarity_indices):
    """Repeating mask for the REPET-SIM"""

    # Number of frequency channels and time frames
    number_frequencies, number_times = np.shape(audio_spectrogram)

    # Initialize the repeating spectrogram
    repeating_spectrogram = np.zeros((number_frequencies, number_times))

    # Loop over the time frames
    for time_index in range(0, number_times):

        # Indices of the frames for the median filter
        time_indices = similarity_indices[time_index]

        # Median filter on the current time frame
        repeating_spectrogram[:, time_index] = np.median(
            audio_spectrogram[:, time_indices], 1
        )

    # Make sure the energy in the repeating spectrogram is smaller than in the audio spectrogram, for every
    # time-frequency bin
    repeating_spectrogram = np.minimum(audio_spectrogram, repeating_spectrogram)

    # Derive the repeating mask by normalizing the repeating spectrogram by the audio spectrogram
    repeating_mask = (repeating_spectrogram + np.finfo(float).eps) / (
        audio_spectrogram + np.finfo(float).eps
    )

    return repeating_mask
