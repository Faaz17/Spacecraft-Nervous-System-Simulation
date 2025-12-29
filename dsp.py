"""
=============================================================================
DIGITAL SIGNAL PROCESSING MODULE - Filtering & Noise Reduction
=============================================================================

This module provides signal processing utilities for spacecraft telemetry.
The primary tool is a configurable Low-Pass Filter designed to remove
high-frequency noise while preserving the low-frequency structural
vibration signals we care about.

FILTER THEORY BACKGROUND:
-------------------------
A Butterworth filter is chosen because it has:
1. Maximally flat frequency response in the passband (no ripples)
2. Monotonically decreasing response in the stopband
3. Good phase linearity (important for preserving waveform shape)

We use `filtfilt` (forward-backward filtering) to achieve:
- Zero phase distortion (critical for time-aligned fusion)
- Doubled effective filter order
- No time delay in the output signal

WHY LOW-PASS FILTERING?
-----------------------
Our ground truth signal is 1 Hz. High-frequency components in the sensor
data are almost certainly noise, not signal. By cutting off frequencies
above 5 Hz (default), we significantly improve SNR without losing
information about the structural vibration.

Author: Spacecraft Nervous System Simulation Team
Course: Quantum-IoT Hybrid Network for Spacecraft Decision Systems
=============================================================================
"""

import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple


class LowPassFilter:
    """
    A configurable Butterworth Low-Pass Filter for telemetry processing.
    
    This filter removes high-frequency noise while preserving the underlying
    low-frequency signal content. It uses zero-phase filtering to avoid
    introducing time delays that would complicate sensor fusion.
    
    Attributes
    ----------
    cutoff_freq : float
        The -3dB cutoff frequency in Hz
    sample_rate : float
        The sampling rate of the input signal in Hz
    order : int
        Filter order (higher = sharper cutoff, but more ringing potential)
    b, a : np.ndarray
        Filter coefficients (numerator and denominator)
    
    Example
    -------
    >>> lpf = LowPassFilter(cutoff_freq=5.0, sample_rate=100)
    >>> filtered_signal = lpf.apply(noisy_signal)
    """
    
    def __init__(
        self,
        cutoff_freq: float = 5.0,
        sample_rate: float = 100.0,
        order: int = 4
    ):
        """
        Initialize the Low-Pass Filter.
        
        Parameters
        ----------
        cutoff_freq : float
            Cutoff frequency in Hz (default: 5 Hz)
            Frequencies above this will be attenuated
        sample_rate : float
            Sampling rate of input signals in Hz (default: 100 Hz)
            Must match the actual data sampling rate!
        order : int
            Butterworth filter order (default: 4)
            Higher order = sharper cutoff but may cause ringing
        
        Raises
        ------
        ValueError
            If cutoff frequency exceeds Nyquist frequency (sample_rate/2)
        """
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order
        
        # Calculate Nyquist frequency (maximum representable frequency)
        # Shannon-Nyquist theorem: f_max = sample_rate / 2
        nyquist_freq = sample_rate / 2.0
        
        if cutoff_freq >= nyquist_freq:
            raise ValueError(
                f"Cutoff frequency ({cutoff_freq} Hz) must be less than "
                f"Nyquist frequency ({nyquist_freq} Hz)"
            )
        
        # Normalized cutoff frequency (0 to 1, where 1 = Nyquist)
        # This is required by scipy.signal.butter
        normalized_cutoff = cutoff_freq / nyquist_freq
        
        # Design Butterworth filter and get coefficients
        # b = numerator coefficients, a = denominator coefficients
        self.b, self.a = butter(order, normalized_cutoff, btype='low')
        
        print(f"[DSP] LowPassFilter initialized: "
              f"fc={cutoff_freq}Hz, fs={sample_rate}Hz, order={order}")
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply the low-pass filter to an input signal.
        
        Uses forward-backward filtering (filtfilt) for zero phase distortion.
        This is crucial for sensor fusion where time alignment matters.
        
        Parameters
        ----------
        signal : np.ndarray
            The input signal to filter (1D array)
        
        Returns
        -------
        filtered : np.ndarray
            The filtered signal (same length as input)
        
        Notes
        -----
        filtfilt applies the filter twice (forward and backward), which:
        1. Eliminates phase distortion
        2. Effectively doubles the filter order
        3. Squares the magnitude response (sharper cutoff)
        """
        # Validate input
        if len(signal) < 3 * max(len(self.b), len(self.a)):
            raise ValueError(
                "Signal too short for filtering. Need at least "
                f"{3 * max(len(self.b), len(self.a))} samples."
            )
        
        # Apply zero-phase filtering
        filtered = filtfilt(self.b, self.a, signal)
        
        return filtered
    
    def get_frequency_response(
        self,
        n_points: int = 512
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate the frequency response of the filter.
        
        Useful for plotting the filter's magnitude response to verify
        the cutoff behavior.
        
        Parameters
        ----------
        n_points : int
            Number of frequency points to calculate (default: 512)
        
        Returns
        -------
        frequencies : np.ndarray
            Frequency values in Hz (0 to Nyquist)
        magnitude_db : np.ndarray
            Magnitude response in decibels
        """
        from scipy.signal import freqz
        
        # Calculate frequency response
        w, h = freqz(self.b, self.a, worN=n_points)
        
        # Convert normalized frequency to Hz
        frequencies = w * self.sample_rate / (2 * np.pi)
        
        # Convert magnitude to decibels (with small epsilon to avoid log(0))
        magnitude_db = 20 * np.log10(np.abs(h) + 1e-10)
        
        return frequencies, magnitude_db


class MovingAverageFilter:
    """
    Simple Moving Average Filter for comparison/baseline.
    
    While not as sophisticated as Butterworth, moving average filters are
    computationally cheap and sometimes used in resource-constrained
    embedded systems.
    
    This is included for educational comparison - the Butterworth filter
    generally performs better for our application.
    """
    
    def __init__(self, window_size: int = 5):
        """
        Initialize the Moving Average Filter.
        
        Parameters
        ----------
        window_size : int
            Number of samples to average (default: 5)
        """
        self.window_size = window_size
        print(f"[DSP] MovingAverageFilter initialized: window={window_size}")
    
    def apply(self, signal: np.ndarray) -> np.ndarray:
        """
        Apply moving average filter using convolution.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        
        Returns
        -------
        filtered : np.ndarray
            Filtered signal (same length as input)
        """
        kernel = np.ones(self.window_size) / self.window_size
        # 'same' mode preserves signal length
        return np.convolve(signal, kernel, mode='same')


# =============================================================================
# MODULE TEST
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create a test signal: 1 Hz sine + 20 Hz noise
    fs = 100  # Sample rate
    t = np.linspace(0, 2, 200)
    clean = np.sin(2 * np.pi * 1 * t)  # 1 Hz signal
    noisy = clean + 0.5 * np.sin(2 * np.pi * 20 * t)  # Add 20 Hz noise
    
    # Apply filter
    lpf = LowPassFilter(cutoff_freq=5.0, sample_rate=fs)
    filtered = lpf.apply(noisy)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(t, noisy, 'b-', alpha=0.5, label='Noisy Signal')
    plt.plot(t, filtered, 'r-', linewidth=2, label='Filtered Signal')
    plt.plot(t, clean, 'g--', label='Original (Ground Truth)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.title('Low-Pass Filter Test: Removing 20Hz Noise from 1Hz Signal')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

