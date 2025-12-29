"""
=============================================================================
DATA GENERATION MODULE - Simulated Spacecraft Telemetry
=============================================================================

This module simulates sensor data from a spacecraft's structural health
monitoring system. It models two sensor types with fundamentally different
characteristics:

1. CLASSICAL SENSOR (Piezoelectric/Strain Gauge):
   - High noise floor due to thermal fluctuations and electronic noise
   - Exhibits drift over time (temperature-dependent offset)
   - Robust and reliable - never fails catastrophically
   
2. QUANTUM SENSOR (Optomechanical/SQUID-based):
   - Ultra-low noise floor (Heisenberg-limited sensitivity)
   - Vulnerable to radiation-induced decoherence events
   - Radiation hits cause massive transient spikes (not real signal)

The ground truth is a simple sinusoidal vibration pattern representing
structural oscillation in the spacecraft (e.g., from solar panel flexing).

Author: Spacecraft Nervous System Simulation Team
Course: Quantum-IoT Hybrid Network for Spacecraft Decision Systems
=============================================================================
"""

import numpy as np
from typing import Tuple


def generate_telemetry(
    duration: float = 10.0,
    rate: int = 100,
    seed: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate simulated telemetry data from classical and quantum sensors.
    
    Parameters
    ----------
    duration : float
        Total simulation duration in seconds (default: 10s)
    rate : int
        Sampling rate in Hz (default: 100 Hz = 100 samples/second)
    seed : int, optional
        Random seed for reproducibility (important for reports/debugging)
    
    Returns
    -------
    time_axis : np.ndarray
        Time values for each sample point
    ground_truth : np.ndarray
        The actual physical signal (what we're trying to measure)
    classical_output : np.ndarray
        Classical sensor reading (noisy but reliable)
    quantum_output : np.ndarray
        Quantum sensor reading (precise but with radiation spikes)
    
    Physical Model
    --------------
    Ground Truth: A(t) = sin(2πft) where f = 1 Hz
    This represents a 1 Hz structural vibration with unit amplitude.
    """
    
    # Set random seed for reproducibility (critical for scientific reports)
    if seed is not None:
        np.random.seed(seed)
    
    # Calculate total number of samples
    n_samples = int(duration * rate)
    
    # Generate time axis
    time_axis = np.linspace(0, duration, n_samples, endpoint=False)
    
    # =========================================================================
    # GROUND TRUTH SIGNAL
    # =========================================================================
    # A pure 1 Hz sinusoid representing actual structural vibration
    # In reality, this would be the "true" mechanical state of the spacecraft
    vibration_frequency = 1.0  # Hz
    ground_truth = np.sin(2 * np.pi * vibration_frequency * time_axis)
    
    # =========================================================================
    # CLASSICAL SENSOR OUTPUT
    # =========================================================================
    # Model: y_classical(t) = ground_truth(t) + noise(t) + drift(t)
    
    # Gaussian noise: σ = 0.5 (significant noise floor)
    # This represents thermal noise, electronic noise, and quantization error
    classical_noise_sigma = 0.5
    classical_noise = np.random.normal(0, classical_noise_sigma, n_samples)
    
    # Thermal drift: A slow, linear drift simulating temperature effects
    # Drift rate: 0.02 units per second (accumulates over time)
    # In real sensors, this would be compensated by periodic recalibration
    drift_rate = 0.02
    thermal_drift = drift_rate * time_axis
    
    # Combine to get classical sensor output
    classical_output = ground_truth + classical_noise + thermal_drift
    
    # =========================================================================
    # QUANTUM SENSOR OUTPUT
    # =========================================================================
    # Model: y_quantum(t) = ground_truth(t) + low_noise(t) + radiation_spikes(t)
    
    # Ultra-low Gaussian noise: σ = 0.05 (quantum-limited precision)
    # This is 10x better than classical - the main advantage of quantum sensors
    quantum_noise_sigma = 0.05
    quantum_noise = np.random.normal(0, quantum_noise_sigma, n_samples)
    
    # Start with clean quantum signal
    quantum_output = ground_truth + quantum_noise
    
    # -------------------------------------------------------------------------
    # RADIATION-INDUCED DECOHERENCE SPIKES
    # -------------------------------------------------------------------------
    # In space, high-energy particles (cosmic rays, solar wind) can hit the
    # quantum sensor and cause sudden decoherence. This manifests as massive
    # transient spikes in the output - NOT representative of the true signal.
    #
    # Spike probability: ~1% of samples affected (adjustable)
    # Spike magnitude: ±5.0 (far outside normal signal range of ±1.0)
    
    spike_probability = 0.01  # 1% chance per sample
    spike_magnitude = 5.0
    
    # Generate random spike locations
    spike_mask = np.random.random(n_samples) < spike_probability
    n_spikes = np.sum(spike_mask)
    
    # Generate random spike values (positive or negative)
    spike_values = np.random.choice([-spike_magnitude, spike_magnitude], n_spikes)
    
    # Inject spikes into quantum output
    quantum_output[spike_mask] = spike_values
    
    # Log spike information (useful for debugging and validation)
    print(f"[DATA_GEN] Generated {n_samples} samples over {duration}s at {rate}Hz")
    print(f"[DATA_GEN] Injected {n_spikes} radiation spikes into quantum channel")
    
    return time_axis, ground_truth, classical_output, quantum_output


def get_spike_indices(quantum_output: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """
    Utility function to identify indices where radiation spikes occurred.
    
    This is useful for visualization (highlighting spikes on plots) and
    for validating the fusion algorithm's spike detection capability.
    
    Parameters
    ----------
    quantum_output : np.ndarray
        The quantum sensor output array
    threshold : float
        Absolute value threshold for spike detection (default: 3.0)
        Normal signal range is approximately ±1.1, so 3.0 is conservative
    
    Returns
    -------
    spike_indices : np.ndarray
        Array of indices where spikes were detected
    """
    return np.where(np.abs(quantum_output) > threshold)[0]


# =============================================================================
# MODULE TEST
# =============================================================================
if __name__ == "__main__":
    # Quick test to verify data generation
    t, truth, classical, quantum = generate_telemetry(duration=5, rate=100, seed=42)
    
    print(f"\nData shapes: time={t.shape}, truth={truth.shape}")
    print(f"Classical: mean={classical.mean():.3f}, std={classical.std():.3f}")
    print(f"Quantum:   mean={quantum.mean():.3f}, std={quantum.std():.3f}")
    print(f"Spike indices: {get_spike_indices(quantum)}")

