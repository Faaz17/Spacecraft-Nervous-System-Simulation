"""
=============================================================================
SENSOR FUSION MODULE - Intelligent Data Integration
=============================================================================

This module implements a sensor fusion algorithm that combines data from
classical and quantum sensors to produce an optimal estimate of the true
spacecraft structural state.

FUSION STRATEGY:
----------------
The core challenge is that each sensor type has complementary strengths:

| Aspect          | Classical Sensor | Quantum Sensor |
|-----------------|------------------|----------------|
| Noise Floor     | High (σ=0.5)     | Low (σ=0.05)   |
| Reliability     | Very High        | Intermittent   |
| Failure Mode    | Gradual drift    | Sudden spikes  |
| Trust Level     | Always usable    | Conditional    |

Our fusion algorithm exploits this complementarity:
1. DETECT anomalies by comparing both sensors (they should roughly agree)
2. REJECT quantum data when radiation spikes are detected
3. TRUST quantum data (with high weight) during normal operation

This is a form of "robust sensor fusion" commonly used in safety-critical
systems like spacecraft, autonomous vehicles, and medical devices.

MATHEMATICAL MODEL:
-------------------
Let C(t) = classical measurement, Q(t) = quantum measurement

If |C(t) - Q(t)| > threshold:
    # Anomaly detected - quantum sensor unreliable
    fused(t) = C(t)
Else:
    # Normal operation - weighted average favoring precision
    fused(t) = 0.8 * Q(t) + 0.2 * C(t)

Author: Spacecraft Nervous System Simulation Team
Course: Quantum-IoT Hybrid Network for Spacecraft Decision Systems
=============================================================================
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass


@dataclass
class FusionResult:
    """
    Container for fusion algorithm outputs.
    
    Attributes
    ----------
    fused_signal : np.ndarray
        The optimally fused sensor data
    anomaly_indices : List[int]
        Indices where quantum sensor was rejected due to anomalies
    trust_quantum : np.ndarray
        Boolean array: True where quantum data was trusted
    """
    fused_signal: np.ndarray
    anomaly_indices: List[int]
    trust_quantum: np.ndarray


def fuse_signals(
    classical: np.ndarray,
    quantum: np.ndarray,
    threshold: float = 2.0,
    quantum_weight: float = 0.8
) -> FusionResult:
    """
    Fuse classical and quantum sensor data with anomaly rejection.
    
    This function implements robust sensor fusion that:
    1. Detects quantum sensor anomalies (radiation-induced spikes)
    2. Rejects anomalous quantum readings
    3. Creates an optimal weighted average during normal operation
    
    Parameters
    ----------
    classical : np.ndarray
        Classical sensor readings (1D array)
    quantum : np.ndarray
        Quantum sensor readings (1D array, same length as classical)
    threshold : float
        Maximum allowed difference between sensors (default: 2.0)
        If |classical - quantum| > threshold, quantum is considered anomalous
    quantum_weight : float
        Weight given to quantum sensor in normal operation (default: 0.8)
        Classical weight is automatically (1 - quantum_weight)
    
    Returns
    -------
    FusionResult
        Dataclass containing:
        - fused_signal: The combined sensor data
        - anomaly_indices: Where anomalies were detected
        - trust_quantum: Boolean mask of quantum trust
    
    Algorithm Walkthrough
    ---------------------
    For each time step t:
    
    1. Calculate disagreement: Δ(t) = |C(t) - Q(t)|
    
    2. If Δ(t) > threshold:
       - Mark as anomaly (radiation hit on quantum sensor)
       - Use classical only: fused(t) = C(t)
       
    3. If Δ(t) ≤ threshold:
       - Both sensors are operating normally
       - Weighted average: fused(t) = 0.8*Q(t) + 0.2*C(t)
       
    Why 80/20 weighting?
    - Quantum has 10x better precision (σ=0.05 vs σ=0.5)
    - But we keep 20% classical to maintain some robustness
    - This ratio can be tuned based on mission requirements
    """
    
    # Validate inputs
    if len(classical) != len(quantum):
        raise ValueError(
            f"Signal length mismatch: classical={len(classical)}, "
            f"quantum={len(quantum)}"
        )
    
    if not 0 < quantum_weight < 1:
        raise ValueError(f"quantum_weight must be in (0,1), got {quantum_weight}")
    
    n_samples = len(classical)
    classical_weight = 1.0 - quantum_weight
    
    # Initialize output arrays
    fused_signal = np.zeros(n_samples)
    trust_quantum = np.ones(n_samples, dtype=bool)  # Start assuming all trusted
    anomaly_indices = []
    
    # =========================================================================
    # POINT-BY-POINT FUSION
    # =========================================================================
    # We iterate explicitly (rather than vectorized) for educational clarity
    # and because anomaly detection logic benefits from explicit control flow
    
    for i in range(n_samples):
        # Calculate sensor disagreement at this time point
        disagreement = abs(classical[i] - quantum[i])
        
        if disagreement > threshold:
            # -------------------------------------------------------------
            # ANOMALY DETECTED: Quantum sensor has likely taken a radiation
            # hit, causing decoherence and a spurious spike output.
            # 
            # Action: Reject quantum reading entirely, use classical only
            # Rationale: Classical sensors don't fail this catastrophically;
            #            they just get noisy. A noisy reading is better than
            #            a completely wrong reading.
            # -------------------------------------------------------------
            fused_signal[i] = classical[i]
            trust_quantum[i] = False
            anomaly_indices.append(i)
            
        else:
            # -------------------------------------------------------------
            # NORMAL OPERATION: Both sensors agree within tolerance
            # 
            # Action: Weighted average favoring quantum precision
            # Rationale: Quantum sensor has 10x better precision, so we
            #            weight it heavily. The 20% classical contribution
            #            helps smooth out any small quantum fluctuations.
            # -------------------------------------------------------------
            fused_signal[i] = (quantum_weight * quantum[i] + 
                               classical_weight * classical[i])
            trust_quantum[i] = True
    
    # Log fusion statistics
    n_anomalies = len(anomaly_indices)
    anomaly_rate = 100 * n_anomalies / n_samples
    print(f"[FUSION] Processed {n_samples} samples")
    print(f"[FUSION] Detected {n_anomalies} anomalies ({anomaly_rate:.1f}%)")
    print(f"[FUSION] Quantum trusted for {100-anomaly_rate:.1f}% of data")
    
    return FusionResult(
        fused_signal=fused_signal,
        anomaly_indices=anomaly_indices,
        trust_quantum=trust_quantum
    )


def calculate_fusion_quality(
    fused: np.ndarray,
    ground_truth: np.ndarray
) -> dict:
    """
    Calculate quality metrics for the fused signal.
    
    This function helps evaluate how well the fusion algorithm performed
    by comparing against the known ground truth.
    
    Parameters
    ----------
    fused : np.ndarray
        The fused sensor output
    ground_truth : np.ndarray
        The actual signal (for simulation evaluation)
    
    Returns
    -------
    metrics : dict
        Dictionary containing:
        - rmse: Root Mean Square Error
        - mae: Mean Absolute Error
        - max_error: Maximum absolute error
        - correlation: Pearson correlation coefficient
    """
    errors = fused - ground_truth
    
    metrics = {
        'rmse': np.sqrt(np.mean(errors**2)),
        'mae': np.mean(np.abs(errors)),
        'max_error': np.max(np.abs(errors)),
        'correlation': np.corrcoef(fused, ground_truth)[0, 1]
    }
    
    return metrics


# =============================================================================
# MODULE TEST
# =============================================================================
if __name__ == "__main__":
    # Test with synthetic data
    n = 100
    ground_truth = np.sin(np.linspace(0, 2*np.pi, n))
    
    # Simulate sensors
    np.random.seed(42)
    classical = ground_truth + np.random.normal(0, 0.5, n)
    quantum = ground_truth + np.random.normal(0, 0.05, n)
    
    # Add radiation spikes to quantum
    spike_indices = [20, 50, 80]
    for idx in spike_indices:
        quantum[idx] = 5.0 if np.random.random() > 0.5 else -5.0
    
    # Run fusion
    result = fuse_signals(classical, quantum)
    
    # Check if spikes were detected
    print(f"\nExpected anomaly indices: {spike_indices}")
    print(f"Detected anomaly indices: {result.anomaly_indices}")
    
    # Calculate quality
    metrics = calculate_fusion_quality(result.fused_signal, ground_truth)
    print(f"\nFusion Quality Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

