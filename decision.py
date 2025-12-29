"""
=============================================================================
DECISION ENGINE MODULE - Autonomous Spacecraft Response System
=============================================================================

This module implements the autonomous decision-making layer that translates
processed sensor data into actionable spacecraft responses. It simulates
a simplified version of the decision logic that would run on a spacecraft's
flight computer.

DECISION ARCHITECTURE:
----------------------
The system uses a finite state machine (FSM) with three states:

    ┌─────────────────────────────────────────────────────────────────┐
    │                    SPACECRAFT STATE MACHINE                      │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   ┌──────────┐        ┌──────────┐        ┌──────────┐          │
    │   │ NOMINAL  │ ──────>│ WARNING  │ ──────>│ CRITICAL │          │
    │   │ (Green)  │<────── │ (Yellow) │<────── │  (Red)   │          │
    │   └──────────┘        └──────────┘        └──────────┘          │
    │                                                                  │
    │   |signal| < 0.8      0.8 ≤ |signal| ≤ 1.5    |signal| > 1.5    │
    │                                                                  │
    │   Action: Log Data    Action: Fire Thrusters  Action: Emergency  │
    │                                               Shutdown           │
    └─────────────────────────────────────────────────────────────────┘

STATE DESCRIPTIONS:
-------------------
1. NOMINAL (< 0.8 amplitude):
   - Normal spacecraft operation
   - Vibrations within design tolerances
   - Action: Continue data logging for analysis

2. WARNING (0.8 - 1.5 amplitude):
   - Elevated vibration levels detected
   - Approaching structural limits
   - Action: Fire attitude control thrusters to dampen oscillation

3. CRITICAL (> 1.5 amplitude):
   - Dangerous vibration levels
   - Risk of structural damage
   - Action: Emergency shutdown of affected systems

PROCESSING LATENCY:
-------------------
Real spacecraft systems have processing latency due to:
- Sensor readout time
- Data bus transfer delays
- CPU processing overhead
- Actuator response time

We simulate this with a small configurable delay.

Author: Spacecraft Nervous System Simulation Team
Course: Quantum-IoT Hybrid Network for Spacecraft Decision Systems
=============================================================================
"""

import time
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


class SystemState(Enum):
    """
    Enumeration of possible spacecraft system states.
    
    Using an Enum provides type safety and prevents invalid state strings.
    """
    NOMINAL = "NOMINAL"    # Green - all systems normal
    WARNING = "WARNING"    # Yellow - elevated concern
    CRITICAL = "CRITICAL"  # Red - immediate action required


@dataclass
class Decision:
    """
    Container for a single decision made by the engine.
    
    Attributes
    ----------
    timestamp : float
        Time at which this decision was made (in simulation time)
    state : SystemState
        The determined system state
    action : str
        The recommended/taken action
    signal_value : float
        The signal amplitude that triggered this decision
    processing_time_ms : float
        Simulated processing latency in milliseconds
    """
    timestamp: float
    state: SystemState
    action: str
    signal_value: float
    processing_time_ms: float


class DecisionEngine:
    """
    Autonomous decision-making system for spacecraft structural health.
    
    This class evaluates sensor fusion output and determines appropriate
    spacecraft responses based on vibration amplitude thresholds.
    
    Attributes
    ----------
    nominal_threshold : float
        Maximum amplitude for NOMINAL state
    warning_threshold : float
        Maximum amplitude for WARNING state (above = CRITICAL)
    processing_latency_ms : float
        Simulated processing time per decision
    decision_history : List[Decision]
        Log of all decisions made (for analysis)
    
    Example
    -------
    >>> engine = DecisionEngine()
    >>> decision = engine.evaluate(signal_value=0.5, timestamp=1.0)
    >>> print(decision.state)
    SystemState.NOMINAL
    """
    
    # Define actions for each state (class constants)
    ACTIONS = {
        SystemState.NOMINAL: "Continue Data Logging - All Systems Normal",
        SystemState.WARNING: "Fire Attitude Thrusters - Dampen Oscillation",
        SystemState.CRITICAL: "EMERGENCY: Initiate Safe Mode - Shutdown Non-Essential Systems"
    }
    
    def __init__(
        self,
        nominal_threshold: float = 0.8,
        warning_threshold: float = 1.5,
        processing_latency_ms: float = 5.0,
        simulate_latency: bool = False
    ):
        """
        Initialize the Decision Engine.
        
        Parameters
        ----------
        nominal_threshold : float
            Signal amplitude below which system is NOMINAL (default: 0.8)
        warning_threshold : float
            Signal amplitude above which system is CRITICAL (default: 1.5)
            Between nominal and warning thresholds = WARNING state
        processing_latency_ms : float
            Simulated processing time in milliseconds (default: 5ms)
        simulate_latency : bool
            If True, actually sleep to simulate latency (default: False)
            Set to False for faster simulation runs
        """
        self.nominal_threshold = nominal_threshold
        self.warning_threshold = warning_threshold
        self.processing_latency_ms = processing_latency_ms
        self.simulate_latency = simulate_latency
        
        # Initialize decision log
        self.decision_history: List[Decision] = []
        
        # State transition counters (for analysis)
        self.state_counts = {
            SystemState.NOMINAL: 0,
            SystemState.WARNING: 0,
            SystemState.CRITICAL: 0
        }
        
        print(f"[DECISION] Engine initialized:")
        print(f"[DECISION]   NOMINAL:  |signal| < {nominal_threshold}")
        print(f"[DECISION]   WARNING:  {nominal_threshold} <= |signal| <= {warning_threshold}")
        print(f"[DECISION]   CRITICAL: |signal| > {warning_threshold}")
    
    def evaluate(self, signal_value: float, timestamp: float = 0.0) -> Decision:
        """
        Evaluate a signal value and determine the appropriate response.
        
        This method:
        1. Classifies the signal amplitude into a state
        2. Determines the appropriate action
        3. Logs the decision for later analysis
        4. Optionally simulates processing latency
        
        Parameters
        ----------
        signal_value : float
            The current signal amplitude from sensor fusion
        timestamp : float
            The simulation time of this measurement (default: 0.0)
        
        Returns
        -------
        Decision
            Dataclass containing state, action, and metadata
        """
        # Simulate processing latency (if enabled)
        if self.simulate_latency:
            time.sleep(self.processing_latency_ms / 1000.0)
        
        # Get absolute value (we care about amplitude, not direction)
        amplitude = abs(signal_value)
        
        # =====================================================================
        # STATE CLASSIFICATION LOGIC
        # =====================================================================
        # This uses a simple threshold-based classifier. In a real system,
        # you might use more sophisticated methods like:
        # - Hysteresis to prevent state oscillation at boundaries
        # - Time-averaging to ignore transient spikes
        # - Machine learning for anomaly detection
        
        if amplitude < self.nominal_threshold:
            # -------------------------------------------------------------
            # NOMINAL: Vibrations within normal operating envelope
            # The spacecraft structure is behaving as expected.
            # No corrective action needed - just keep monitoring.
            # -------------------------------------------------------------
            state = SystemState.NOMINAL
            
        elif amplitude <= self.warning_threshold:
            # -------------------------------------------------------------
            # WARNING: Elevated vibration levels detected
            # Not immediately dangerous, but trending toward problems.
            # Proactive response: Use thrusters to reduce oscillation.
            # This is like a car's stability control activating.
            # -------------------------------------------------------------
            state = SystemState.WARNING
            
        else:
            # -------------------------------------------------------------
            # CRITICAL: Dangerous vibration amplitude
            # Risk of structural fatigue or resonance damage.
            # Immediate response: Shut down non-essential systems to
            # reduce loads and prevent cascade failures.
            # -------------------------------------------------------------
            state = SystemState.CRITICAL
        
        # Get the action for this state
        action = self.ACTIONS[state]
        
        # Create decision record
        decision = Decision(
            timestamp=timestamp,
            state=state,
            action=action,
            signal_value=signal_value,
            processing_time_ms=self.processing_latency_ms
        )
        
        # Update statistics
        self.decision_history.append(decision)
        self.state_counts[state] += 1
        
        return decision
    
    def evaluate_signal_array(
        self,
        signal: np.ndarray,
        time_axis: np.ndarray
    ) -> List[Decision]:
        """
        Evaluate an entire signal array and return all decisions.
        
        This is a convenience method for batch processing.
        
        Parameters
        ----------
        signal : np.ndarray
            Array of signal values to evaluate
        time_axis : np.ndarray
            Corresponding time values for each sample
        
        Returns
        -------
        decisions : List[Decision]
            List of Decision objects for each sample
        """
        decisions = []
        for i in range(len(signal)):
            decision = self.evaluate(signal[i], time_axis[i])
            decisions.append(decision)
        
        return decisions
    
    def get_state_timeline(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the state timeline from decision history.
        
        Converts the decision history into numerical arrays suitable
        for plotting.
        
        Returns
        -------
        timestamps : np.ndarray
            Array of timestamps
        state_values : np.ndarray
            Numerical state values (0=NOMINAL, 1=WARNING, 2=CRITICAL)
        """
        # Map states to numbers for plotting
        state_map = {
            SystemState.NOMINAL: 0,
            SystemState.WARNING: 1,
            SystemState.CRITICAL: 2
        }
        
        timestamps = np.array([d.timestamp for d in self.decision_history])
        state_values = np.array([state_map[d.state] for d in self.decision_history])
        
        return timestamps, state_values
    
    def print_summary(self):
        """Print a summary of the decision engine's performance."""
        total = sum(self.state_counts.values())
        
        print("\n" + "="*60)
        print("DECISION ENGINE SUMMARY")
        print("="*60)
        
        for state, count in self.state_counts.items():
            percentage = 100 * count / total if total > 0 else 0
            bar = "#" * int(percentage / 2)  # Visual bar
            print(f"  {state.value:10s}: {count:5d} ({percentage:5.1f}%) {bar}")
        
        print("="*60)
        
        # Find critical events
        critical_decisions = [d for d in self.decision_history 
                             if d.state == SystemState.CRITICAL]
        if critical_decisions:
            print(f"\n[!] CRITICAL EVENTS DETECTED: {len(critical_decisions)}")
            for d in critical_decisions[:5]:  # Show first 5
                print(f"    t={d.timestamp:.3f}s: amplitude={d.signal_value:.3f}")
            if len(critical_decisions) > 5:
                print(f"    ... and {len(critical_decisions)-5} more")


# =============================================================================
# MODULE TEST
# =============================================================================
if __name__ == "__main__":
    # Test the decision engine with synthetic data
    engine = DecisionEngine(simulate_latency=False)
    
    # Create test signal that transitions through all states
    test_signal = np.array([
        0.3, 0.5, 0.7,      # NOMINAL
        0.9, 1.0, 1.2,      # WARNING
        1.8, 2.0, 1.6,      # CRITICAL
        1.0, 0.5, 0.2       # Back down
    ])
    test_times = np.linspace(0, 12, 12)
    
    print("\nProcessing test signal...")
    for i in range(len(test_signal)):
        decision = engine.evaluate(test_signal[i], test_times[i])
        print(f"  t={test_times[i]:.1f}s: signal={test_signal[i]:.1f} -> "
              f"{decision.state.value}")
    
    # Print summary
    engine.print_summary()

