"""
=============================================================================
MAIN ORCHESTRATOR - Using Real Hardware Simulation Data
=============================================================================

This script runs the Spacecraft Nervous System simulation using REAL sensor
data from the hardware simulation team (Vanshika & Varenya) instead of
mock/generated data.

USAGE:
------
    python main_real_data.py                    # Uses Varenya's complete data
    python main_real_data.py --source vanshika  # Uses Vanshika's data
    python main_real_data.py --stage env        # Uses environmental disturbance stage
    python main_real_data.py --stage adc        # Uses ADC stage

Author: Spacecraft Nervous System Simulation Team
Course: Quantum-IoT Hybrid Network for Spacecraft Decision Systems
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
import argparse

# Import our modules
from data_loader import load_vanshika_data, load_varenya_data, detect_radiation_spikes, SensorData
from dsp import LowPassFilter
from fusion import fuse_signals, calculate_fusion_quality
from decision import DecisionEngine, SystemState


def run_real_data_simulation(
    data: SensorData,
    filter_cutoff: float = 50.0  # Higher cutoff for real data (1000 Hz sample rate)
) -> dict:
    """
    Run the simulation pipeline on real sensor data.
    
    Parameters
    ----------
    data : SensorData
        Loaded sensor data from hardware simulation
    filter_cutoff : float
        Low-pass filter cutoff frequency
    
    Returns
    -------
    results : dict
        All simulation results
    """
    print("="*70)
    print("    SPACECRAFT NERVOUS SYSTEM SIMULATION")
    print("    Using Real Hardware Simulation Data")
    print("="*70)
    print(f"    Data Source: {data.source}")
    print("="*70)
    print()
    
    # =========================================================================
    # STEP 1: ANALYZE RAW DATA
    # =========================================================================
    print("[STEP 1] Analyzing real sensor data...")
    print("-"*50)
    print(f"[DATA] Samples: {len(data.time)}")
    print(f"[DATA] Duration: {data.duration:.3f} seconds")
    print(f"[DATA] Sample Rate: {data.sample_rate:.1f} Hz")
    print(f"[DATA] Classical - Mean: {data.classical.mean():.3f}, Std: {data.classical.std():.3f}")
    print(f"[DATA] Quantum   - Mean: {data.quantum.mean():.3f}, Std: {data.quantum.std():.3f}")
    
    # Detect anomalies in raw data
    spike_indices = detect_radiation_spikes(data.quantum, data.ground_truth)
    print()
    
    # =========================================================================
    # STEP 2: DIGITAL SIGNAL PROCESSING
    # =========================================================================
    print("[STEP 2] Applying low-pass filtering...")
    print("-"*50)
    
    # Adjust filter cutoff based on signal characteristics
    # The ground truth appears to be around 11 Hz (based on the MATLAB files)
    lpf = LowPassFilter(
        cutoff_freq=filter_cutoff,
        sample_rate=data.sample_rate,
        order=4
    )
    
    classical_filtered = lpf.apply(data.classical)
    quantum_filtered = lpf.apply(data.quantum)
    
    print(f"[DSP] Filtered {len(data.classical)} samples")
    print()
    
    # =========================================================================
    # STEP 3: SENSOR FUSION
    # =========================================================================
    print("[STEP 3] Fusing sensor data with anomaly rejection...")
    print("-"*50)
    
    # Use a threshold based on the data characteristics
    # Classical has higher noise, so use that as reference
    fusion_threshold = 2.0 * data.classical.std()
    
    fusion_result = fuse_signals(
        classical_filtered,
        quantum_filtered,
        threshold=fusion_threshold,
        quantum_weight=0.8
    )
    
    # Calculate quality metrics
    quality_metrics = calculate_fusion_quality(
        fusion_result.fused_signal,
        data.ground_truth
    )
    
    print(f"\n[FUSION] Quality Metrics:")
    print(f"[FUSION]   RMSE:        {quality_metrics['rmse']:.4f}")
    print(f"[FUSION]   MAE:         {quality_metrics['mae']:.4f}")
    print(f"[FUSION]   Correlation: {quality_metrics['correlation']:.4f}")
    print()
    
    # =========================================================================
    # STEP 4: DECISION MAKING
    # =========================================================================
    print("[STEP 4] Running autonomous decision engine...")
    print("-"*50)
    
    # Adjust thresholds based on signal amplitude
    signal_amplitude = np.percentile(np.abs(data.ground_truth), 95)
    nominal_threshold = 0.8 * signal_amplitude
    warning_threshold = 1.2 * signal_amplitude
    
    decision_engine = DecisionEngine(
        nominal_threshold=nominal_threshold,
        warning_threshold=warning_threshold,
        simulate_latency=False
    )
    
    decisions = decision_engine.evaluate_signal_array(
        fusion_result.fused_signal,
        data.time
    )
    
    decision_engine.print_summary()
    
    # =========================================================================
    # COMPILE RESULTS
    # =========================================================================
    results = {
        'time': data.time,
        'ground_truth': data.ground_truth,
        'classical_raw': data.classical,
        'quantum_raw': data.quantum,
        'spike_indices': spike_indices,
        'classical_filtered': classical_filtered,
        'quantum_filtered': quantum_filtered,
        'fused_signal': fusion_result.fused_signal,
        'anomaly_indices': fusion_result.anomaly_indices,
        'trust_quantum': fusion_result.trust_quantum,
        'quality_metrics': quality_metrics,
        'decisions': decisions,
        'decision_engine': decision_engine,
        'data_source': data.source,
        'params': {
            'sample_rate': data.sample_rate,
            'duration': data.duration,
            'filter_cutoff': filter_cutoff,
            'fusion_threshold': fusion_threshold
        }
    }
    
    return results


def create_real_data_dashboard(results: dict, save_path: str = None):
    """
    Create visualization dashboard for real sensor data results.
    """
    # Extract data
    t = results['time']
    ground_truth = results['ground_truth']
    classical_raw = results['classical_raw']
    quantum_raw = results['quantum_raw']
    spike_indices = results['spike_indices']
    fused_signal = results['fused_signal']
    anomaly_indices = results['anomaly_indices']
    decisions = results['decisions']
    
    # Create figure
    fig = plt.figure(figsize=(14, 12))
    fig.patch.set_facecolor('#1a1a2e')
    
    gs = gridspec.GridSpec(4, 1, height_ratios=[1, 1, 1, 0.5], hspace=0.35)
    
    # Colors
    colors = {
        'classical': '#ff6b6b',
        'quantum': '#4ecdc4',
        'ground_truth': '#ffe66d',
        'fused': '#95e1d3',
        'spike': '#ff4757',
        'nominal': '#2ed573',
        'warning': '#ffa502',
        'critical': '#ff4757',
        'grid': '#2d2d44',
        'text': '#eee'
    }
    
    # =========================================================================
    # SUBPLOT 1: Raw Classical Sensor
    # =========================================================================
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor('#16213e')
    
    ax1.plot(t, classical_raw, color=colors['classical'], alpha=0.7, 
             linewidth=0.5, label='Classical Sensor')
    ax1.plot(t, ground_truth, color=colors['ground_truth'], 
             linewidth=1.5, linestyle='--', alpha=0.9, label='Ground Truth')
    
    ax1.set_ylabel('Amplitude', fontsize=10, color=colors['text'])
    ax1.set_title('Classical Sensor (High Noise, Thermal Drift)', 
                  fontsize=12, fontweight='bold', color=colors['text'], pad=10)
    ax1.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333',
               labelcolor=colors['text'], fontsize=9)
    ax1.set_xlim(t[0], t[-1])
    ax1.grid(True, alpha=0.3, color=colors['grid'])
    ax1.tick_params(colors=colors['text'])
    
    # =========================================================================
    # SUBPLOT 2: Raw Quantum Sensor with Spikes
    # =========================================================================
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor('#16213e')
    
    ax2.plot(t, quantum_raw, color=colors['quantum'], alpha=0.7,
             linewidth=0.5, label='Quantum Sensor')
    ax2.plot(t, ground_truth, color=colors['ground_truth'], 
             linewidth=1.5, linestyle='--', alpha=0.9, label='Ground Truth')
    
    if len(spike_indices) > 0:
        ax2.scatter(t[spike_indices], quantum_raw[spike_indices],
                   color=colors['spike'], s=30, marker='X', 
                   zorder=5, label=f'Anomalies ({len(spike_indices)})',
                   edgecolors='white', linewidths=0.3)
    
    ax2.set_ylabel('Amplitude', fontsize=10, color=colors['text'])
    ax2.set_title('Quantum Sensor (High Precision, Radiation Vulnerable)', 
                  fontsize=12, fontweight='bold', color=colors['text'], pad=10)
    ax2.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333',
               labelcolor=colors['text'], fontsize=9)
    ax2.set_xlim(t[0], t[-1])
    ax2.grid(True, alpha=0.3, color=colors['grid'])
    ax2.tick_params(colors=colors['text'])
    
    # =========================================================================
    # SUBPLOT 3: Fused Signal Result
    # =========================================================================
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor('#16213e')
    
    ax3.plot(t, ground_truth, color=colors['ground_truth'], 
             linewidth=2, linestyle='--', alpha=0.9, label='Ground Truth')
    ax3.plot(t, fused_signal, color=colors['fused'], 
             linewidth=1, label='Fused Signal')
    
    if len(anomaly_indices) > 0:
        ax3.scatter(t[anomaly_indices], fused_signal[anomaly_indices],
                   color=colors['warning'], s=20, marker='o', alpha=0.7,
                   zorder=5, label=f'Anomaly Rejected ({len(anomaly_indices)})')
    
    ax3.set_ylabel('Amplitude', fontsize=10, color=colors['text'])
    ax3.set_title('Sensor Fusion Result (Quantum Precision + Classical Reliability)', 
                  fontsize=12, fontweight='bold', color=colors['text'], pad=10)
    ax3.legend(loc='upper right', facecolor='#1a1a2e', edgecolor='#333',
               labelcolor=colors['text'], fontsize=9)
    ax3.set_xlim(t[0], t[-1])
    ax3.grid(True, alpha=0.3, color=colors['grid'])
    ax3.tick_params(colors=colors['text'])
    
    # Add quality metrics
    metrics = results['quality_metrics']
    metrics_text = f"RMSE: {metrics['rmse']:.4f}\nCorr: {metrics['correlation']:.4f}"
    ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='#1a1a2e', 
                      edgecolor=colors['fused'], alpha=0.9),
             color=colors['text'], fontfamily='monospace')
    
    # =========================================================================
    # SUBPLOT 4: System State Timeline
    # =========================================================================
    ax4 = fig.add_subplot(gs[3])
    ax4.set_facecolor('#16213e')
    
    state_map = {
        SystemState.NOMINAL: 0,
        SystemState.WARNING: 1,
        SystemState.CRITICAL: 2
    }
    
    state_values = np.array([state_map[d.state] for d in decisions])
    state_colors = [colors['nominal'] if s == 0 else 
                   colors['warning'] if s == 1 else 
                   colors['critical'] for s in state_values]
    
    for i in range(len(t) - 1):
        ax4.fill_between([t[i], t[i+1]], [0, 0], [state_values[i]+1, state_values[i]+1],
                        color=state_colors[i], alpha=0.7, linewidth=0)
    
    ax4.axhline(y=1, color='white', linestyle=':', alpha=0.5, linewidth=0.8)
    ax4.axhline(y=2, color='white', linestyle=':', alpha=0.5, linewidth=0.8)
    
    ax4.set_yticks([0.5, 1.5, 2.5])
    ax4.set_yticklabels(['NOMINAL', 'WARNING', 'CRITICAL'], fontsize=9)
    ax4.set_ylabel('State', fontsize=10, color=colors['text'])
    ax4.set_xlabel('Time (seconds)', fontsize=10, color=colors['text'])
    ax4.set_title('Spacecraft System State Timeline', 
                  fontsize=12, fontweight='bold', color=colors['text'], pad=10)
    ax4.set_xlim(t[0], t[-1])
    ax4.set_ylim(0, 3)
    ax4.grid(True, axis='x', alpha=0.3, color=colors['grid'])
    ax4.tick_params(colors=colors['text'])
    
    legend_elements = [
        Patch(facecolor=colors['nominal'], label='NOMINAL'),
        Patch(facecolor=colors['warning'], label='WARNING'),
        Patch(facecolor=colors['critical'], label='CRITICAL')
    ]
    ax4.legend(handles=legend_elements, loc='upper right', 
               facecolor='#1a1a2e', edgecolor='#333',
               labelcolor=colors['text'], fontsize=8)
    
    # Main title
    fig.suptitle(f'QUANTUM-IoT SPACECRAFT DECISION SYSTEM\nData Source: {results["data_source"]}',
                 fontsize=14, fontweight='bold', color=colors['text'], y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    if save_path:
        plt.savefig(save_path, dpi=150, facecolor='#1a1a2e', edgecolor='none',
                   bbox_inches='tight')
        print(f"\n[VISUALIZATION] Dashboard saved to: {save_path}")
    
    plt.show()
    return fig


def main():
    """Main entry point with command-line argument support."""
    
    parser = argparse.ArgumentParser(
        description='Spacecraft Nervous System Simulation - Real Data Mode'
    )
    parser.add_argument(
        '--source', 
        choices=['vanshika', 'varenya'],
        default='varenya',
        help='Data source: vanshika or varenya (default: varenya)'
    )
    parser.add_argument(
        '--stage',
        choices=['raw', 'env', 'adc'],
        default='raw',
        help='Processing stage for Varenya data: raw, env, or adc (default: raw)'
    )
    parser.add_argument(
        '--cutoff',
        type=float,
        default=50.0,
        help='Low-pass filter cutoff frequency in Hz (default: 50.0)'
    )
    
    args = parser.parse_args()
    
    # Load data based on source selection
    try:
        if args.source == 'vanshika':
            data = load_vanshika_data()
            output_file = 'dashboard_vanshika.png'
        else:
            data = load_varenya_data(use_stage=args.stage)
            output_file = f'dashboard_varenya_{args.stage}.png'
    except FileNotFoundError as e:
        print(f"[ERROR] Could not load data file: {e}")
        print("[ERROR] Make sure the data folders are in the correct location.")
        return
    
    # Run simulation
    results = run_real_data_simulation(data, filter_cutoff=args.cutoff)
    
    print("\n" + "="*70)
    print("    SIMULATION COMPLETE - GENERATING DASHBOARD")
    print("="*70)
    
    # Create dashboard
    create_real_data_dashboard(results, save_path=output_file)
    
    # Print summary
    params = results['params']
    metrics = results['quality_metrics']
    engine = results['decision_engine']
    
    print("\n" + "="*70)
    print("    FINAL SUMMARY")
    print("="*70)
    print(f"""
    Data Source: {results['data_source']}
    
    Simulation Parameters:
    ----------------------
    - Duration:      {params['duration']:.3f} seconds
    - Sample Rate:   {params['sample_rate']:.1f} Hz
    - Total Samples: {len(results['time'])}
    - Filter Cutoff: {params['filter_cutoff']:.1f} Hz
    
    Sensor Fusion Results:
    ----------------------
    - Anomalies Detected:  {len(results['spike_indices'])}
    - Anomalies Rejected:  {len(results['anomaly_indices'])}
    - RMSE:                {metrics['rmse']:.4f}
    - MAE:                 {metrics['mae']:.4f}
    - Correlation:         {metrics['correlation']:.4f}
    
    Decision Engine Summary:
    ------------------------
    - NOMINAL:  {engine.state_counts[SystemState.NOMINAL]} samples
    - WARNING:  {engine.state_counts[SystemState.WARNING]} samples
    - CRITICAL: {engine.state_counts[SystemState.CRITICAL]} samples
    """)
    
    print("="*70)
    print(f"    Output saved: {output_file}")
    print("="*70)


if __name__ == "__main__":
    main()

