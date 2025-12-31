"""
=============================================================================
FINAL INTEGRATION: QUANTUM-IoT AI DECISION SYSTEM
=============================================================================
The Grand Finale - Real Hardware Data + Trained AI Brain

This script demonstrates the complete integration:
1. Loads REAL sensor data from the Hardware Team (Varenya/Vanshika)
2. Applies DSP filtering and sensor fusion
3. Loads TRAINED AI MODELS (Autoencoder + PPO Agent)
4. Deploys the AI to make autonomous decisions on Real Data
5. Generates comprehensive visualization dashboard

┌─────────────────────────────────────────────────────────────────────────┐
│                    THE COMPLETE NERVOUS SYSTEM                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   HARDWARE DATA ──► DSP ──► FUSION ──► ANOMALY AI ──► RL AGENT ──► ACT │
│   (Varenya's)      (LPF)    (Merge)    (Autoencoder)   (PPO)      (Ship)│
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

Usage: python main_final_integration.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.gridspec import GridSpec
from datetime import datetime
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import project modules
from data_loader import load_varenya_data, load_vanshika_data, SensorData
from dsp import LowPassFilter
from fusion import fuse_signals, calculate_fusion_quality
from anomaly import AnomalyDetector
from rl_env import SpacecraftEnv

# Import ML libraries
try:
    import torch
    from stable_baselines3 import PPO
except ImportError as e:
    print(f"[ERROR] Missing ML library: {e}")
    print("Run: pip install torch stable-baselines3")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration matching main_ai.py training parameters."""
    
    # Anomaly Detector (MUST match training!)
    ANOMALY_WINDOW_SIZE = 15
    ANOMALY_LATENT_DIM = 6
    
    # Physics - Realistic energy management (more challenging!)
    INITIAL_HEALTH = 100.0
    INITIAL_ENERGY = 100.0
    PASSIVE_RECHARGE = 0.15    # Slow recharge (was 0.5)
    SHIELD_COST = 0.5          # Expensive shield (was 0.1)
    THRUSTER_COST = 1.5        # Very expensive thruster
    RAD_DAMAGE = 5.0
    
    # Paths
    MODEL_DIR = "ai_models"
    ANOMALY_MODEL = "ai_models/anomaly_detector.pt"
    RL_MODEL = "ai_models/ppo_spacecraft"
    
    # Output
    OUTPUT_FILE = "FINAL_INTEGRATED_DEMO.png"


# =============================================================================
# MAIN INTEGRATION FUNCTION
# =============================================================================

def run_final_integration(data_source='varenya', stage='adc'):
    """
    Run the complete AI integration on real hardware data.
    
    Parameters
    ----------
    data_source : str
        'varenya' or 'vanshika'
    stage : str
        For Varenya: 'raw', 'env', or 'adc'
    """
    config = Config()
    
    print("\n" + "="*75)
    print("   🛰️  QUANTUM-IoT SPACECRAFT NERVOUS SYSTEM - FINAL INTEGRATION  🛰️")
    print("="*75)
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*75)
    
    # =========================================================================
    # PHASE 1: LOAD REAL HARDWARE DATA
    # =========================================================================
    print("\n" + "─"*75)
    print("  PHASE 1: LOADING REAL HARDWARE DATA")
    print("─"*75)
    
    try:
        if data_source == 'varenya':
            data = load_varenya_data(use_stage=stage)
        else:
            data = load_vanshika_data()
        
        print(f"  ✓ Source:      {data.source}")
        print(f"  ✓ Samples:     {len(data.time):,}")
        print(f"  ✓ Duration:    {data.duration:.2f} seconds")
        print(f"  ✓ Sample Rate: {data.sample_rate:.0f} Hz")
        print(f"  ✓ Classical:   μ={data.classical.mean():.3f}, σ={data.classical.std():.3f}")
        print(f"  ✓ Quantum:     μ={data.quantum.mean():.3f}, σ={data.quantum.std():.3f}")
        
    except FileNotFoundError as e:
        print(f"  ✗ Error loading data: {e}")
        print("  → Using synthetic fallback data...")
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 1.0 * t)
        data = SensorData(
            time=t,
            ground_truth=signal,
            classical=signal + np.random.normal(0, 0.5, len(t)),
            quantum=signal + np.random.normal(0, 0.05, len(t)),
            sample_rate=100.0,
            duration=10.0,
            source="Synthetic Fallback"
        )
    
    # =========================================================================
    # PHASE 2: DIGITAL SIGNAL PROCESSING & FUSION
    # =========================================================================
    print("\n" + "─"*75)
    print("  PHASE 2: DSP & SENSOR FUSION")
    print("─"*75)
    
    # Apply Low-Pass Filter
    cutoff = min(50.0, data.sample_rate / 4)  # Safe cutoff
    lpf = LowPassFilter(cutoff_freq=cutoff, sample_rate=data.sample_rate, order=4)
    classical_filtered = lpf.apply(data.classical)
    quantum_filtered = lpf.apply(data.quantum)
    print(f"  ✓ Low-Pass Filter: {cutoff:.1f} Hz cutoff")
    
    # Sensor Fusion
    fusion_threshold = 2.0 * np.std(data.classical)
    fusion_result = fuse_signals(
        classical_filtered, 
        quantum_filtered,
        threshold=fusion_threshold,
        quantum_weight=0.8
    )
    fused_signal = fusion_result.fused_signal
    
    # Quality metrics
    quality = calculate_fusion_quality(fused_signal, data.ground_truth)
    print(f"  ✓ Fusion RMSE:        {quality['rmse']:.4f}")
    print(f"  ✓ Fusion Correlation: {quality['correlation']:.4f}")
    print(f"  ✓ Anomalies Rejected: {len(fusion_result.anomaly_indices)}")
    
    # -------------------------------------------------------------------------
    # SIGNAL-TO-NOISE RATIO (SNR) ANALYSIS
    # -------------------------------------------------------------------------
    # Calculate noise as difference from ground truth
    classical_noise = data.classical - data.ground_truth
    quantum_noise = data.quantum - data.ground_truth
    classical_filtered_noise = classical_filtered - data.ground_truth
    quantum_filtered_noise = quantum_filtered - data.ground_truth
    fused_noise = fused_signal - data.ground_truth
    
    # Signal power (ground truth variance)
    signal_power = np.var(data.ground_truth)
    
    # Noise power (variance of noise)
    classical_noise_power = np.var(classical_noise)
    quantum_noise_power = np.var(quantum_noise)
    classical_filtered_noise_power = np.var(classical_filtered_noise)
    quantum_filtered_noise_power = np.var(quantum_filtered_noise)
    fused_noise_power = np.var(fused_noise)
    
    # SNR in dB: SNR = 10 * log10(signal_power / noise_power)
    def calc_snr_db(noise_power):
        if noise_power > 0:
            return 10 * np.log10(signal_power / noise_power)
        return float('inf')
    
    snr_classical_raw = calc_snr_db(classical_noise_power)
    snr_quantum_raw = calc_snr_db(quantum_noise_power)
    snr_classical_filtered = calc_snr_db(classical_filtered_noise_power)
    snr_quantum_filtered = calc_snr_db(quantum_filtered_noise_power)
    snr_fused = calc_snr_db(fused_noise_power)
    
    # Noise reduction percentage
    classical_noise_reduction = 100 * (1 - classical_filtered_noise_power / classical_noise_power) if classical_noise_power > 0 else 0
    quantum_noise_reduction = 100 * (1 - quantum_filtered_noise_power / quantum_noise_power) if quantum_noise_power > 0 else 0
    
    print(f"\n  --- Signal-to-Noise Ratio Analysis ---")
    print(f"  Classical Sensor:")
    print(f"    Raw SNR:      {snr_classical_raw:>8.2f} dB")
    print(f"    Filtered SNR: {snr_classical_filtered:>8.2f} dB")
    print(f"    Noise Reduction: {classical_noise_reduction:>5.1f}%")
    print(f"  Quantum Sensor:")
    print(f"    Raw SNR:      {snr_quantum_raw:>8.2f} dB")
    print(f"    Filtered SNR: {snr_quantum_filtered:>8.2f} dB")
    print(f"    Noise Reduction: {quantum_noise_reduction:>5.1f}%")
    print(f"  Fused Signal:")
    print(f"    SNR:          {snr_fused:>8.2f} dB")
    
    # Store SNR data for visualization
    snr_data = {
        'classical_raw': snr_classical_raw,
        'classical_filtered': snr_classical_filtered,
        'quantum_raw': snr_quantum_raw,
        'quantum_filtered': snr_quantum_filtered,
        'fused': snr_fused,
        'classical_noise_reduction': classical_noise_reduction,
        'quantum_noise_reduction': quantum_noise_reduction,
        'classical_noise': classical_noise,
        'quantum_noise': quantum_noise,
        'fused_noise': fused_noise
    }
    
    # Normalize fused signal to match training distribution (mean=0, similar amplitude)
    # Training data was sin waves oscillating around 0 with amplitude ~1.5
    fused_mean = np.mean(fused_signal)
    fused_std = np.std(fused_signal)
    fused_signal_normalized = (fused_signal - fused_mean) / (fused_std / 1.5)  # Scale to ~1.5 amplitude
    print(f"  ✓ Signal normalized: μ={np.mean(fused_signal_normalized):.3f}, σ={np.std(fused_signal_normalized):.3f}")
    
    # =========================================================================
    # PHASE 3: LOAD TRAINED AI MODELS
    # =========================================================================
    print("\n" + "─"*75)
    print("  PHASE 3: LOADING TRAINED AI MODELS")
    print("─"*75)
    
    # Load Anomaly Detector
    try:
        detector = AnomalyDetector(
            window_size=config.ANOMALY_WINDOW_SIZE,
            latent_dim=config.ANOMALY_LATENT_DIM
        )
        detector.load(config.ANOMALY_MODEL)
        print(f"  ✓ Anomaly Detector loaded (window={config.ANOMALY_WINDOW_SIZE})")
        print(f"    → Threshold: {detector.threshold:.6f}")
    except Exception as e:
        print(f"  ✗ Failed to load Anomaly Detector: {e}")
        print("  → Run 'python main_ai.py' first to train models!")
        return None
    
    # Load RL Agent
    try:
        rl_agent = PPO.load(config.RL_MODEL)
        print(f"  ✓ PPO Agent loaded from {config.RL_MODEL}")
    except Exception as e:
        print(f"  ✗ Failed to load RL Agent: {e}")
        print("  → Run 'python main_ai.py' first to train models!")
        return None
    
    # =========================================================================
    # PHASE 4: ANOMALY DETECTION
    # =========================================================================
    print("\n" + "─"*75)
    print("  PHASE 4: ANOMALY DETECTION (AUTOENCODER)")
    print("─"*75)
    
    # Detect anomalies
    anomaly_flags, reconstruction_errors = detector.detect_batch(fused_signal)
    
    # Pad to match signal length
    n_samples = len(fused_signal)
    full_errors = np.zeros(n_samples)
    full_flags = np.zeros(n_samples, dtype=bool)
    
    offset = config.ANOMALY_WINDOW_SIZE - 1
    if len(reconstruction_errors) > 0:
        end_idx = min(offset + len(reconstruction_errors), n_samples)
        full_errors[offset:end_idx] = reconstruction_errors[:end_idx - offset]
        full_flags[offset:end_idx] = anomaly_flags[:end_idx - offset]
    
    # Normalize errors to [0, 1] range for the observation space
    # and create a sensible danger threshold based on the actual data distribution
    max_err = np.max(full_errors) if np.max(full_errors) > 0 else 1.0
    normalized_errors = full_errors / max_err  # Now in [0, 1]
    
    # Use 80th percentile as danger threshold (top 20% most anomalous = danger)
    # This ensures the agent sees a reasonable number of danger events
    nonzero_errors = full_errors[full_errors > 0]
    if len(nonzero_errors) > 0:
        dynamic_threshold = np.percentile(nonzero_errors, 80) / max_err
    else:
        dynamic_threshold = 0.5
    
    # Create binary danger mask
    danger_mask_new = normalized_errors > dynamic_threshold
    
    n_anomalies = np.sum(full_flags)
    anomaly_pct = 100 * n_anomalies / n_samples
    n_danger = np.sum(danger_mask_new)
    print(f"  ✓ Detected {n_anomalies:,} anomalous windows ({anomaly_pct:.1f}%)")
    print(f"  ✓ Raw error range: [{full_errors.min():.4f}, {full_errors.max():.4f}]")
    print(f"  ✓ Normalized error range: [0.0, 1.0]")
    print(f"  ✓ Dynamic danger threshold: {dynamic_threshold:.4f} (80th percentile)")
    print(f"  ✓ Danger events: {n_danger:,} ({100*n_danger/n_samples:.1f}%)")
    
    # =========================================================================
    # PHASE 5: RL AGENT DEPLOYMENT
    # =========================================================================
    print("\n" + "─"*75)
    print("  PHASE 5: RL AGENT DEPLOYMENT (PPO)")
    print("─"*75)
    print("  → Processing every frame with trained agent...")
    
    # Initialize state
    health = config.INITIAL_HEALTH
    energy = config.INITIAL_ENERGY
    
    # History tracking
    history = {
        'health': [],
        'energy': [],
        'actions': [],
        'rewards': [],
        'anomalies': [],
        'errors': []
    }
    
    total_reward = 0.0
    action_names = {0: "Idle", 1: "Shield", 2: "Thruster"}
    action_counts = {0: 0, 1: 0, 2: 0}
    ai_overrides = 0  # Count safety wrapper overrides
    
    # Process each timestep
    for i in range(n_samples):
        # Create observation (matching rl_env.py format)
        # Use normalized signal and normalized errors
        obs = np.array([
            np.clip(fused_signal_normalized[i], -5.0, 5.0),  # Normalized signal
            health,                                           # Health %
            energy,                                           # Energy %
            normalized_errors[i]                              # Normalized error [0, 1]
        ], dtype=np.float32)
        
        # Get AI decision
        ai_action, _ = rl_agent.predict(obs, deterministic=True)
        ai_action = int(ai_action)
        
        # Determine if this is a danger zone
        is_danger = danger_mask_new[i]
        
        # SAFETY WRAPPER: Override AI if it would cause damage
        # This demonstrates a common AI safety pattern: human-designed guardrails
        if is_danger and ai_action == 0:
            # AI chose idle during danger - override with shield for survival
            action = 1  # Force shield
            ai_overrides += 1
        else:
            action = ai_action
        
        action_counts[action] += 1
        
        # Apply physics (matching rl_env.py exactly!)
        reward = 0.0
        
        # Energy dynamics
        if action == 1:  # Shield
            energy -= config.SHIELD_COST
        elif action == 2:  # Thruster
            energy -= config.THRUSTER_COST
        else:  # Idle
            energy += config.PASSIVE_RECHARGE
        energy = np.clip(energy, 0, 100)
        
        # Health & Reward (matching rl_env.py reward structure)
        reward += 0.05 * (energy / 100.0)  # Energy greed
        
        if is_danger:
            if action == 1:  # Shield during danger = good
                reward += 1.5
            elif action == 2:  # Thruster during danger = okay
                reward += 0.5
            else:  # Idle during danger = bad
                health -= config.RAD_DAMAGE
                reward -= 6.0
        else:
            if action == 1:  # Shield when safe = wasteful
                reward -= 7.0
            elif action == 2:  # Thruster when safe = wasteful
                reward -= 8.0
            else:  # Idle when safe = good
                if health > 80:
                    reward += 0.5
                reward += 3.0
        
        health = np.clip(health, 0, 100)
        total_reward += reward
        
        # Record history
        history['health'].append(health)
        history['energy'].append(energy)
        history['actions'].append(action)
        history['rewards'].append(reward)
        history['anomalies'].append(is_danger)
        history['errors'].append(normalized_errors[i])
        
        # Progress updates
        if (i + 1) % 1000 == 0:
            print(f"    Step {i+1:,}/{n_samples:,}: Health={health:.1f}%, "
                  f"Energy={energy:.1f}%, Action={action_names[action]}")
        
        # Check for death
        if health <= 0:
            print(f"  ⚠ Spacecraft destroyed at step {i+1}!")
            break
    
    # Final statistics
    survived = len(history['health'])
    print(f"\n  ═══════════════════════════════════════════════════════")
    print(f"  ✓ SIMULATION COMPLETE")
    print(f"  ═══════════════════════════════════════════════════════")
    print(f"    Steps Survived:  {survived:,} / {n_samples:,}")
    print(f"    Final Health:    {health:.1f}%")
    print(f"    Final Energy:    {energy:.1f}%")
    print(f"    Total Reward:    {total_reward:.2f}")
    print(f"    Actions Taken:")
    print(f"      • Idle:        {action_counts[0]:,} ({100*action_counts[0]/survived:.1f}%)")
    print(f"      • Shield:      {action_counts[1]:,} ({100*action_counts[1]/survived:.1f}%)")
    print(f"      • Thruster:    {action_counts[2]:,} ({100*action_counts[2]/survived:.1f}%)")
    print(f"    Safety Overrides: {ai_overrides:,} (AI→Shield when danger)")
    
    # =========================================================================
    # PHASE 6: GENERATE VISUALIZATION
    # =========================================================================
    print("\n" + "─"*75)
    print("  PHASE 6: GENERATING VISUALIZATION DASHBOARD")
    print("─"*75)
    
    results = {
        'time': data.time[:survived],
        'classical': data.classical[:survived],
        'quantum': data.quantum[:survived],
        'ground_truth': data.ground_truth[:survived],
        'fused_signal': fused_signal[:survived],
        'classical_filtered': classical_filtered[:survived],
        'quantum_filtered': quantum_filtered[:survived],
        'history': history,
        'detector': detector,
        'full_errors': normalized_errors[:survived],
        'threshold': dynamic_threshold,
        'quality': quality,
        'snr_data': snr_data,
        'stats': {
            'survived': survived,
            'total_samples': n_samples,
            'final_health': health,
            'final_energy': energy,
            'total_reward': total_reward,
            'action_counts': action_counts,
            'anomaly_count': n_anomalies
        },
        'source': data.source
    }
    
    create_final_dashboard(results, config)
    
    return results


# =============================================================================
# VISUALIZATION DASHBOARD
# =============================================================================

def create_final_dashboard(results, config):
    """Create a comprehensive visualization dashboard with LIGHT THEME."""
    
    # Extract data
    t = results['time']
    history = results['history']
    errors = results['full_errors']
    threshold = results['threshold']
    stats = results['stats']
    snr = results['snr_data']
    
    # Set up the figure with LIGHT theme
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Force legends to be completely opaque
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams['legend.facecolor'] = 'white'
    plt.rcParams['legend.edgecolor'] = '#444444'
    
    fig = plt.figure(figsize=(18, 24))
    fig.patch.set_facecolor('#ffffff')
    
    # Create grid layout - 6 rows with proper spacing
    gs = GridSpec(6, 2, figure=fig, height_ratios=[1.0, 0.7, 1.0, 1.0, 0.7, 0.5], 
                  hspace=0.50, wspace=0.25, top=0.95, bottom=0.04)
    
    # Color palette for light theme
    colors = {
        'classical': '#e74c3c',      # Red
        'quantum': '#27ae60',        # Green
        'fused': '#9b59b6',          # Purple
        'ground_truth': '#f39c12',   # Orange
        'health': '#2ecc71',         # Light green
        'energy': '#3498db',         # Blue
        'danger': '#e74c3c',         # Red
        'safe': '#27ae60',           # Green
        'shield': '#f1c40f',         # Yellow
        'thruster': '#e74c3c',       # Red
        'idle': '#95a5a6',           # Gray
        'error': '#8e44ad',          # Purple
        'text': '#2c3e50',           # Dark blue-gray
        'grid': '#ecf0f1'            # Light gray
    }
    
    # =========================================================================
    # ROW 1: SENSOR SIGNALS (Full width)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor('#fafafa')
    
    ax1.plot(t, results['classical'], color=colors['classical'], alpha=0.6, 
             linewidth=0.8, label='Classical Sensor (Noisy)')
    ax1.plot(t, results['quantum'], color=colors['quantum'], alpha=0.6,
             linewidth=0.8, label='Quantum Sensor (Precise)')
    ax1.plot(t, results['fused_signal'], color=colors['fused'], 
             linewidth=2, label='Fused Signal')
    ax1.plot(t, results['ground_truth'], color=colors['ground_truth'],
             linewidth=2, linestyle='--', alpha=0.9, label='Ground Truth')
    
    ax1.set_title('Sensor Fusion Result', 
                  fontsize=10, fontweight='bold', color=colors['text'], pad=5)
    ax1.set_ylabel('Amplitude', color=colors['text'], fontsize=11)
    legend1 = ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7,
                         framealpha=1.0, fancybox=False, facecolor='white', 
                         edgecolor='#444444', borderaxespad=0)
    legend1.get_frame().set_linewidth(1.0)
    ax1.set_xlim(t[0], t[-1])
    ax1.grid(True, alpha=0.4)
    ax1.tick_params(labelsize=9)
    ax1.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
    
    # Mini stats box
    stats_text = (f"Health: {stats['final_health']:.0f}% | "
                  f"Energy: {stats['final_energy']:.0f}% | "
                  f"Survived: {stats['survived']:,}/{stats['total_samples']:,} | "
                  f"Reward: {stats['total_reward']:.0f}")
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes,
             fontsize=9, fontfamily='monospace', color=colors['text'],
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                      edgecolor='#bdc3c7', alpha=0.95))
    
    # =========================================================================
    # ROW 2: SNR COMPARISON (Bar charts)
    # =========================================================================
    # Left: SNR Bar Chart
    ax_snr = fig.add_subplot(gs[1, 0])
    ax_snr.set_facecolor('#fafafa')
    
    # Short labels to prevent overlap
    snr_labels = ['C-Raw', 'C-Filt', 'Q-Raw', 'Q-Filt', 'Fused']
    snr_values = [snr['classical_raw'], snr['classical_filtered'], 
                  snr['quantum_raw'], snr['quantum_filtered'], snr['fused']]
    snr_colors = [colors['classical'], colors['classical'], 
                  colors['quantum'], colors['quantum'], colors['fused']]
    
    x_pos = np.arange(len(snr_labels))
    bars = ax_snr.bar(x_pos, snr_values, color=snr_colors, alpha=0.8, edgecolor='white', linewidth=1.5, width=0.6)
    ax_snr.set_xticks(x_pos)
    ax_snr.set_xticklabels(snr_labels, fontsize=8)
    ax_snr.set_ylabel('SNR (dB)', fontsize=10, color=colors['text'])
    ax_snr.set_title('Signal-to-Noise Ratio', fontsize=10, fontweight='bold', 
                     color=colors['text'], pad=6)
    ax_snr.tick_params(labelsize=9)
    ax_snr.grid(True, axis='y', alpha=0.4)
    
    # Add value labels INSIDE the bars (centered)
    for bar, val in zip(bars, snr_values):
        # Position label inside bar, near the end
        if val >= 0:
            y_pos = bar.get_height() / 2
        else:
            y_pos = bar.get_height() / 2  # Middle of negative bar
        ax_snr.text(bar.get_x() + bar.get_width()/2, y_pos, 
                   f'{val:.1f}', ha='center', va='center', fontsize=8, fontweight='bold',
                   color='white')
    
    # Right: Noise Reduction
    ax_noise = fig.add_subplot(gs[1, 1])
    ax_noise.set_facecolor('#fafafa')
    
    noise_labels = ['Classical', 'Quantum']
    noise_values = [snr['classical_noise_reduction'], snr['quantum_noise_reduction']]
    noise_colors = [colors['classical'], colors['quantum']]
    
    x_pos_noise = np.arange(len(noise_labels))
    bars2 = ax_noise.bar(x_pos_noise, noise_values, color=noise_colors, alpha=0.8, 
                         edgecolor='white', linewidth=1.5, width=0.4)
    ax_noise.set_xticks(x_pos_noise)
    ax_noise.set_xticklabels(noise_labels, fontsize=9)
    ax_noise.set_ylabel('Reduction (%)', fontsize=10, color=colors['text'])
    ax_noise.set_title('Noise Reduction (Low-Pass Filter)', fontsize=10, fontweight='bold',
                       color=colors['text'], pad=6)
    max_noise_val = max(noise_values) if max(noise_values) > 0 else 100
    ax_noise.set_ylim(0, max_noise_val * 1.4)
    ax_noise.tick_params(labelsize=9)
    ax_noise.grid(True, axis='y', alpha=0.4)
    
    for bar, val in zip(bars2, noise_values):
        ax_noise.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max_noise_val * 0.02), 
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold',
                     color=colors['text'])
    
    # =========================================================================
    # ROW 3: ANOMALY DETECTION
    # =========================================================================
    ax2 = fig.add_subplot(gs[2, :])
    ax2.set_facecolor('#fafafa')
    
    ax2.fill_between(t, 0, errors, color=colors['error'], alpha=0.3)
    ax2.plot(t, errors, color=colors['error'], linewidth=1, label='Reconstruction Error')
    ax2.axhline(y=threshold, color=colors['danger'], linestyle='--', 
                linewidth=2, label=f'Danger Threshold ({threshold:.4f})')
    
    danger_mask = np.array(history['anomalies'])
    for i in range(len(t) - 1):
        if danger_mask[i]:
            ax2.axvspan(t[i], t[i+1], alpha=0.2, color=colors['danger'], linewidth=0)
    
    ax2.set_title('Autoencoder Anomaly Detection', 
                  fontsize=11, fontweight='bold', color=colors['text'], pad=8)
    ax2.set_ylabel('Reconstruction Error', color=colors['text'], fontsize=11)
    legend2 = ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7,
                         framealpha=1.0, fancybox=False, facecolor='white',
                         edgecolor='#444444', borderaxespad=0)
    legend2.get_frame().set_linewidth(1.0)
    ax2.set_xlim(t[0], t[-1])
    ax2.grid(True, alpha=0.4)
    ax2.tick_params(labelsize=9)
    ax2.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
    
    # =========================================================================
    # ROW 4: SPACECRAFT STATE
    # =========================================================================
    ax3 = fig.add_subplot(gs[3, :])
    ax3.set_facecolor('#fafafa')
    
    health_arr = np.array(history['health'])
    energy_arr = np.array(history['energy'])
    
    ax3.fill_between(t, 0, health_arr, color=colors['health'], alpha=0.3)
    ax3.plot(t, health_arr, color=colors['health'], linewidth=2.5, label='Health')
    
    ax3.fill_between(t, 0, energy_arr, color=colors['energy'], alpha=0.2)
    ax3.plot(t, energy_arr, color=colors['energy'], linewidth=2.5, label='Energy')
    
    ax3.axhline(y=50, color='#e67e22', linestyle=':', alpha=0.7, linewidth=1.5, label='Warning Level')
    ax3.axhline(y=20, color=colors['danger'], linestyle=':', alpha=0.7, linewidth=1.5, label='Critical Level')
    
    ax3.set_title('Spacecraft System Status', 
                  fontsize=11, fontweight='bold', color=colors['text'], pad=8)
    ax3.set_ylabel('Percentage (%)', color=colors['text'], fontsize=11)
    ax3.set_ylim(-5, 105)
    legend3 = ax3.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=7,
                         framealpha=1.0, fancybox=False, facecolor='white',
                         edgecolor='#444444', borderaxespad=0)
    legend3.get_frame().set_linewidth(1.0)
    ax3.set_xlim(t[0], t[-1])
    ax3.grid(True, alpha=0.4)
    ax3.tick_params(labelsize=9)
    ax3.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
    
    # =========================================================================
    # ROW 5: AI ACTIONS
    # =========================================================================
    ax4 = fig.add_subplot(gs[4, :])
    ax4.set_facecolor('#fafafa')
    
    actions = np.array(history['actions'])
    
    for i in range(len(t) - 1):
        action = actions[i]
        if action == 0:
            color, alpha = colors['idle'], 0.4
        elif action == 1:
            color, alpha = colors['shield'], 0.8
        else:
            color, alpha = colors['thruster'], 0.9
        ax4.axvspan(t[i], t[i+1], alpha=alpha, color=color, linewidth=0)
    
    for i in range(len(t) - 1):
        if danger_mask[i]:
            ax4.plot([t[i], t[i+1]], [2.8, 2.8], color=colors['danger'], linewidth=4)
    
    ax4.set_yticks([0.5, 1.5, 2.5])
    ax4.set_yticklabels(['IDLE', 'SHIELD', 'THRUSTER'], fontsize=10)
    ax4.set_ylim(0, 3)
    ax4.set_xlim(t[0], t[-1])
    
    legend_elements = [
        Patch(facecolor=colors['idle'], alpha=0.6, label=f"Idle ({stats['action_counts'][0]:,})"),
        Patch(facecolor=colors['shield'], alpha=0.9, label=f"Shield ({stats['action_counts'][1]:,})"),
        Patch(facecolor=colors['thruster'], alpha=0.9, label=f"Thruster ({stats['action_counts'][2]:,})"),
        Patch(facecolor=colors['danger'], alpha=0.9, label="Danger Zone")
    ]
    legend4 = ax4.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1), 
                         fontsize=7, framealpha=1.0, fancybox=False, facecolor='white',
                         edgecolor='#444444', borderaxespad=0)
    legend4.get_frame().set_linewidth(1.0)
    ax4.set_title('PPO Agent Decisions', 
                  fontsize=11, fontweight='bold', color=colors['text'], pad=8)
    ax4.grid(True, axis='x', alpha=0.4)
    ax4.tick_params(labelsize=9)
    ax4.tick_params(axis='x', labelbottom=False)  # Hide x-axis labels
    
    # =========================================================================
    # ROW 6: CUMULATIVE REWARD
    # =========================================================================
    ax5 = fig.add_subplot(gs[5, :])
    ax5.set_facecolor('#fafafa')
    
    cumulative_reward = np.cumsum(history['rewards'])
    
    ax5.fill_between(t, 0, cumulative_reward, 
                     where=cumulative_reward >= 0, color=colors['safe'], alpha=0.3)
    ax5.fill_between(t, 0, cumulative_reward,
                     where=cumulative_reward < 0, color=colors['danger'], alpha=0.3)
    ax5.plot(t, cumulative_reward, color=colors['text'], linewidth=2)
    ax5.axhline(y=0, color=colors['text'], linestyle='-', alpha=0.3)
    
    ax5.set_title('Cumulative Reward', 
                  fontsize=11, fontweight='bold', color=colors['text'], pad=8)
    ax5.set_xlabel('Time (seconds)', color=colors['text'], fontsize=12)
    ax5.set_ylabel('Reward', color=colors['text'], fontsize=11)
    ax5.set_xlim(t[0], t[-1])
    ax5.grid(True, alpha=0.4)
    ax5.tick_params(labelsize=9)
    
    # =========================================================================
    # MAIN TITLE
    # =========================================================================
    fig.suptitle(
        'Quantum-IoT Spacecraft Nervous System',
        fontsize=13, fontweight='bold', color=colors['text'], y=0.995
    )
    
    # Manually adjust subplot positions - leave room for legends on right
    plt.subplots_adjust(
        left=0.06,
        right=0.85,
        top=0.96,
        bottom=0.04,
        hspace=0.45,
        wspace=0.22
    )
    
    plt.savefig(config.OUTPUT_FILE, dpi=150, facecolor='white', 
                edgecolor='none', bbox_inches='tight')
    print(f"  [OK] Dashboard saved to: {config.OUTPUT_FILE}")
    
    plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Final Integration Demo')
    parser.add_argument('--source', choices=['varenya', 'vanshika'], 
                        default='varenya', help='Data source')
    parser.add_argument('--stage', choices=['raw', 'env', 'adc'],
                        default='adc', help='Varenya data stage')
    
    args = parser.parse_args()
    
    results = run_final_integration(data_source=args.source, stage=args.stage)
    
    if results:
        print("\n" + "="*75)
        print("   ✅ FINAL INTEGRATION COMPLETE!")
        print("="*75)
        print(f"   Output: {Config.OUTPUT_FILE}")
        print("="*75 + "\n")
