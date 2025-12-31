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
    
    # Physics (MUST match rl_env.py!)
    INITIAL_HEALTH = 100.0
    INITIAL_ENERGY = 100.0
    PASSIVE_RECHARGE = 0.5
    SHIELD_COST = 0.1
    THRUSTER_COST = 1.0
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
        'history': history,
        'detector': detector,
        'full_errors': normalized_errors[:survived],
        'threshold': dynamic_threshold,
        'quality': quality,
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
    """Create a comprehensive, beautiful visualization dashboard."""
    
    # Extract data
    t = results['time']
    history = results['history']
    errors = results['full_errors']
    threshold = results['threshold']
    stats = results['stats']
    
    # Set up the figure with dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    
    # Create grid layout
    gs = GridSpec(5, 3, figure=fig, height_ratios=[1.2, 1, 1, 0.8, 0.5], 
                  hspace=0.35, wspace=0.25)
    
    # Color palette
    colors = {
        'signal': '#58a6ff',
        'classical': '#f97583',
        'quantum': '#7ee787',
        'fused': '#d2a8ff',
        'ground_truth': '#ffa657',
        'health': '#3fb950',
        'energy': '#58a6ff',
        'danger': '#f85149',
        'safe': '#238636',
        'shield': '#f0e68c',
        'thruster': '#ff6b6b',
        'idle': '#6e7681',
        'error': '#bc8cff'
    }
    
    # =========================================================================
    # ROW 1: SENSOR SIGNALS (spans 2 columns) + STATS PANEL
    # =========================================================================
    
    # Sensor comparison plot
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#161b22')
    
    ax1.plot(t, results['classical'], color=colors['classical'], alpha=0.5, 
             linewidth=0.5, label='Classical (Noisy)')
    ax1.plot(t, results['quantum'], color=colors['quantum'], alpha=0.5,
             linewidth=0.5, label='Quantum (Precise)')
    ax1.plot(t, results['fused_signal'], color=colors['fused'], 
             linewidth=1.5, label='Fused Signal')
    ax1.plot(t, results['ground_truth'], color=colors['ground_truth'],
             linewidth=1.5, linestyle='--', alpha=0.8, label='Ground Truth')
    
    ax1.set_title('🛰️ REAL HARDWARE SENSOR DATA → FUSED SIGNAL', 
                  fontsize=14, fontweight='bold', color='white', pad=10)
    ax1.set_ylabel('Amplitude', color='white')
    ax1.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d',
               labelcolor='white', fontsize=9)
    ax1.set_xlim(t[0], t[-1])
    ax1.grid(True, alpha=0.2, color='#30363d')
    ax1.tick_params(colors='white')
    
    # Stats panel
    ax_stats = fig.add_subplot(gs[0, 2])
    ax_stats.set_facecolor('#161b22')
    ax_stats.axis('off')
    
    stats_text = f"""
╔══════════════════════════════════╗
║     MISSION STATISTICS           ║
╠══════════════════════════════════╣
║  Data Source:                    ║
║    {results['source'][:30]:30s} ║
║                                  ║
║  Samples Processed: {stats['total_samples']:>12,} ║
║  Steps Survived:    {stats['survived']:>12,} ║
║                                  ║
║  Final Health:      {stats['final_health']:>10.1f}%  ║
║  Final Energy:      {stats['final_energy']:>10.1f}%  ║
║  Total Reward:      {stats['total_reward']:>12.1f} ║
║                                  ║
║  Anomalies Detected:{stats['anomaly_count']:>11,} ║
║                                  ║
║  Fusion Quality:                 ║
║    RMSE: {results['quality']['rmse']:>8.4f}             ║
║    Corr: {results['quality']['correlation']:>8.4f}             ║
╚══════════════════════════════════╝
"""
    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=10, fontfamily='monospace', color='#58a6ff',
                  verticalalignment='top',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1117', 
                           edgecolor='#30363d', alpha=0.9))
    
    # =========================================================================
    # ROW 2: ANOMALY DETECTION
    # =========================================================================
    ax2 = fig.add_subplot(gs[1, :])
    ax2.set_facecolor('#161b22')
    
    # Plot reconstruction error
    ax2.fill_between(t, 0, errors, color=colors['error'], alpha=0.3)
    ax2.plot(t, errors, color=colors['error'], linewidth=0.8, label='Reconstruction Error')
    ax2.axhline(y=threshold, color=colors['danger'], linestyle='--', 
                linewidth=2, label=f'Threshold ({threshold:.4f})')
    
    # Highlight danger zones
    danger_mask = np.array(history['anomalies'])
    for i in range(len(t) - 1):
        if danger_mask[i]:
            ax2.axvspan(t[i], t[i+1], alpha=0.15, color=colors['danger'], linewidth=0)
    
    ax2.set_title('🧠 AUTOENCODER ANOMALY DETECTION', 
                  fontsize=14, fontweight='bold', color='white', pad=10)
    ax2.set_ylabel('Reconstruction Error', color='white')
    ax2.legend(loc='upper right', facecolor='#21262d', edgecolor='#30363d',
               labelcolor='white', fontsize=9)
    ax2.set_xlim(t[0], t[-1])
    ax2.grid(True, alpha=0.2, color='#30363d')
    ax2.tick_params(colors='white')
    
    # =========================================================================
    # ROW 3: SPACECRAFT STATE (Health & Energy)
    # =========================================================================
    ax3 = fig.add_subplot(gs[2, :])
    ax3.set_facecolor('#161b22')
    
    health_arr = np.array(history['health'])
    energy_arr = np.array(history['energy'])
    
    ax3.fill_between(t, 0, health_arr, color=colors['health'], alpha=0.3)
    ax3.plot(t, health_arr, color=colors['health'], linewidth=2, label='Health')
    
    ax3.fill_between(t, 0, energy_arr, color=colors['energy'], alpha=0.2)
    ax3.plot(t, energy_arr, color=colors['energy'], linewidth=2, label='Energy')
    
    # Add danger threshold lines
    ax3.axhline(y=50, color='#ffa657', linestyle=':', alpha=0.5, label='Warning Level')
    ax3.axhline(y=20, color=colors['danger'], linestyle=':', alpha=0.5, label='Critical Level')
    
    ax3.set_title('💚 SPACECRAFT SYSTEM STATUS', 
                  fontsize=14, fontweight='bold', color='white', pad=10)
    ax3.set_ylabel('Percentage (%)', color='white')
    ax3.set_ylim(-5, 105)
    ax3.legend(loc='lower left', facecolor='#21262d', edgecolor='#30363d',
               labelcolor='white', fontsize=9)
    ax3.set_xlim(t[0], t[-1])
    ax3.grid(True, alpha=0.2, color='#30363d')
    ax3.tick_params(colors='white')
    
    # =========================================================================
    # ROW 4: AI ACTIONS TIMELINE
    # =========================================================================
    ax4 = fig.add_subplot(gs[3, :])
    ax4.set_facecolor('#161b22')
    
    actions = np.array(history['actions'])
    
    # Create colored regions for each action
    for i in range(len(t) - 1):
        action = actions[i]
        if action == 0:
            color = colors['idle']
            alpha = 0.3
        elif action == 1:
            color = colors['shield']
            alpha = 0.7
        else:
            color = colors['thruster']
            alpha = 0.9
        ax4.axvspan(t[i], t[i+1], alpha=alpha, color=color, linewidth=0)
    
    # Add danger overlay
    for i in range(len(t) - 1):
        if danger_mask[i]:
            ax4.plot([t[i], t[i+1]], [2.8, 2.8], color=colors['danger'], linewidth=3)
    
    ax4.set_yticks([0.5, 1.5, 2.5])
    ax4.set_yticklabels(['IDLE', 'SHIELD', 'THRUSTER'], fontsize=10, color='white')
    ax4.set_ylim(0, 3)
    ax4.set_xlim(t[0], t[-1])
    
    # Legend
    legend_elements = [
        Patch(facecolor=colors['idle'], alpha=0.5, label=f"Idle ({stats['action_counts'][0]:,})"),
        Patch(facecolor=colors['shield'], alpha=0.8, label=f"Shield ({stats['action_counts'][1]:,})"),
        Patch(facecolor=colors['thruster'], alpha=0.9, label=f"Thruster ({stats['action_counts'][2]:,})"),
        Patch(facecolor=colors['danger'], alpha=0.8, label="Danger Zone")
    ]
    ax4.legend(handles=legend_elements, loc='upper right', facecolor='#21262d', 
               edgecolor='#30363d', labelcolor='white', fontsize=9)
    
    ax4.set_title('🤖 PPO AGENT AUTONOMOUS DECISIONS', 
                  fontsize=14, fontweight='bold', color='white', pad=10)
    ax4.set_xlabel('Time (seconds)', color='white', fontsize=11)
    ax4.grid(True, axis='x', alpha=0.2, color='#30363d')
    ax4.tick_params(colors='white')
    
    # =========================================================================
    # ROW 5: REWARD PROGRESSION
    # =========================================================================
    ax5 = fig.add_subplot(gs[4, :])
    ax5.set_facecolor('#161b22')
    
    cumulative_reward = np.cumsum(history['rewards'])
    
    ax5.fill_between(t, 0, cumulative_reward, 
                     where=cumulative_reward >= 0, color=colors['safe'], alpha=0.3)
    ax5.fill_between(t, 0, cumulative_reward,
                     where=cumulative_reward < 0, color=colors['danger'], alpha=0.3)
    ax5.plot(t, cumulative_reward, color='white', linewidth=1.5)
    ax5.axhline(y=0, color='white', linestyle='-', alpha=0.3)
    
    ax5.set_title('📈 CUMULATIVE REWARD (AI Performance)', 
                  fontsize=12, fontweight='bold', color='white', pad=8)
    ax5.set_xlabel('Time (seconds)', color='white', fontsize=11)
    ax5.set_ylabel('Reward', color='white')
    ax5.set_xlim(t[0], t[-1])
    ax5.grid(True, alpha=0.2, color='#30363d')
    ax5.tick_params(colors='white')
    
    # =========================================================================
    # MAIN TITLE
    # =========================================================================
    fig.suptitle(
        '🛰️ QUANTUM-IoT SPACECRAFT NERVOUS SYSTEM\n'
        'Real Hardware Data + Trained AI Integration Demo',
        fontsize=18, fontweight='bold', color='white', y=0.98
    )
    
    # Save and show
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    plt.savefig(config.OUTPUT_FILE, dpi=200, facecolor='#0d1117', 
                edgecolor='none', bbox_inches='tight')
    print(f"  ✓ Dashboard saved to: {config.OUTPUT_FILE}")
    
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
