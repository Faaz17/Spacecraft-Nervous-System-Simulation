"""
=============================================================================
MAIN AI ORCHESTRATOR - Advanced Decision System
=============================================================================

This is the main script that demonstrates the complete AI-powered decision
system for spacecraft operations. It integrates:

1. ANOMALY DETECTION (anomaly.py)
   - PyTorch Autoencoder trained on normal sensor patterns
   - Detects radiation spikes, structural stress, communication delays

2. REINFORCEMENT LEARNING (rl_env.py)
   - Custom Gymnasium environment simulating spacecraft
   - PPO agent learns optimal response policies

3. SENSOR FUSION (fusion.py)
   - Combines classical and quantum sensor data
   - Weights signals based on reliability

EXECUTION PHASES:
-----------------
┌────────────────────────────────────────────────────────────────────────┐
│  PHASE 1: ANOMALY DETECTOR TRAINING                                    │
│  • Generate 1000+ points of normal sensor data                         │
│  • Train autoencoder to learn normal patterns                          │
│  • Set anomaly threshold based on reconstruction error                 │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 2: RL AGENT TRAINING                                            │
│  • Create spacecraft environment with sensor data                      │
│  • Train PPO agent for 10,000+ timesteps                              │
│  • Agent learns: when to act, what action to take, conserve energy    │
├────────────────────────────────────────────────────────────────────────┤
│  PHASE 3: DEPLOYMENT SIMULATION                                        │
│  • Generate test data with realistic anomalies                         │
│  • Run inference loop:                                                 │
│    1. Fuse sensor signals                                              │
│    2. Detect anomalies with autoencoder                                │
│    3. Feed state to RL agent → Get action                              │
│    4. Execute action → Update system                                   │
│  • Visualize results and save report                                   │
└────────────────────────────────────────────────────────────────────────┘

Author: Spacecraft Nervous System Simulation Team
Course: Quantum-IoT Hybrid Network for Spacecraft Decision Systems
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import our modules
from data_gen import generate_telemetry
from dsp import LowPassFilter
from fusion import fuse_signals
from anomaly import AnomalyDetector
from rl_env import SpacecraftEnv

# Import ML libraries
try:
    import torch
    print(f"[SYSTEM] PyTorch version: {torch.__version__}")
except ImportError:
    print("[ERROR] PyTorch not installed! Run: pip install torch")
    exit(1)

try:
    import gymnasium as gym
    print(f"[SYSTEM] Gymnasium available")
except ImportError:
    print("[ERROR] Gymnasium not installed! Run: pip install gymnasium")
    exit(1)

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    print(f"[SYSTEM] Stable-Baselines3 available")
except ImportError:
    print("[ERROR] Stable-Baselines3 not installed! Run: pip install stable-baselines3")
    exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """Configuration parameters for the AI system."""
    
    # Data generation
    NORMAL_DATA_DURATION = 20      # seconds of normal data for training
    NORMAL_DATA_RATE = 100         # samples per second
    TEST_DATA_DURATION = 10        # seconds of test data
    TEST_DATA_RATE = 100
    
    # Anomaly detector
    ANOMALY_WINDOW_SIZE = 15       # samples per window
    ANOMALY_LATENT_DIM = 6         # autoencoder bottleneck size
    ANOMALY_EPOCHS = 80            # training epochs
    ANOMALY_THRESHOLD_PERCENTILE = 99.5  # Balanced to allow more safe periods while avoiding false spikes
    
    # RL training
    RL_TIMESTEPS = 30000           # Training steps (per request)
    RL_LEARNING_RATE = 0.0003      # PPO learning rate
    RL_N_STEPS = 512               # Steps before update
    RL_BATCH_SIZE = 64             # minibatch size
    
    # Simulation
    INITIAL_HEALTH = 100.0
    INITIAL_ENERGY = 100.0
    ENERGY_REGEN_RATE = 0.05
    
    # Output
    SAVE_MODELS = True
    OUTPUT_DIR = "ai_models"


# =============================================================================
# TRAINING CALLBACK
# =============================================================================

class TrainingCallback(BaseCallback):
    """Custom callback for monitoring RL training progress."""
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode info when available
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            if 'r' in ep_info:
                self.episode_rewards.append(ep_info['r'])
                self.episode_lengths.append(ep_info['l'])
        return True


# =============================================================================
# PHASE 1: TRAIN ANOMALY DETECTOR
# =============================================================================

def train_anomaly_detector(config: Config) -> AnomalyDetector:
    """
    Train the autoencoder-based anomaly detector on normal data.
    
    The key insight is: train on ONLY normal data, so the model
    learns what 'normal' looks like. Anything different = anomaly.
    """
    print("\n" + "="*70)
    print("PHASE 1: TRAINING ANOMALY DETECTOR")
    print("="*70)
    
    # Generate normal training data (clean sine wave with minimal noise)
    print("\n[PHASE 1] Generating normal training data...")
    n_samples = config.NORMAL_DATA_DURATION * config.NORMAL_DATA_RATE
    t = np.linspace(0, config.NORMAL_DATA_DURATION, n_samples)
    
    # Multi-frequency signal to simulate realistic vibration data
    normal_signal = (
        np.sin(2 * np.pi * 1.0 * t) +           # Primary frequency
        0.3 * np.sin(2 * np.pi * 2.5 * t) +     # Harmonic
        0.1 * np.sin(2 * np.pi * 0.5 * t) +     # Low frequency drift
        np.random.normal(0, 0.05, n_samples)     # Small sensor noise
    )
    
    print(f"[PHASE 1] Generated {n_samples} normal samples")
    
    # Create and train detector
    detector = AnomalyDetector(
        window_size=config.ANOMALY_WINDOW_SIZE,
        latent_dim=config.ANOMALY_LATENT_DIM,
        threshold_percentile=config.ANOMALY_THRESHOLD_PERCENTILE
    )
    
    history = detector.train(
        normal_signal,
        epochs=config.ANOMALY_EPOCHS,
        batch_size=32,
        learning_rate=0.001,
        verbose=True
    )
    
    # Save model
    if config.SAVE_MODELS:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        detector.save(f"{config.OUTPUT_DIR}/anomaly_detector.pt")
    
    return detector


# =============================================================================
# PHASE 2: TRAIN RL AGENT
# =============================================================================

def train_rl_agent(config: Config, anomaly_detector: AnomalyDetector) -> PPO:
    """
    Train the PPO reinforcement learning agent.
    
    The agent learns to:
    1. Recognize when action is needed (high anomaly + low health)
    2. Choose minimal sufficient response (conserve energy)
    3. Survive as long as possible with high health
    """
    print("\n" + "="*70)
    print("PHASE 2: TRAINING RL AGENT (PPO)")
    print("="*70)
    print("\n*** INTELLIGENT SWITCHING MODE ***")
    print("  - Shield cost: 0.1, Recharge: 0.5 (cheap but drains if held)")
    print("  - Wasteful Shield Penalty: -7 when shielding with NO anomaly")
    print("  - Idle Reward: +3 when safe (energy greed active)")
    print("  - Goal: Mostly Do Nothing, Shield only during spikes!")
    
    # Generate training environment data
    print("\n[PHASE 2] Generating environment data...")
    n_samples = 3000  # More samples for RL training variety
    t = np.linspace(0, 30, n_samples)
    
    # Create signal with embedded anomalies
    sensor_data = (
        np.sin(2 * np.pi * 1.0 * t) +
        0.3 * np.sin(2 * np.pi * 2.5 * t) +
        np.random.normal(0, 0.1, n_samples)
    )
    
    # Generate anomaly flags AND reconstruction errors using our trained detector
    _, reconstruction_errors = anomaly_detector.detect_batch(sensor_data)
    
    # Pad errors to match sensor_data length
    full_errors = np.zeros(n_samples)
    full_errors[config.ANOMALY_WINDOW_SIZE-1:config.ANOMALY_WINDOW_SIZE-1+len(reconstruction_errors)] = reconstruction_errors
    
    # Create anomaly flags from errors
    anomaly_flags = full_errors > anomaly_detector.threshold
    
    # Add some deliberate anomalies for training (3% - sparse spikes, larger magnitude)
    spike_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    sensor_data[spike_indices] += np.random.choice([-4, 4], size=len(spike_indices))
    anomaly_flags[spike_indices] = True
    full_errors[spike_indices] = anomaly_detector.threshold * 5  # Very high error for real spikes
    
    print(f"[PHASE 2] Training data: {n_samples} samples, {np.sum(anomaly_flags)} anomalies ({100*np.mean(anomaly_flags):.1f}%)")
    print(f"[PHASE 2] Reconstruction errors: min={full_errors.min():.4f}, max={full_errors.max():.4f}, threshold={anomaly_detector.threshold:.4f}")
    
    # Create environment with reconstruction errors
    env = SpacecraftEnv(max_steps=500)
    env.set_data(sensor_data, anomaly_flags, full_errors, anomaly_detector.threshold)
    
    # Create PPO agent
    print("\n[PHASE 2] Initializing PPO agent...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config.RL_LEARNING_RATE,
        n_steps=config.RL_N_STEPS,
        batch_size=config.RL_BATCH_SIZE,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )
    
    # Train
    print(f"\n[PHASE 2] Training for {config.RL_TIMESTEPS} timesteps...")
    callback = TrainingCallback()
    model.learn(
        total_timesteps=config.RL_TIMESTEPS,
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    if config.SAVE_MODELS:
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        model.save(f"{config.OUTPUT_DIR}/ppo_spacecraft")
        print(f"[PHASE 2] Model saved to {config.OUTPUT_DIR}/ppo_spacecraft")
    
    return model


# =============================================================================
# PHASE 3: DEPLOYMENT SIMULATION
# =============================================================================

def run_deployment_simulation(
    config: Config,
    anomaly_detector: AnomalyDetector,
    rl_agent: PPO
) -> dict:
    """
    Run the full deployment simulation.
    
    This simulates real operation:
    1. Receive sensor data
    2. Fuse signals
    3. Detect anomalies
    4. Get RL agent action
    5. Execute and update state
    """
    print("\n" + "="*70)
    print("PHASE 3: DEPLOYMENT SIMULATION")
    print("="*70)
    
    # Generate test data using our existing data_gen module
    print("\n[PHASE 3] Generating test telemetry data...")
    time_vec, truth, classical, quantum = generate_telemetry(
        duration=config.TEST_DATA_DURATION,
        rate=config.TEST_DATA_RATE
    )
    
    # Apply DSP filtering
    lpf = LowPassFilter(cutoff_freq=10.0, sample_rate=config.TEST_DATA_RATE)
    classical_filtered = lpf.apply(classical)
    quantum_filtered = lpf.apply(quantum)
    
    # Fuse signals
    print("[PHASE 3] Fusing sensor signals...")
    fusion_result = fuse_signals(classical_filtered, quantum_filtered)
    fused_signal = fusion_result.fused_signal
    
    # Detect anomalies and get reconstruction errors
    print("[PHASE 3] Running anomaly detection...")
    anomaly_flags, reconstruction_errors = anomaly_detector.detect_batch(fused_signal)
    
    # Pad errors and flags to match fused_signal length
    full_reconstruction_errors = np.zeros(len(fused_signal))
    full_reconstruction_errors[config.ANOMALY_WINDOW_SIZE-1:config.ANOMALY_WINDOW_SIZE-1+len(reconstruction_errors)] = reconstruction_errors
    
    full_anomaly_flags = np.zeros(len(fused_signal), dtype=bool)
    full_anomaly_flags[config.ANOMALY_WINDOW_SIZE-1:config.ANOMALY_WINDOW_SIZE-1+len(anomaly_flags)] = anomaly_flags
    
    print(f"[PHASE 3] Detected {np.sum(full_anomaly_flags)} anomalous windows")
    print(f"[PHASE 3] Agent can now SEE reconstruction errors (Low=Safe, High=Danger)")
    
    # Create deployment environment WITH reconstruction errors
    env = SpacecraftEnv(max_steps=len(fused_signal))
    env.set_data(fused_signal, full_anomaly_flags, full_reconstruction_errors, anomaly_detector.threshold)
    
    # Run simulation
    print("\n[PHASE 3] Running deployment loop...")
    obs, info = env.reset()
    
    # Recording
    health_history = [config.INITIAL_HEALTH]
    energy_history = [config.INITIAL_ENERGY]
    action_history = []
    anomaly_history = [False]
    reward_history = []
    
    step = 0
    done = False
    
    while not done:
        # Get action from trained agent
        action, _ = rl_agent.predict(obs, deterministic=True)
        action = int(action)
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Record
        health_history.append(info['health'])
        energy_history.append(info['energy'])
        action_history.append(action)
        anomaly_history.append(info['anomaly'])
        reward_history.append(reward)
        
        step += 1
        
        # Progress update
        if step % 200 == 0:
            print(f"[PHASE 3]   Step {step}: Health={info['health']:.1f}%, Action={env.action_names[action]}")
    
    # Final stats
    stats = env.get_episode_stats()
    
    print(f"\n[PHASE 3] Simulation Complete!")
    print(f"[PHASE 3]   Steps survived: {stats['steps_survived']}")
    print(f"[PHASE 3]   Final health: {stats['final_health']:.1f}%")
    print(f"[PHASE 3]   Total reward: {stats['total_reward']:.2f}")
    print(f"[PHASE 3]   Actions: Nothing={stats['actions_taken']['nothing']}, "
          f"Shield={stats['actions_taken']['shield']}, Thruster={stats['actions_taken']['thruster']}")
    
    return {
        'time': time_vec[:len(health_history)],
        'fused_signal': fused_signal[:len(health_history)],
        'health': health_history,
        'energy': energy_history,
        'actions': action_history,
        'anomalies': anomaly_history,
        'rewards': reward_history,
        'reconstruction_errors': reconstruction_errors,
        'stats': stats,
        'classical': classical[:len(health_history)],
        'quantum': quantum[:len(health_history)],
        'truth': truth[:len(health_history)]
    }


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visualization(results: dict, config: Config):
    """Create comprehensive visualization of the AI system performance."""
    print("\n[VIZ] Creating visualization...")
    
    fig = plt.figure(figsize=(16, 14))
    
    # Color scheme
    colors = {
        'health': '#2ecc71',
        'energy': '#3498db', 
        'anomaly': '#e74c3c',
        'action': '#9b59b6',
        'signal': '#34495e',
        'classical': '#f39c12',
        'quantum': '#1abc9c'
    }
    
    # 1. Sensor Signals
    ax1 = fig.add_subplot(4, 2, 1)
    t = results['time'][:len(results['classical'])]
    ax1.plot(t, results['classical'], color=colors['classical'], alpha=0.7, label='Classical', linewidth=0.8)
    ax1.plot(t, results['quantum'], color=colors['quantum'], alpha=0.7, label='Quantum', linewidth=0.8)
    ax1.plot(t, results['truth'], 'k--', alpha=0.5, label='Ground Truth', linewidth=1)
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Raw Sensor Signals', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Fused Signal with Anomalies
    ax2 = fig.add_subplot(4, 2, 2)
    t_fused = results['time'][:len(results['fused_signal'])]
    ax2.plot(t_fused, results['fused_signal'], color=colors['signal'], linewidth=0.8, label='Fused Signal')
    
    # Mark anomalies
    anomaly_indices = np.where(results['anomalies'][:len(t_fused)])[0]
    if len(anomaly_indices) > 0:
        ax2.scatter(t_fused[anomaly_indices], results['fused_signal'][anomaly_indices], 
                   c=colors['anomaly'], s=30, label='Anomaly Detected', zorder=5)
    
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Fused Signal with Anomaly Detection', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 3. Reconstruction Error
    ax3 = fig.add_subplot(4, 2, 3)
    errors = results['reconstruction_errors']
    # Create time vector matching errors length
    t_err = np.linspace(0, config.TEST_DATA_DURATION, len(errors))
    ax3.plot(t_err, errors, color=colors['action'], linewidth=0.8)
    ax3.axhline(y=np.percentile(errors, config.ANOMALY_THRESHOLD_PERCENTILE), 
                color=colors['anomaly'], linestyle='--', label='Threshold')
    ax3.fill_between(t_err, 0, errors, where=errors > np.percentile(errors, config.ANOMALY_THRESHOLD_PERCENTILE),
                     color=colors['anomaly'], alpha=0.3)
    ax3.set_ylabel('Reconstruction Error')
    ax3.set_title('Autoencoder Anomaly Detection', fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 4. System Health Over Time
    ax4 = fig.add_subplot(4, 2, 4)
    t_health = results['time'][:len(results['health'])]
    ax4.fill_between(t_health, 0, results['health'], color=colors['health'], alpha=0.3)
    ax4.plot(t_health, results['health'], color=colors['health'], linewidth=2, label='Health')
    ax4.axhline(y=50, color='orange', linestyle='--', alpha=0.7, label='Warning Level')
    ax4.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Critical Level')
    ax4.set_ylabel('Health (%)')
    ax4.set_ylim(0, 105)
    ax4.set_title('Spacecraft Health Over Time', fontweight='bold')
    ax4.legend(loc='lower left')
    ax4.grid(True, alpha=0.3)
    
    # 5. Energy Over Time
    ax5 = fig.add_subplot(4, 2, 5)
    t_energy = results['time'][:len(results['energy'])]
    ax5.fill_between(t_energy, 0, results['energy'], color=colors['energy'], alpha=0.3)
    ax5.plot(t_energy, results['energy'], color=colors['energy'], linewidth=2)
    ax5.set_ylabel('Energy (%)')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylim(0, 105)
    ax5.set_title('Energy Level Over Time', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. RL Agent Actions
    ax6 = fig.add_subplot(4, 2, 6)
    t_actions = results['time'][:len(results['actions'])]
    actions = np.array(results['actions'])
    
    # Create action regions
    for i, (action_val, color, label) in enumerate([
        (0, '#95a5a6', 'Do Nothing'),
        (1, '#f1c40f', 'Shield'),
        (2, '#e74c3c', 'Thruster')
    ]):
        mask = actions == action_val
        if np.any(mask):
            ax6.scatter(t_actions[mask], actions[mask], c=color, s=20, label=label, alpha=0.7)
    
    ax6.set_yticks([0, 1, 2])
    ax6.set_yticklabels(['Nothing', 'Shield', 'Thruster'])
    ax6.set_ylabel('Action')
    ax6.set_xlabel('Time (s)')
    ax6.set_title('RL Agent Actions Over Time', fontweight='bold')
    ax6.legend(loc='upper right')
    ax6.grid(True, alpha=0.3)
    
    # 7. Cumulative Reward
    ax7 = fig.add_subplot(4, 2, 7)
    cumulative_reward = np.cumsum(results['rewards'])
    t_rewards = results['time'][:len(cumulative_reward)]
    ax7.plot(t_rewards, cumulative_reward, color=colors['action'], linewidth=2)
    ax7.fill_between(t_rewards, 0, cumulative_reward, 
                     where=np.array(cumulative_reward) > 0, color='green', alpha=0.3)
    ax7.fill_between(t_rewards, 0, cumulative_reward,
                     where=np.array(cumulative_reward) < 0, color='red', alpha=0.3)
    ax7.set_ylabel('Cumulative Reward')
    ax7.set_xlabel('Time (s)')
    ax7.set_title('RL Agent Learning Progress', fontweight='bold')
    ax7.grid(True, alpha=0.3)
    
    # 8. Action Distribution Pie Chart
    ax8 = fig.add_subplot(4, 2, 8)
    stats = results['stats']
    action_counts = [
        stats['actions_taken']['nothing'],
        stats['actions_taken']['shield'],
        stats['actions_taken']['thruster']
    ]
    action_labels = ['Do Nothing', 'Shield', 'Thruster']
    action_colors = ['#95a5a6', '#f1c40f', '#e74c3c']
    
    wedges, texts, autotexts = ax8.pie(
        action_counts, 
        labels=action_labels,
        colors=action_colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=(0, 0.05, 0.1)
    )
    ax8.set_title('Action Distribution', fontweight='bold')
    
    # Overall title
    fig.suptitle(
        'Quantum-IoT Hybrid Network: AI Decision System Performance\n'
        f'Final Health: {stats["final_health"]:.1f}% | Steps: {stats["steps_survived"]} | '
        f'Total Reward: {stats["total_reward"]:.1f}',
        fontsize=14, fontweight='bold', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'ai_decision_system_{timestamp}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"[VIZ] Saved visualization to {filename}")
    
    plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("   QUANTUM-IoT HYBRID NETWORK: AI DECISION SYSTEM")
    print("   Spacecraft Nervous System Simulation")
    print("="*70)
    print(f"\nTimestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration
    config = Config()
    
    # Phase 1: Train Anomaly Detector
    anomaly_detector = train_anomaly_detector(config)
    
    # Phase 2: Train RL Agent
    rl_agent = train_rl_agent(config, anomaly_detector)
    
    # Phase 3: Run Deployment Simulation
    results = run_deployment_simulation(config, anomaly_detector, rl_agent)
    
    # Create Visualization
    create_visualization(results, config)
    
    print("\n" + "="*70)
    print("   SIMULATION COMPLETE")
    print("="*70)
    print("\nKey Results:")
    print(f"  - Final Health: {results['stats']['final_health']:.1f}%")
    print(f"  - Steps Survived: {results['stats']['steps_survived']}")
    print(f"  - Total Reward: {results['stats']['total_reward']:.1f}")
    print(f"  - Anomalies Detected: {np.sum(results['anomalies'])}")
    print(f"  - Model saved to: {config.OUTPUT_DIR}/")
    print("\nThe AI system successfully demonstrated:")
    print("  1. Autoencoder-based anomaly detection")
    print("  2. PPO reinforcement learning for action selection")
    print("  3. Real-time decision making under uncertainty")


if __name__ == "__main__":
    main()
