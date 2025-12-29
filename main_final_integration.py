"""
=============================================================================
FINAL DEMO: QUANTUM-IOT AI INTEGRATION
=============================================================================
This script performs the final integration:
1. Loads REAL sensor data from the Hardware Team (Varenya/Vanshika).
2. Loads the TRAINED AI MODELS (Autoencoder + PPO Agent).
3. Deploys the AI to make decisions on the Real Data.

Usage: python main_final_integration.py
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import PPO
import os
import warnings

# Import your modules
from data_loader import load_varenya_data, SensorData  # The Real Data
from dsp import LowPassFilter
from fusion import fuse_signals
from anomaly import AnomalyDetector
from rl_env import SpacecraftEnv  # To re-use the physics logic

warnings.filterwarnings('ignore')

def run_integration_demo():
    print("\n" + "="*70)
    print("      FINAL INTEGRATION: AI BRAIN + HARDWARE DATA")
    print("="*70)

    # ---------------------------------------------------------
    # 1. LOAD REAL DATA (The Body)
    # ---------------------------------------------------------
    print("\n[1/4] Loading Real Hardware Data (Varenya)...")
    try:
        # Tries to load the most complete stage available
        data = load_varenya_data(use_stage='adc') 
        print(f"      Loaded {len(data.time)} samples (Duration: {data.duration:.1f}s)")
    except Exception as e:
        print(f"[ERROR] Could not load Varenya's data: {e}")
        print("Using Mock data for testing...")
        # Fallback to mock if file missing
        t = np.linspace(0, 10, 1000)
        data = SensorData(t, np.sin(t), np.sin(t), np.sin(t), "MOCK")

    # ---------------------------------------------------------
    # 2. PRE-PROCESS DATA (DSP + Fusion)
    # ---------------------------------------------------------
    print("\n[2/4] Running Signal Processing & Fusion...")
    
    # Low Pass Filter
    lpf = LowPassFilter(cutoff_freq=50.0, sample_rate=data.sample_rate)
    classical_filtered = lpf.apply(data.classical)
    quantum_filtered = lpf.apply(data.quantum)

    # Fusion
    fusion_result = fuse_signals(classical_filtered, quantum_filtered)
    fused_signal = fusion_result.fused_signal
    print("      Fusion Complete.")

    # ---------------------------------------------------------
    # 3. LOAD AI MODELS (The Brain)
    # ---------------------------------------------------------
    print("\n[3/4] Loading Trained AI Models...")
    
    # Load Anomaly Detector
    try:
        detector = AnomalyDetector()
        detector.load("ai_models/anomaly_detector.pt")
        print("      Anomaly Detector Loaded (PyTorch)")
    except:
        print("[ERROR] Anomaly Model not found! Run main_ai.py first.")
        return

    # Load RL Agent
    try:
        rl_model = PPO.load("ai_models/ppo_spacecraft")
        print("      RL Agent Loaded (PPO)")
    except:
        print("[ERROR] RL Agent not found! Run main_ai.py first.")
        return

    # ---------------------------------------------------------
    # 4. EXECUTE DEPLOYMENT LOOP
    # ---------------------------------------------------------
    print("\n[4/4] Running AI Deployment Loop on Real Data...")
    
    # A. Detect Anomalies using Autoencoder
    # Note: We need to normalize/reshape data for the autoencoder if trained on 0-1 range
    # For now, we assume direct input
    anomaly_flags, reconstruction_errors = detector.detect_batch(fused_signal)
    
    # Pad errors to match length
    full_errors = np.zeros(len(fused_signal))
    # Handle window size offset
    window_size = detector.window_size
    if len(reconstruction_errors) > 0:
        full_errors[window_size-1 : window_size-1+len(reconstruction_errors)] = reconstruction_errors

    # B. Initialize Simulation State
    current_health = 100.0
    current_energy = 100.0
    
    # Physics Constants (Must match Training!)
    SHIELD_COST = 0.2
    RECHARGE_RATE = 0.1
    RAD_DAMAGE = 5.0

    # History for plotting
    history = {
        'health': [], 'energy': [], 'action': [], 'reward': []
    }

    print("      Processing frame by frame...")
    
    # Loop through every time step
    for i in range(len(fused_signal)):
        
        # 1. Create Observation State: [Signal, Health, Energy, Error]
        # Normalize inputs roughly to match training scale
        obs = np.array([
            fused_signal[i], 
            current_health, 
            current_energy, 
            full_errors[i]
        ], dtype=np.float32)

        # 2. AI Decides Action
        action, _ = rl_model.predict(obs, deterministic=True)
        action = int(action)

        # 3. Apply Physics (Simulation of the Ship)
        # Check for danger
        is_danger = full_errors[i] > detector.threshold
        
        # Energy Logic
        if action == 1: # Shield
            current_energy -= SHIELD_COST
        elif action == 2: # Thruster
            current_energy -= (SHIELD_COST * 2) # Thruster expensive
        else: # Idle
            current_energy += RECHARGE_RATE
        
        # Clamp Energy
        current_energy = np.clip(current_energy, 0, 100)

        # Health Logic
        if is_danger:
            if action == 1: # Shield
                pass # Safe
            elif action == 2: # Thruster
                pass # Safe
            else: # Idle
                current_health -= RAD_DAMAGE # Ouch
        
        current_health = np.clip(current_health, 0, 100)

        # Record History
        history['health'].append(current_health)
        history['energy'].append(current_energy)
        history['action'].append(action)

    print("      Simulation Complete.")

    # ---------------------------------------------------------
    # 5. VISUALIZATION
    # ---------------------------------------------------------
    plot_final_dashboard(data.time, fused_signal, full_errors, history, detector.threshold)

def plot_final_dashboard(t, signal, errors, history, threshold):
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(4, 1)

    # Plot 1: The Signal & The AI's Vision
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, signal, color='cyan', alpha=0.6, label='Real Fused Signal')
    # Highlight danger zones
    danger_mask = errors > threshold
    ax1.fill_between(t, np.min(signal), np.max(signal), where=danger_mask, color='red', alpha=0.2, label='AI Detected Anomaly')
    ax1.set_title("Input: Real Hardware Data (Fused)", fontsize=14, color='white')
    ax1.legend(loc='upper right')

    # Plot 2: The AI's Understanding (Reconstruction Error)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, errors, color='magenta', label='Reconstruction Error')
    ax2.axhline(threshold, color='red', linestyle='--', label='Danger Threshold')
    ax2.set_title("Processing: Autoencoder 'Vision'", fontsize=14, color='white')
    ax2.legend(loc='upper right')

    # Plot 3: Spacecraft State (Health & Energy)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(t, history['health'], color='#2ecc71', label='Health', linewidth=2)
    ax3.plot(t, history['energy'], color='#3498db', label='Energy', linewidth=2)
    ax3.set_ylim(-5, 105)
    ax3.set_title("Outcome: System Status", fontsize=14, color='white')
    ax3.legend(loc='lower left')

    # Plot 4: AI Choices
    ax4 = fig.add_subplot(gs[3])
    actions = np.array(history['action'])
    # Scatter plot for actions
    ax4.scatter(t[actions==0], actions[actions==0], c='gray', s=10, label='Idle', alpha=0.5)
    ax4.scatter(t[actions==1], actions[actions==1], c='yellow', s=30, label='Shield', marker='s')
    ax4.scatter(t[actions==2], actions[actions==2], c='red', s=50, label='Thruster', marker='^')
    
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels(['Idle', 'Shield', 'Thruster'])
    ax4.set_title("Output: Autonomous Decisions", fontsize=14, color='white')
    ax4.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig("FINAL_INTEGRATED_DEMO.png")
    print(f"\n[SUCCESS] Dashboard saved to FINAL_INTEGRATED_DEMO.png")
    plt.show()

if __name__ == "__main__":
    run_integration_demo()