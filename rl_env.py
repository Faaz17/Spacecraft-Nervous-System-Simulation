import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SpacecraftEnv(gym.Env):
    """
    Custom Environment for Spacecraft Decision Making.
    Scenario: Balance Energy Conservation vs. Radiation Protection.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, max_steps=1000, initial_health=100.0, initial_energy=100.0, energy_regen_rate=0.05):
        super(SpacecraftEnv, self).__init__()
        
        # Parameters
        self.max_steps = max_steps
        self.initial_health = initial_health
        self.initial_energy = initial_energy
        
        # Strict Instructor Physics (scarcity but survivable)
        self.passive_recharge = 0.5   # Fast enough to recover
        self.SHIELD_COST = 0.1        # Cheap to use
        self.THRUSTER_COST = 1.0      # Leave as-is
        self.RAD_DAMAGE = 5.0         # Must shield during spikes
        
        # Action Space: 0=Nothing, 1=Shield, 2=Thruster
        self.action_space = spaces.Discrete(3)
        self.action_names = {0: "Do Nothing", 1: "Shield", 2: "Thruster"}
        
        # Observation Space
        self.observation_space = spaces.Box(
            low=np.array([-5.0, 0.0, 0.0, 0.0]), 
            high=np.array([5.0, 100.0, 100.0, 1.0]), 
            dtype=np.float32
        )
        
        # Placeholders
        self.signal_data = np.zeros(max_steps)
        self.error_data = np.zeros(max_steps)
        self.threshold = 0.1
        
        self.reset()

    def set_data(self, fused_signal, anomaly_flags, reconstruction_errors, anomaly_threshold):
        self.signal_data = fused_signal
        self.error_data = reconstruction_errors
        self.threshold = anomaly_threshold
        self.max_steps = len(fused_signal)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.health = self.initial_health
        self.energy = self.initial_energy
        # Initialize history
        self.history = {'health': [], 'energy': [], 'actions': [], 'rewards': []}
        return self._get_obs(), {}

    def _get_obs(self):
        if len(self.signal_data) == 0:
            return np.array([0.0, 100.0, 100.0, 0.0], dtype=np.float32)

        idx = min(self.current_step, len(self.signal_data) - 1)
        return np.array([
            self.signal_data[idx], 
            self.health, 
            self.energy, 
            self.error_data[idx]
        ], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        
        # --- 1. RECORD ACTION (Prevents NaN Crash) ---
        self.history['actions'].append(int(action))
        
        # --- 2. PHYSICS: ENERGY ---
        if action == 1: # Shield
            self.energy -= self.SHIELD_COST
        elif action == 2: # Thruster
            self.energy -= self.THRUSTER_COST
        else: # Idle
            self.energy += self.passive_recharge
            
        self.energy = np.clip(self.energy, 0, 100)

        # --- 3. PHYSICS: HEALTH & REWARD ---
        idx = min(self.current_step, len(self.error_data) - 1)
        # Use the actual threshold - don't make it too sensitive
        is_anomaly = self.error_data[idx] > self.threshold
        
        # ENERGY GREED: steady reward for keeping battery high
        reward += 0.05 * (self.energy / 100.0)

        # STRICT INSTRUCTOR RULES
        # Wasteful Shield Penalty (critical): if shielding when safe, big penalty
        # Good Idle Reward: strong reward when idling safely

        if is_anomaly:
            # DANGER ZONE – must respond
            if action == 1:
                reward += 1.5   # Reward for shielding
            elif action == 2:
                reward += 0.5   # Thruster acceptable
            else:
                self.health -= self.RAD_DAMAGE
                reward -= 6.0   # Penalty for ignoring anomaly
        else:
            # SAFE ZONE – should relax
            if action == 1:
                reward -= 7.0   # Strong penalty for wasting shield
            elif action == 2:
                reward -= 8.0   # Strong penalty for wasting thruster
            else:
                # Good idle reward when healthy
                if self.health > 80:
                    reward += 0.5
                reward += 3.0  # strong base idle reward to incentivize idling
        
        self.health = np.clip(self.health, 0, 100)
        
        # --- 4. RECORD REWARD (Fixes "Total Reward 0.0") ---
        self.history['rewards'].append(reward)
        
        # --- 5. TERMINATION ---
        # Only health ends the episode; energy can recharge
        terminated = (self.health <= 0)
        self.current_step += 1
        truncated = (self.current_step >= self.max_steps - 1)
        
        info = {'health': self.health, 'energy': self.energy, 'anomaly': is_anomaly}
        
        return self._get_obs(), reward, terminated, truncated, info

    def get_episode_stats(self):
        unique, counts = np.unique(self.history.get('actions', []), return_counts=True)
        action_counts = dict(zip(unique, counts))
        # Sum rewards correctly
        total_rew = sum(self.history.get('rewards', []))
        
        return {
            'steps_survived': self.current_step,
            'final_health': self.health,
            'total_reward': total_rew, 
            'actions_taken': {
                'nothing': action_counts.get(0, 0),
                'shield': action_counts.get(1, 0),
                'thruster': action_counts.get(2, 0)
            }
        }