"""
=============================================================================
ANOMALY DETECTION MODULE - PyTorch Autoencoder
=============================================================================

This module implements a neural network-based anomaly detector using an
Autoencoder architecture. The key insight is:

    "A model trained to reconstruct NORMAL data will fail to reconstruct
     ABNORMAL data, producing high reconstruction error."

AUTOENCODER ARCHITECTURE:
-------------------------
                    Encoder                     Decoder
    Input (n) --> [Linear] --> [ReLU] --> [Linear] --> [ReLU] --> [Linear] --> Output (n)
       |            |                        |                        |
    Sensor      Compress to              Expand back            Reconstructed
     Data       latent space             to original              Signal

ANOMALY DETECTION THEORY:
-------------------------
1. TRAINING PHASE: Feed ONLY normal sensor data (clean sine waves)
   - The autoencoder learns to compress and reconstruct normal patterns
   - It essentially learns "what normal looks like"

2. DETECTION PHASE: Feed new data (possibly anomalous)
   - Calculate Mean Squared Error between input and reconstruction
   - High error = input doesn't match learned "normal" pattern = ANOMALY

APPLICATIONS IN SPACECRAFT:
---------------------------
- Radiation spike detection (sudden deviations in quantum sensor)
- Structural stress detection (abnormal vibration patterns)
- Communication delay detection (irregular signal timing)
- Thermal anomaly detection (unexpected temperature spikes)

Author: Spacecraft Nervous System Simulation Team
Course: Quantum-IoT Hybrid Network for Spacecraft Decision Systems
=============================================================================
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class AnomalyResult:
    """Container for anomaly detection results."""
    is_anomaly: bool
    reconstruction_error: float
    threshold: float
    confidence: float  # How confident we are (error/threshold ratio)


class Autoencoder(nn.Module):
    """
    Simple Autoencoder Neural Network for learning normal patterns.
    
    Architecture:
    - Encoder: Compresses input to a smaller latent representation
    - Decoder: Reconstructs input from latent representation
    
    The bottleneck (latent space) forces the network to learn
    the most important features of the data.
    """
    
    def __init__(self, input_dim: int = 10, latent_dim: int = 4):
        """
        Initialize the Autoencoder.
        
        Parameters
        ----------
        input_dim : int
            Size of input window (number of consecutive samples)
        latent_dim : int
            Size of compressed representation (bottleneck)
        """
        super(Autoencoder, self).__init__()
        
        # Encoder: Input -> Hidden -> Latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.ReLU()
        )
        
        # Decoder: Latent -> Hidden -> Output
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
            # No activation - we want raw values for reconstruction
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode then decode."""
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get latent representation (useful for visualization)."""
        return self.encoder(x)


class AnomalyDetector:
    """
    Anomaly Detection System using Autoencoder.
    
    This class wraps the Autoencoder and provides high-level methods
    for training on normal data and detecting anomalies in new data.
    
    Usage:
    ------
    >>> detector = AnomalyDetector(window_size=10)
    >>> detector.train(normal_signal_data)
    >>> result = detector.detect(new_data_point)
    >>> if result.is_anomaly:
    ...     print("Anomaly detected!")
    
    Attributes
    ----------
    window_size : int
        Number of consecutive samples used for detection
    model : Autoencoder
        The neural network model
    threshold : float
        Reconstruction error threshold for anomaly detection
    """
    
    def __init__(
        self,
        window_size: int = 10,
        latent_dim: int = 4,
        threshold_percentile: float = 95.0,
        device: str = 'cpu'
    ):
        """
        Initialize the Anomaly Detector.
        
        Parameters
        ----------
        window_size : int
            Number of samples to consider at once (default: 10)
        latent_dim : int
            Compression size for autoencoder (default: 4)
        threshold_percentile : float
            Percentile of training errors to use as threshold (default: 95)
            Higher = fewer false positives, lower = more sensitive
        device : str
            'cpu' or 'cuda' for GPU acceleration
        """
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.device = torch.device(device)
        
        # Initialize model
        self.model = Autoencoder(input_dim=window_size, latent_dim=latent_dim)
        self.model.to(self.device)
        
        # Threshold will be set during training
        self.threshold = None
        self.is_trained = False
        
        # Store training history for analysis
        self.training_losses = []
        
        print(f"[ANOMALY] Initialized AnomalyDetector")
        print(f"[ANOMALY]   Window size: {window_size}")
        print(f"[ANOMALY]   Latent dim: {latent_dim}")
        print(f"[ANOMALY]   Device: {device}")
    
    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """
        Convert time series to sliding windows.
        
        Example: [1,2,3,4,5] with window_size=3 becomes:
                 [[1,2,3], [2,3,4], [3,4,5]]
        """
        windows = []
        for i in range(len(data) - self.window_size + 1):
            windows.append(data[i:i + self.window_size])
        return np.array(windows)
    
    def train(
        self,
        normal_data: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        verbose: bool = True
    ) -> dict:
        """
        Train the autoencoder on NORMAL data only.
        
        The model learns to reconstruct normal patterns. After training,
        any data that produces high reconstruction error is anomalous.
        
        Parameters
        ----------
        normal_data : np.ndarray
            1D array of normal sensor readings (no anomalies!)
        epochs : int
            Number of training iterations
        batch_size : int
            Samples per gradient update
        learning_rate : float
            Adam optimizer learning rate
        verbose : bool
            Print training progress
        
        Returns
        -------
        history : dict
            Training history with losses
        """
        if verbose:
            print(f"[ANOMALY] Training on {len(normal_data)} normal samples...")
        
        # Create sliding windows from the data
        windows = self._create_windows(normal_data)
        X = torch.FloatTensor(windows).to(self.device)
        
        # Setup training
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        self.model.train()
        self.training_losses = []
        
        # Training loop
        for epoch in range(epochs):
            # Shuffle data each epoch
            indices = torch.randperm(len(X))
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, len(X), batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = X[batch_indices]
                
                # Forward pass
                optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_loss = epoch_loss / n_batches
            self.training_losses.append(avg_loss)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"[ANOMALY]   Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Calculate threshold based on training reconstruction errors
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1).cpu().numpy()
        
        # Set threshold at specified percentile of training errors
        self.threshold = np.percentile(errors, self.threshold_percentile)
        self.is_trained = True
        
        if verbose:
            print(f"[ANOMALY] Training complete!")
            print(f"[ANOMALY]   Final loss: {self.training_losses[-1]:.6f}")
            print(f"[ANOMALY]   Threshold set to: {self.threshold:.6f}")
        
        return {
            'losses': self.training_losses,
            'threshold': self.threshold,
            'final_loss': self.training_losses[-1]
        }
    
    def detect(self, data: np.ndarray) -> AnomalyResult:
        """
        Detect if the given data window is anomalous.
        
        Parameters
        ----------
        data : np.ndarray
            Array of length window_size (or longer, uses last window_size)
        
        Returns
        -------
        AnomalyResult
            Contains is_anomaly, reconstruction_error, threshold, confidence
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        # Use last window_size samples if data is longer
        if len(data) > self.window_size:
            data = data[-self.window_size:]
        elif len(data) < self.window_size:
            raise ValueError(f"Need at least {self.window_size} samples, got {len(data)}")
        
        # Convert to tensor and get reconstruction
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(data).unsqueeze(0).to(self.device)
            reconstructed = self.model(x)
            error = torch.mean((x - reconstructed) ** 2).item()
        
        # Determine if anomaly
        is_anomaly = error > self.threshold
        confidence = min(error / self.threshold, 10.0)  # Cap at 10x
        
        return AnomalyResult(
            is_anomaly=is_anomaly,
            reconstruction_error=error,
            threshold=self.threshold,
            confidence=confidence
        )
    
    def detect_batch(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies in a full time series.
        
        Parameters
        ----------
        data : np.ndarray
            Full time series to analyze
        
        Returns
        -------
        anomaly_flags : np.ndarray
            Boolean array indicating anomalies
        errors : np.ndarray
            Reconstruction errors for each window
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained! Call train() first.")
        
        windows = self._create_windows(data)
        X = torch.FloatTensor(windows).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(X)
            errors = torch.mean((X - reconstructed) ** 2, dim=1).cpu().numpy()
        
        anomaly_flags = errors > self.threshold
        
        return anomaly_flags, errors
    
    def save(self, path: str):
        """Save model and threshold to file."""
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold,
            'window_size': self.window_size,
            'latent_dim': self.latent_dim
        }, path)
        print(f"[ANOMALY] Model saved to {path}")
    
    def load(self, path: str):
        """Load model and threshold from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.threshold = checkpoint['threshold']
        self.is_trained = True
        print(f"[ANOMALY] Model loaded from {path}")


# =============================================================================
# MODULE TEST
# =============================================================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("ANOMALY DETECTOR TEST")
    print("="*60)
    
    # Generate normal data (sine wave)
    t_normal = np.linspace(0, 10, 1000)
    normal_data = np.sin(2 * np.pi * 1 * t_normal) + np.random.normal(0, 0.05, 1000)
    
    # Generate test data with anomalies
    t_test = np.linspace(0, 5, 500)
    test_data = np.sin(2 * np.pi * 1 * t_test) + np.random.normal(0, 0.05, 500)
    # Inject anomalies (radiation spikes)
    test_data[100] = 5.0  # Spike
    test_data[200] = -4.0  # Spike
    test_data[300:310] = 3.0  # Sustained anomaly
    
    # Create and train detector
    detector = AnomalyDetector(window_size=20, latent_dim=4)
    history = detector.train(normal_data, epochs=50, verbose=True)
    
    # Detect anomalies
    flags, errors = detector.detect_batch(test_data)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Training loss
    axes[0].plot(history['losses'], 'b-')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].grid(True)
    
    # Test data with anomalies highlighted
    axes[1].plot(t_test[:-19], test_data[:-19], 'b-', alpha=0.7, label='Signal')
    anomaly_indices = np.where(flags)[0]
    if len(anomaly_indices) > 0:
        axes[1].scatter(t_test[anomaly_indices], test_data[anomaly_indices], 
                       c='red', s=50, label='Anomaly Detected')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Anomaly Detection Results')
    axes[1].legend()
    axes[1].grid(True)
    
    # Reconstruction error
    axes[2].plot(t_test[:-19], errors, 'g-')
    axes[2].axhline(y=detector.threshold, color='r', linestyle='--', label='Threshold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Reconstruction Error')
    axes[2].set_title('Reconstruction Error Over Time')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('anomaly_detection_test.png', dpi=150)
    plt.show()
    
    print(f"\nDetected {np.sum(flags)} anomalous windows out of {len(flags)}")

