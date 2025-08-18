"""
QuadController: Portable neural network controller for quadcopter racing

This controller was trained with domain randomization to work across different
quadcopter platforms without retraining.
"""

import numpy as np
import torch
import os
from pathlib import Path
from stable_baselines3 import PPO

class QuadController:
    """
    A portable quadcopter controller trained with domain randomization.
    
    This controller can directly control quadcopters through gym-style interfaces.
    It maps 20-dimensional observations to 4-dimensional motor commands.
    
    Observation space (20 dimensions):
    - Position relative to target gate (3D)
    - Velocity relative to target gate (3D) 
    - Euler angles relative to target gate (3D)
    - Body rates (3D)
    - Motor speeds (4D)
    - Next gate position relative to current gate (3D)
    - Next gate orientation (1D)
    
    Action space (4 dimensions):
    - Motor commands in range [-1, 1] for motors 1-4
    """
    
    def __init__(self, model_path=None, device="auto"):
        """
        Initialize the quadcopter controller.
        
        Args:
            model_path (str, optional): Path to model weights. If None, uses bundled weights.
            device (str): Device to run on ("auto", "cpu", "cuda")
        """
        self.device = self._setup_device(device)
        
        # Load the model
        if model_path is None:
            # Use bundled model weights
            package_dir = Path(__file__).parent
            model_path = package_dir / "100000000.zip"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        self.model = PPO.load(model_path, device=self.device)
        self.obs_dim = 20
        self.action_dim = 4
        
        print(f"QuadController loaded successfully on {self.device}")
        print(f"Model architecture: {self.model.policy}")
        
    def _setup_device(self, device):
        """Setup the computation device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def predict(self, observation, deterministic=True):
        """
        Predict motor commands for the given observation.
        
        Args:
            observation (np.ndarray or list): 20-dimensional observation vector
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            np.ndarray: 4-dimensional motor command vector in range [-1, 1]
        """
        # Convert to numpy array if needed
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation, dtype=np.float32)
        
        # Validate observation shape
        if observation.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observation dimension {self.obs_dim}, got {observation.shape}")
        
        # Handle batch dimension
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        
        # Predict using the trained model
        action, _states = self.model.predict(observation, deterministic=deterministic)
        
        # Return single action if single observation was provided
        if action.shape[0] == 1:
            return action[0]
        return action
    
    def predict_batch(self, observations, deterministic=True):
        """
        Predict motor commands for a batch of observations.
        
        Args:
            observations (np.ndarray): Shape (batch_size, 20)
            deterministic (bool): Whether to use deterministic policy
            
        Returns:
            np.ndarray: Shape (batch_size, 4) motor commands
        """
        observations = np.array(observations, dtype=np.float32)
        if observations.ndim != 2 or observations.shape[1] != self.obs_dim:
            raise ValueError(f"Expected shape (batch_size, {self.obs_dim}), got {observations.shape}")
        
        actions, _states = self.model.predict(observations, deterministic=deterministic)
        return actions
    
    def get_action_bounds(self):
        """
        Get the action space bounds.
        
        Returns:
            tuple: (low_bounds, high_bounds) each as 4-dimensional arrays
        """
        return np.array([-1.0, -1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0, 1.0])
    
    def get_observation_info(self):
        """
        Get information about the expected observation format.
        
        Returns:
            dict: Information about observation dimensions and meaning
        """
        return {
            "total_dim": 20,
            "components": {
                "gate_relative_position": {"indices": [0, 1, 2], "description": "Position relative to target gate (x, y, z)"},
                "gate_relative_velocity": {"indices": [3, 4, 5], "description": "Velocity relative to target gate (vx, vy, vz)"},
                "gate_relative_attitude": {"indices": [6, 7, 8], "description": "Euler angles relative to target gate (roll, pitch, yaw)"},
                "body_rates": {"indices": [9, 10, 11], "description": "Angular velocities in body frame (p, q, r)"},
                "motor_speeds": {"indices": [12, 13, 14, 15], "description": "Current motor speeds (normalized)"},
                "next_gate_position": {"indices": [16, 17, 18], "description": "Next gate position relative to current gate"},
                "next_gate_orientation": {"indices": [19], "description": "Next gate yaw angle"}
            }
        }
    
    def test_controller(self, num_random_tests=5):
        """
        Test the controller with random observations to verify it's working.
        
        Args:
            num_random_tests (int): Number of random tests to run
        """
        print(f"Testing controller with {num_random_tests} random observations...")
        
        for i in range(num_random_tests):
            # Generate random observation
            obs = np.random.randn(20).astype(np.float32)
            
            # Get action
            action = self.predict(obs)
            
            # Check action bounds
            low, high = self.get_action_bounds()
            assert np.all(action >= low) and np.all(action <= high), f"Action out of bounds: {action}"
            
            print(f"Test {i+1}: obs_norm={np.linalg.norm(obs):.2f}, action={action}")
        
        print("✅ All tests passed!")
    
    def __str__(self):
        return f"QuadController(obs_dim={self.obs_dim}, action_dim={self.action_dim}, device={self.device})"
    
    def __repr__(self):
        return self.__str__()
