#!/usr/bin/env python3
"""
Example usage of the QuadController package

This script demonstrates how to use the controller in different scenarios.
"""

import numpy as np
import time
from quad_controller import QuadController

def basic_usage_example():
    """Basic usage example"""
    print("=== Basic Usage Example ===")
    
    # Initialize controller
    controller = QuadController()
    print(f"Controller: {controller}")
    
    # Single prediction
    obs = np.random.randn(20).astype(np.float32)
    action = controller.predict(obs)
    print(f"Single prediction: obs_norm={np.linalg.norm(obs):.2f}, action={action}")
    
    # Batch prediction
    batch_obs = np.random.randn(5, 20).astype(np.float32)
    batch_actions = controller.predict_batch(batch_obs)
    print(f"Batch prediction: shape={batch_actions.shape}")
    
    print()

def controller_info_example():
    """Show controller information"""
    print("=== Controller Information ===")
    
    controller = QuadController()
    
    # Action bounds
    low, high = controller.get_action_bounds()
    print(f"Action bounds: [{low}] to [{high}]")
    
    # Observation info
    obs_info = controller.get_observation_info()
    print(f"Observation dimension: {obs_info['total_dim']}")
    print("Observation components:")
    for name, info in obs_info['components'].items():
        indices = info['indices']
        desc = info['description']
        print(f"  {name:20} [{indices[0]:2d}-{indices[-1]:2d}]: {desc}")
    
    print()

def simulation_loop_example():
    """Example of using controller in a simulation loop"""
    print("=== Simulation Loop Example ===")
    
    controller = QuadController()
    
    # Simulate 10 episodes
    for episode in range(3):
        print(f"Episode {episode + 1}:")
        
        # Reset environment (simulated)
        obs = np.random.randn(20).astype(np.float32)
        total_reward = 0
        
        # Run episode for 100 steps
        for step in range(10):
            # Get action from controller
            action = controller.predict(obs, deterministic=True)
            
            # Simulate environment step (dummy)
            obs = np.random.randn(20).astype(np.float32)  # Next observation
            reward = np.random.randn()  # Dummy reward
            done = step >= 9  # Episode ends after 10 steps
            
            total_reward += reward
            
            if step % 5 == 0:
                print(f"  Step {step:2d}: action_norm={np.linalg.norm(action):.2f}, reward={reward:.2f}")
            
            if done:
                break
        
        print(f"  Episode total reward: {total_reward:.2f}")
    
    print()

def performance_test():
    """Test controller performance"""
    print("=== Performance Test ===")
    
    controller = QuadController()
    
    # Test single predictions
    num_single = 1000
    start_time = time.time()
    for _ in range(num_single):
        obs = np.random.randn(20).astype(np.float32)
        action = controller.predict(obs)
    single_time = time.time() - start_time
    
    print(f"Single predictions: {num_single} calls in {single_time:.3f}s ({num_single/single_time:.1f} Hz)")
    
    # Test batch predictions
    batch_sizes = [1, 10, 100]
    for batch_size in batch_sizes:
        obs_batch = np.random.randn(batch_size, 20).astype(np.float32)
        start_time = time.time()
        for _ in range(100):
            actions = controller.predict_batch(obs_batch)
        batch_time = time.time() - start_time
        total_predictions = 100 * batch_size
        
        print(f"Batch predictions (size {batch_size:3d}): {total_predictions} predictions in {batch_time:.3f}s ({total_predictions/batch_time:.1f} Hz)")
    
    print()

def integration_example():
    """Example of how to integrate with a custom gym environment"""
    print("=== Integration Example ===")
    
    print("""
# Example integration with a custom gym environment:

import gym
from quad_controller import QuadController

# Your custom environment
env = gym.make('YourQuadEnvironment-v0')
controller = QuadController()

obs = env.reset()
done = False
total_reward = 0

while not done:
    # Get action from trained controller
    action = controller.predict(obs, deterministic=True)
    
    # Step environment
    obs, reward, done, info = env.step(action)
    total_reward += reward
    
    # Optional: Add your own logic here
    if info.get('collision', False):
        print("Collision detected!")
        break

print(f"Episode finished with reward: {total_reward}")
    """)

if __name__ == "__main__":
    print("QuadController Package Example Usage\n")
    
    try:
        basic_usage_example()
        controller_info_example()
        simulation_loop_example()
        performance_test()
        integration_example()
        
        # Run built-in tests
        print("=== Built-in Tests ===")
        controller = QuadController()
        controller.test_controller(num_random_tests=5)
        
        print("\n🎉 All examples completed successfully!")
        print("\nYou can now use this controller in your own simulations!")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()
