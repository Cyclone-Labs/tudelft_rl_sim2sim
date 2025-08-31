#!/usr/bin/env python3
"""
Test the trained QuadController on Drake's quadrotor environment

This script adapts the observation space from Drake to work with our trained controller.
"""

import numpy as np
import matplotlib.pyplot as plt
from drake_quadrotor_env import QuadrotorEnv
from quad_controller_package.quad_controller import QuadController
import time

def drake_to_controller_observation(drake_state, target_pos=np.array([0, 0, 1]), next_gate_pos=np.array([1, 0, 1])):
    """
    Convert Drake quadrotor state to our controller's expected observation format.
    
    Drake state (12D): [x, y, z, roll, pitch, yaw, vx, vy, vz, wx, wy, wz]
    Controller obs (20D): see QuadController.get_observation_info()
    
    Args:
        drake_state: 12-dimensional state from Drake
        target_pos: Current target position (gate center)
        next_gate_pos: Next gate position
    
    Returns:
        20-dimensional observation for our controller
    """
    
    # Extract Drake state components
    pos = drake_state[:3]          # x, y, z
    euler = drake_state[3:6]       # roll, pitch, yaw  
    vel = drake_state[6:9]         # vx, vy, vz
    rates = drake_state[9:12]      # wx, wy, wz
    
    # Our controller expects observations relative to target gate
    # For simplicity, we'll create a basic mapping
    
    # Position relative to target gate
    gate_rel_pos = pos - target_pos
    
    # Velocity relative to target gate (assuming static gates)
    gate_rel_vel = vel
    
    # Attitude relative to target gate (simplified - just use current attitude)
    gate_rel_att = euler
    
    # Body rates
    body_rates = rates
    
    # Mock motor speeds (normalized between 0.5-1.0 for reasonable values)
    motor_speeds = np.array([0.7, 0.7, 0.7, 0.7])
    
    # Next gate position relative to current gate
    next_gate_rel = next_gate_pos - target_pos
    
    # Next gate orientation (simplified)
    next_gate_yaw = 0.0
    
    # Construct the 20D observation
    observation = np.concatenate([
        gate_rel_pos,      # [0:3]   - Position relative to gate
        gate_rel_vel,      # [3:6]   - Velocity relative to gate  
        gate_rel_att,      # [6:9]   - Attitude relative to gate
        body_rates,        # [9:12]  - Body rates
        motor_speeds,      # [12:16] - Motor speeds
        next_gate_rel,     # [16:19] - Next gate position
        [next_gate_yaw]    # [19]    - Next gate yaw
    ])
    
    return observation.astype(np.float32)

def run_controller_test():
    """Test the controller on Drake environment"""
    
    print("Testing QuadController on Drake environment...")
    
    # Initialize controller
    controller = QuadController()
    print(f"Controller loaded: {controller}")
    
    # Create Drake environment with meshcat visualization
    from pydrake.all import Meshcat
    meshcat = Meshcat()
    
    print(f"🌐 Meshcat visualization available at: {meshcat.web_url()}")
    print("   Open this URL in your browser to see the 3D visualization")
    print("   You'll be prompted to press Enter after opening the visualization")
    
    # Note: Racing controller was trained at z=-1.5m, but let's try hovering first
    target_position = np.array([0, 0, 1])  # Hover at 1m height
    target_velocity = np.zeros(3)
    env = QuadrotorEnv.build(meshcat=meshcat, 
                            target_position=target_position,
                            target_velocity=target_velocity,
                            realtime_rate=0.1,  # Very slow for better visualization
                            drone_type="5inch")  # Use 5-inch racing quad parameters
    
    print("Drake environment created with visualization")
    
    # Test parameters
    num_episodes = 3
    max_steps_per_episode = 1000
    dt = 0.01
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Reset environment
        obs = env.reset()
        # Handle different return formats from reset
        if isinstance(obs, tuple):
            drake_state = obs[0]
        else:
            drake_state = obs
        
        print(f"  Reset state shape: {np.array(drake_state).shape}, type: {type(drake_state)}")
        print(f"  Reset state: {drake_state}")
        
        episode_reward = 0
        episode_length = 0
        states_history = []
        actions_history = []
        
        for step in range(max_steps_per_episode):
            # Convert Drake state to controller observation
            controller_obs = drake_to_controller_observation(
                drake_state, 
                target_pos=target_position,
                next_gate_pos=np.array([1, 0, 1])  # Mock next gate
            )
            
            # Get action from trained controller
            action = controller.predict(controller_obs, deterministic=True)
            
            # Our controller outputs [-1, 1]^4, convert to propeller forces
            # Drake's plant expects propeller forces, not normalized commands
            
            # Convert controller action [-1,1] to force commands
            # Use hovering force as baseline (mg/4) and add control around it
            hover_force = env.plant.m() * env.plant.g() / 4  # Force per motor to hover
            max_additional_force = hover_force * 1.5  # 150% additional thrust capability
            
            # Convert [-1,1] to [0, 2*hover_force]
            normalized_action = (action + 1.0) / 2.0  # Convert to [0,1]
            normalized_action = np.clip(normalized_action, 0.0, 1.0)
            
            # Scale to force range
            drake_action = normalized_action * (hover_force + max_additional_force)
            
            # Step environment
            try:
                step_result = env.step(drake_action)
                if len(step_result) == 4:
                    next_obs, reward, done, info = step_result
                else:
                    # Handle different return format
                    next_obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                drake_state = next_obs
                
                episode_reward += reward
                episode_length += 1
                
                # Store for analysis
                states_history.append(drake_state.copy())
                actions_history.append(action.copy())
                
                # Print progress occasionally
                if step % 100 == 0:
                    pos = drake_state[:3]
                    vel_norm = np.linalg.norm(drake_state[6:9])
                    print(f"  Step {step:3d}: pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                          f"vel={vel_norm:.2f}, reward={reward:.3f}")
                
                if done:
                    print(f"  Episode finished early at step {step}")
                    break
                    
            except Exception as e:
                print(f"  Error at step {step}: {e}")
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # Episode summary
        final_pos = drake_state[:3]
        distance_to_target = np.linalg.norm(final_pos - target_position)
        
        print(f"  Episode {episode + 1} Results:")
        print(f"    Total reward: {episode_reward:.2f}")
        print(f"    Episode length: {episode_length}")
        print(f"    Final position: [{final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f}]")
        print(f"    Distance to target: {distance_to_target:.2f}m")
        print(f"    Target was: [{target_position[0]:.2f}, {target_position[1]:.2f}, {target_position[2]:.2f}]")
    
    # Overall results
    print(f"\n=== Overall Results ===")
    print(f"Episodes completed: {len(episode_rewards)}")
    print(f"Average reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    
    return episode_rewards, episode_lengths

def test_observation_conversion():
    """Test the observation conversion function"""
    print("\n=== Testing Observation Conversion ===")
    
    # Create a sample Drake state
    drake_state = np.array([
        0.1, 0.2, 0.9,     # position
        0.05, -0.03, 0.1,  # euler angles
        0.2, 0.1, 0.05,    # velocity
        0.01, -0.02, 0.03  # angular rates
    ])
    
    print(f"Drake state (12D): {drake_state}")
    
    # Convert to controller observation
    controller_obs = drake_to_controller_observation(drake_state)
    
    print(f"Controller obs (20D): {controller_obs}")
    print(f"Observation shape: {controller_obs.shape}")
    
    # Test with controller
    controller = QuadController()
    action = controller.predict(controller_obs)
    print(f"Controller action: {action}")
    
    # Verify action bounds
    low, high = controller.get_action_bounds()
    in_bounds = np.all(action >= low) and np.all(action <= high)
    print(f"Action in bounds: {in_bounds}")

if __name__ == "__main__":
    print("QuadController + Drake Environment Test")
    print("=" * 50)
    
    try:
        # Test observation conversion first
        test_observation_conversion()
        
        # Run the main test
        episode_rewards, episode_lengths = run_controller_test()
        
        print("\n🎉 Test completed successfully!")
        print("\nNote: This is a basic integration test. The controller was trained")
        print("on a different environment, so performance may not be optimal.")
        print("For best results, the controller should be retrained on Drake or")
        print("the observation mapping should be improved.")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure Drake is installed: pip install drake")
        print("And that the quad_controller package is available")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
