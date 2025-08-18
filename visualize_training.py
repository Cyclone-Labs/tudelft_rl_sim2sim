#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

def plot_training_progress(log_dir="logs/general_session/general_model_0"):
    """Plot training progress from TensorBoard logs"""
    
    # Load the event accumulator
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get available tags
    tags = event_acc.Tags()['scalars']
    print("Available metrics:", tags)
    
    # Plot training metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Progress - General Domain Randomization Model', fontsize=16)
    
    # Episode reward mean
    if 'rollout/ep_rew_mean' in tags:
        data = event_acc.Scalars('rollout/ep_rew_mean')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        axes[0,0].plot(steps, values)
        axes[0,0].set_title('Episode Reward Mean')
        axes[0,0].set_xlabel('Training Steps')
        axes[0,0].set_ylabel('Reward')
        axes[0,0].grid(True)
    
    # Episode length mean
    if 'rollout/ep_len_mean' in tags:
        data = event_acc.Scalars('rollout/ep_len_mean')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        axes[0,1].plot(steps, values)
        axes[0,1].set_title('Episode Length Mean')
        axes[0,1].set_xlabel('Training Steps')
        axes[0,1].set_ylabel('Steps')
        axes[0,1].grid(True)
    
    # Policy loss
    if 'train/policy_loss' in tags:
        data = event_acc.Scalars('train/policy_loss')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        axes[1,0].plot(steps, values)
        axes[1,0].set_title('Policy Loss')
        axes[1,0].set_xlabel('Training Steps')
        axes[1,0].set_ylabel('Loss')
        axes[1,0].grid(True)
    
    # Value loss
    if 'train/value_loss' in tags:
        data = event_acc.Scalars('train/value_loss')
        steps = [x.step for x in data]
        values = [x.value for x in data]
        axes[1,1].plot(steps, values)
        axes[1,1].set_title('Value Loss')
        axes[1,1].set_xlabel('Training Steps')
        axes[1,1].set_ylabel('Loss')
        axes[1,1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Training progress plot saved as 'training_progress.png'")

def get_final_performance():
    """Get the final model performance"""
    log_dir = "logs/general_session/general_model_0"
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    # Get final episode reward
    if 'rollout/ep_rew_mean' in event_acc.Tags()['scalars']:
        data = event_acc.Scalars('rollout/ep_rew_mean')
        final_reward = data[-1].value
        final_step = data[-1].step
        print(f"Final episode reward: {final_reward:.2f} at step {final_step}")
        
        # Compare to paper results (Table III)
        print("\nComparison to paper results:")
        print("Paper general model: 64.67 episode reward (3-inch sim), 76.78 (5-inch sim)")
        print(f"Your model: {final_reward:.2f} episode reward")

if __name__ == "__main__":
    print("Visualizing training progress...")
    plot_training_progress()
    get_final_performance()
