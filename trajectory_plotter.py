"""
Trajectory plotting utilities for quadrotor flight analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

class TrajectoryPlotter:
    """Class for plotting quadrotor trajectories with gates and analysis."""
    
    def __init__(self, gate_pos=None, gate_yaw=None):
        """
        Initialize plotter with optional gate information.
        
        Args:
            gate_pos: Array of gate positions (N, 3)
            gate_yaw: Array of gate yaw angles (N,)
        """
        self.gate_pos = gate_pos
        self.gate_yaw = gate_yaw
        self.trajectories = []
        
    def add_trajectory(self, x, y, z, time=None, label="Trajectory", color='blue'):
        """Add a trajectory to plot."""
        if time is None:
            time = np.arange(len(x)) * 0.01  # Assume 100Hz
        
        self.trajectories.append({
            'x': np.array(x),
            'y': np.array(y), 
            'z': np.array(z),
            'time': np.array(time),
            'label': label,
            'color': color
        })
    
    def plot_xy_trajectory(self, ax=None, show_gates=True):
        """Plot XY trajectory with optional gates."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Plot trajectories
        for traj in self.trajectories:
            ax.plot(traj['x'], traj['y'], color=traj['color'], 
                   linewidth=2, label=traj['label'], alpha=0.8)
            # Mark start and end
            ax.plot(traj['x'][0], traj['y'][0], 'o', color=traj['color'], 
                   markersize=8, label=f"{traj['label']} Start")
            ax.plot(traj['x'][-1], traj['y'][-1], 's', color=traj['color'], 
                   markersize=8, label=f"{traj['label']} End")
        
        # Plot gates if available
        if show_gates and self.gate_pos is not None and self.gate_yaw is not None:
            for i, (pos, yaw) in enumerate(zip(self.gate_pos, self.gate_yaw)):
                x, y = pos[0], pos[1]
                gate_width = 0.75
                dx = np.sin(yaw) * gate_width
                dy = np.cos(yaw) * gate_width
                ax.plot([x-dx, x+dx], [y-dy, y+dy], 'k-', linewidth=4, alpha=0.7)
                ax.text(x+0.15, y+0.15, f'G{i}', ha='center', va='center', 
                       fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.6))
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Flight Trajectory (Top View)')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        return ax
    
    def plot_altitude_profile(self, ax=None):
        """Plot altitude vs time."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        
        for traj in self.trajectories:
            ax.plot(traj['time'], traj['z'], color=traj['color'], 
                   linewidth=2, label=traj['label'])
        
        # Mark gate altitude if available
        if self.gate_pos is not None:
            gate_alt = self.gate_pos[0, 2]  # Assume all gates at same altitude
            ax.axhline(y=gate_alt, color='red', linestyle='--', 
                      label=f'Gate Altitude ({gate_alt:.1f}m)', alpha=0.7)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Z Position (m)')
        ax.set_title('Altitude Profile')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return ax
    
    def plot_3d_trajectory(self, ax=None):
        """Plot 3D trajectory."""
        if ax is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        for traj in self.trajectories:
            ax.plot(traj['x'], traj['y'], traj['z'], color=traj['color'], 
                   linewidth=2, label=traj['label'])
            # Mark start and end
            ax.scatter(traj['x'][0], traj['y'][0], traj['z'][0], 
                      color=traj['color'], s=100, marker='o', label=f"{traj['label']} Start")
            ax.scatter(traj['x'][-1], traj['y'][-1], traj['z'][-1], 
                      color=traj['color'], s=100, marker='s', label=f"{traj['label']} End")
        
        # Plot gates in 3D
        if self.gate_pos is not None and self.gate_yaw is not None:
            for i, (pos, yaw) in enumerate(zip(self.gate_pos, self.gate_yaw)):
                x, y, z = pos
                gate_width = 0.75
                gate_height = 1.5
                
                # Gate corners
                dx = np.sin(yaw) * gate_width
                dy = np.cos(yaw) * gate_width
                
                # Draw gate frame
                corners_x = [x-dx, x+dx, x+dx, x-dx, x-dx]
                corners_y = [y-dy, y+dy, y+dy, y-dy, y-dy]
                corners_z = [z-gate_height/2, z-gate_height/2, z+gate_height/2, z+gate_height/2, z-gate_height/2]
                
                ax.plot(corners_x, corners_y, corners_z, 'k-', linewidth=3, alpha=0.7)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_zlabel('Z Position (m)')
        ax.set_title('3D Flight Trajectory')
        ax.legend()
        
        return ax
    
    def plot_comprehensive_analysis(self, figsize=(15, 10), save_path=None):
        """Create a comprehensive trajectory analysis plot."""
        fig = plt.figure(figsize=figsize)
        
        # XY trajectory
        ax1 = plt.subplot(2, 3, (1, 4))
        self.plot_xy_trajectory(ax1)
        
        # Altitude profile  
        ax2 = plt.subplot(2, 3, 2)
        self.plot_altitude_profile(ax2)
        
        # 3D trajectory
        ax3 = plt.subplot(2, 3, 3, projection='3d')
        self.plot_3d_trajectory(ax3)
        
        # Velocity profile
        ax4 = plt.subplot(2, 3, 5)
        for traj in self.trajectories:
            # Calculate velocity magnitude
            vx = np.gradient(traj['x']) / np.gradient(traj['time'])
            vy = np.gradient(traj['y']) / np.gradient(traj['time'])
            vz = np.gradient(traj['z']) / np.gradient(traj['time'])
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            
            ax4.plot(traj['time'], v_mag, color=traj['color'], 
                    linewidth=2, label=f"{traj['label']} Speed")
        
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Speed (m/s)')
        ax4.set_title('Speed Profile')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Distance from start
        ax5 = plt.subplot(2, 3, 6)
        for traj in self.trajectories:
            start_pos = np.array([traj['x'][0], traj['y'][0], traj['z'][0]])
            distances = []
            for i in range(len(traj['x'])):
                pos = np.array([traj['x'][i], traj['y'][i], traj['z'][i]])
                dist = np.linalg.norm(pos - start_pos)
                distances.append(dist)
            
            ax5.plot(traj['time'], distances, color=traj['color'], 
                    linewidth=2, label=f"{traj['label']} Distance")
        
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Distance from Start (m)')
        ax5.set_title('Distance from Starting Point')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Trajectory analysis saved to: {save_path}")
        
        plt.show()
        return fig
    
    def get_trajectory_stats(self):
        """Get statistics for all trajectories."""
        stats = {}
        for traj in self.trajectories:
            label = traj['label']
            
            # Calculate velocities
            vx = np.gradient(traj['x']) / np.gradient(traj['time'])
            vy = np.gradient(traj['y']) / np.gradient(traj['time'])
            vz = np.gradient(traj['z']) / np.gradient(traj['time'])
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2)
            
            # Calculate distance traveled
            distances = np.sqrt(np.diff(traj['x'])**2 + np.diff(traj['y'])**2 + np.diff(traj['z'])**2)
            total_distance = np.sum(distances)
            
            stats[label] = {
                'duration': traj['time'][-1] - traj['time'][0],
                'total_distance': total_distance,
                'max_speed': np.max(v_mag),
                'avg_speed': np.mean(v_mag),
                'altitude_range': (np.min(traj['z']), np.max(traj['z'])),
                'final_position': (traj['x'][-1], traj['y'][-1], traj['z'][-1]),
                'max_distance_from_start': np.max(np.sqrt(
                    (traj['x'] - traj['x'][0])**2 + 
                    (traj['y'] - traj['y'][0])**2 + 
                    (traj['z'] - traj['z'][0])**2
                ))
            }
        
        return stats
    
    def print_stats(self):
        """Print trajectory statistics."""
        stats = self.get_trajectory_stats()
        
        print("\n" + "="*50)
        print("TRAJECTORY ANALYSIS")
        print("="*50)
        
        for label, stat in stats.items():
            print(f"\n{label}:")
            print(f"  Duration: {stat['duration']:.2f} seconds")
            print(f"  Total distance: {stat['total_distance']:.2f} m")
            print(f"  Max speed: {stat['max_speed']:.2f} m/s")
            print(f"  Avg speed: {stat['avg_speed']:.2f} m/s")
            print(f"  Altitude range: {stat['altitude_range'][0]:.2f} to {stat['altitude_range'][1]:.2f} m")
            print(f"  Final position: ({stat['final_position'][0]:.2f}, {stat['final_position'][1]:.2f}, {stat['final_position'][2]:.2f}) m")
            print(f"  Max distance from start: {stat['max_distance_from_start']:.2f} m")
    
    def clear_trajectories(self):
        """Clear all stored trajectories."""
        self.trajectories = []


def create_racing_gate_plotter():
    """Create a plotter configured for the racing environment."""
    # Race track configuration from training
    r = 1.5
    gate_pos = np.array([
        [ r,  -r, -1.5],
        [ 0,   0, -1.5],
        [-r,   r, -1.5],
        [ 0, 2*r, -1.5],
        [ r,   r, -1.5],
        [ 0,   0, -1.5],
        [-r,  -r, -1.5],
        [ 0,-2*r, -1.5]
    ])
    gate_yaw = np.array([1,2,1,0,-1,-2,-1,0])*np.pi/2
    
    return TrajectoryPlotter(gate_pos=gate_pos, gate_yaw=gate_yaw)
