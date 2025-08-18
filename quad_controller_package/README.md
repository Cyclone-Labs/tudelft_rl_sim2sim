# Quad Controller

A portable neural network controller for quadcopter racing trained with domain randomization. This controller generalizes across different quadcopter platforms without retraining.

## Features

- **Cross-platform compatibility**: Works on both 3-inch and 5-inch racing drones
- **Domain randomization**: Trained across wide parameter variations for robustness
- **Gym-compatible**: Easy integration with any gym-style simulation environment
- **Lightweight**: Single class import with bundled weights
- **High performance**: 64x64x64 neural network optimized for real-time control

## Installation

```bash
pip install -e .
```

Or install with development dependencies:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from quad_controller import QuadController

# Initialize controller
controller = QuadController()

# Use in your simulation loop
observation = env.reset()  # 20-dimensional observation
action = controller.predict(observation)  # 4-dimensional motor commands
next_obs, reward, done, info = env.step(action)
```

## Observation Format

The controller expects a 20-dimensional observation vector:

| Indices | Component | Description |
|---------|-----------|-------------|
| 0-2 | Position | Position relative to target gate (x, y, z) |
| 3-5 | Velocity | Velocity relative to target gate (vx, vy, vz) |
| 6-8 | Attitude | Euler angles relative to target gate (roll, pitch, yaw) |
| 9-11 | Body rates | Angular velocities in body frame (p, q, r) |
| 12-15 | Motor speeds | Current motor speeds (normalized) |
| 16-18 | Next gate pos | Next gate position relative to current gate |
| 19 | Next gate yaw | Next gate orientation angle |

## Action Format

The controller outputs 4-dimensional motor commands in range [-1, 1]:
- `action[0]`: Motor 1 command
- `action[1]`: Motor 2 command  
- `action[2]`: Motor 3 command
- `action[3]`: Motor 4 command

## Example Usage

### Basic Control Loop
```python
from quad_controller import QuadController
import numpy as np

controller = QuadController()

# Test with random observation
obs = np.random.randn(20)
action = controller.predict(obs)
print(f"Action: {action}")

# Batch prediction
batch_obs = np.random.randn(10, 20)
batch_actions = controller.predict_batch(batch_obs)
print(f"Batch actions shape: {batch_actions.shape}")
```

### Integration with Custom Environment
```python
from quad_controller import QuadController

controller = QuadController()

# Your environment setup
env = YourQuadEnvironment()
obs = env.reset()

for step in range(1000):
    action = controller.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    if done:
        obs = env.reset()
```

### Getting Controller Information
```python
controller = QuadController()

# Get action bounds
low_bounds, high_bounds = controller.get_action_bounds()
print(f"Action range: [{low_bounds}] to [{high_bounds}]")

# Get observation format info
obs_info = controller.get_observation_info()
print(f"Observation components: {obs_info['components'].keys()}")

# Test the controller
controller.test_controller(num_random_tests=5)
```

## Training Details

This controller was trained using:
- **Algorithm**: Proximal Policy Optimization (PPO)
- **Training steps**: 100 million timesteps
- **Parallel environments**: 100 simulated drones
- **Domain randomization**: Wide parameter variations covering different drone types
- **Architecture**: 3-layer MLP with 64 neurons per layer

## Performance

Based on the original paper results:
- **Simulation**: 64.67 episode reward (3-inch), 76.78 (5-inch)
- **Real-world**: Successfully tested on physical 3-inch and 5-inch racing drones
- **Speed**: Up to 10 m/s flight speeds
- **Robustness**: Works across different battery levels and environmental conditions

## Citation

If you use this controller in your research, please cite:

```bibtex
@article{ferede2025domain,
  title={One Net to Rule Them All: Domain Randomization in Quadcopter Racing Across Different Platforms},
  author={Ferede, Robin and Blaha, Till and Lucassen, Erin and De Wagter, Christophe and de Croon, Guido C.H.E.},
  journal={arXiv preprint arXiv:2504.21586},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
