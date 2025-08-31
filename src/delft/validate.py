from delft.quad_race_env import Quadcopter3DGates
from delft.randomization import randomization_fixed_params_5inch
import rerun as rr
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation

# init rerun
rr.init("quadcopter_controller_validation", spawn=False)

def main():
    model_path = '/home/miller/code/optimal_quad_control_RL/src/quad_controller_package/quad_controller/100000000.zip'
    model = PPO.load(model_path)
    env = Quadcopter3DGates(num_envs=1, randomization=randomization_fixed_params_5inch)
    obs = env.reset()
    # position_history = []
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        # log drone position as transform for first env
        rr.log('quad_transform',
               rr.Transform3D(
                     translation=obs[0,:3],
                     mat3x3=Rotation.from_euler('XYZ', obs[0,6:9]).as_matrix(),
                     axis_length=0.01
               ))
        # position_history.append(obs[0,:3].copy())
        # rr.log('quad_trajectory', rr.LineStrips3D(positions=position_history, radii=0.005))

    env.close()

# save rerun data
rr.save('data.rrd')

if __name__ == "__main__":
    main()