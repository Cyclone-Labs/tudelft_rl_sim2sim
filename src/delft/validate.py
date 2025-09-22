from quad_race_env import Quadcopter3DGates
from randomization import randomization_fixed_params_5inch
import rerun as rr
from stable_baselines3 import PPO
from scipy.spatial.transform import Rotation
import numpy as np
from pathlib import Path
import subprocess
import os
import matplotlib.pyplot as plt
import time

# DEFINE RACE TRACK
easy_pos = np.array([
    [ 13.5 , 6. , -6.2],
    [ 11.  , 14., -6.2],
    [ 6.   , 22., -6.2],
    [ 11.  , 30., -6.2],
    [ 11. - 2.1*np.cos(np.pi/3.), 30.+ 2.1*np.sin(np.pi/3.), -5.1],
    [ 11.  , 30., -4.1],
    [ 19.  , 34., -6.2],
    [ 27.  , 30., -6.2],
    [ 32.  , 22., -6.2],
    [ 29.  , 14., -6.2],
    [ 30.  , 6. , -6.2],
    [ 17.  , 18., -6.2],
    [ 13.5 , 6. , -4.1]
])
easy_yaw = np.array([
                     7/12,
                     1/3,
                     2/3,
                     1/6,
                     -5/6,
                     1/6,
                     0,
                     -1/6,
                     -1/2,
                     -5/12,
                     -7/12,
                     1,
                     -5/12])*np.pi
easy_start = easy_pos[0] + np.array([1.,-3.,0])

hard_pos = np.delete(easy_pos, 4, axis=0)
hard_yaw = np.delete(easy_yaw, 4)
hard_start = hard_pos[0] + np.array([1.,-3.,0])

fly_around_pos = np.array([
    [ 13.5 , 6. , -6.2],
    [ 11.  , 14., -6.2]
])
fly_around_yaw = np.array([
                     7/12,
                     1/3
                     ])*np.pi
fly_around_start = fly_around_pos[0] + np.array([1.,-3.,0])

# init rerun
rr.init("quadcopter_controller_validation", spawn=False)

file_path = Path(__file__).parent
   
def main():
    model_path = file_path / 'models' / 'general_session' / 'general_model'

    try:
        most_recent_model = max(model_path.glob('*'), key=os.path.getmtime)
        print(f"Most recent file: {most_recent_model}")
    except ValueError:
        print("No files found in the models folder")

    model_path = most_recent_model

    model = PPO.load(str(model_path))
    env = Quadcopter3DGates(num_envs=1, 
                            randomization=randomization_fixed_params_5inch, 
                            gates_pos=hard_pos,
                            gate_yaw=hard_yaw,
                            start_pos=hard_start,
                            initialize_at_random_gates=False)
    obs = env.reset()
    
    trajectory_points = []
    step_count = 0

    exitCondition = False

    for _ in range(5000):
        
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        if done[0]:
            # Debug - Jacob
            exitCondition = True
            print("Terminated: ", done[0])
            print("")
            report = env.return_dones_report()
            print("max steps: ",report[0])
            print("ground col: ",report[1])
            print("out of bounds: ",report[2])
            print("gate col: ",report[3])
            print("")
            #
            break
        
        # Extract drone state from world_states (first environment) - this is the correct source
        drone_position = env.world_states[0, 0:3]  # [x, y, z] in world frame
        drone_velocity = env.world_states[0, 3:6]  # [vx, vy, vz] in world frame
        drone_euler = env.world_states[0, 6:9]     # [phi, theta, psi] in world frame
        
        rr.log(
                'drone/drone_model',
                rr.Transform3D(
                    translation=drone_position,
                    quaternion=Rotation.from_euler('xyz', drone_euler).as_quat()
                    
                )
            )
        rr.log(
                'drone/drone_model',
                rr.Asset3D(path=file_path / 'Drone.obj')
            )
        
        # Log drone transform using world coordinates
        rr.log('drone/transform', 
               rr.Transform3D(
                   translation=drone_position,
                   mat3x3=Rotation.from_euler('xyz', drone_euler).as_matrix(),
                   axis_length=1.2
               ))
        
        rr.log('drone/phi', rr.Scalars(drone_euler[0]))
        rr.log('drone/theta', rr.Scalars(drone_euler[1]))
        rr.log('drone/psi', rr.Scalars(drone_euler[2]))
        
        # Add to trajectory and log it
        trajectory_points.append(drone_position.copy())
        rr.log('drone/trajectory', rr.LineStrips3D([trajectory_points], colors=[0, 255, 0]))
        
        # # Log current position as a point
        # rr.log('drone/position', rr.Points3D([drone_position], colors=[255, 0, 0], radii=[0.1]))
        
        # Log velocity vector
        if np.linalg.norm(drone_velocity) > 0.1:  # Only show if moving
            rr.log('drone/velocity', 
                   rr.Arrows3D(
                       origins=[drone_position], 
                       vectors=[drone_velocity * 0.5],  # Scale for visibility
                       colors=[0, 0, 255]
                   ))
        # Log gates
        for i, (gate_pos, gate_yaw) in enumerate(zip(env.gate_pos, env.gate_yaw)):
            color = [0, 255, 0] if i == env.target_gates[0] else [128, 128, 128]  # Green for target, gray for others
            
            # Log gate center
            rr.log(f'gates/gate_{i}/position', rr.Points3D([gate_pos], colors=[color], radii=[0.05])) # .75 radii for gate size
            
            # Log gate orientation (as a small arrow)
            gate_direction = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0]) * 0.5
            rr.log(f'gates/gate_{i}/direction', 
                    rr.Arrows3D(
                        origins=[gate_pos], 
                        vectors=[gate_direction],
                        colors=[color]
                    ))
            
            # Log 3d gate model
            obj_file_path = file_path / "gate.obj"

            if not os.path.exists(obj_file_path):
                print(f"WARNING: File {obj_file_path} not found!")

            instance_path = f"gate_models/gate_{i}"
            rr.log(
                instance_path,
                rr.Transform3D(
                    translation=gate_pos,
                    rotation=rr.Quaternion(xyzw=Rotation.from_euler('XYZ', [0.0, 0.0, gate_yaw]).as_quat())
                )
            )
            rr.log(
                f"{instance_path}/model",
                rr.Asset3D(path=obj_file_path)
            )
            # rr.log(
            #     f"{instance_path}/texture",
            #     rr.Image(plt.imread(png_file_path),color_model="RGBA")
            # )

        # Log metrics
        rr.log('metrics/reward', rr.Scalars(reward[0]))
        rr.log('metrics/step', rr.Scalars(step_count))
        rr.log('metrics/distance_to_target', 
               rr.Scalars(np.linalg.norm(drone_position - env.gate_pos[env.target_gates[0]])))
        
        # Log motor speeds
        motor_speeds = env.world_states[0, 12:16]
        for i, speed in enumerate(motor_speeds):
            rr.log(f'motors/motor_{i+1}', rr.Scalars(speed))
        
        # Log collision events
        if info[0].get("gate_passed", False):
            rr.log('events/gate_passed', rr.TextLog(f"Gate {env.target_gates[0]-1} passed!"))
        
        if info[0].get("gate_collision", False):
            rr.log('events/collision', rr.TextLog("Gate collision!"))
            
        if info[0].get("ground_collision", False):
            rr.log('events/collision', rr.TextLog("Ground collision!"))
        
        step_count += 1
        
    
    # Debug - Jacob
    if exitCondition == False:
        print("")
        print("END OF TIME")
        print("")
    
    env.close()


# save rerun data
rr.save('quadcopter_validation.rrd')

if __name__ == "__main__":
    main()

subprocess.Popen(['rerun', 'quadcopter_validation.rrd'])