import gymnasium as gym

from pydrake.all import (
    DiagramBuilder, SceneGraph, Simulator, System, Context,
    MeshcatVisualizer, EventStatus
)
from pydrake.gym._drake_gym_env import DrakeGymEnv
from custom_quadrotor_plant import CustomQuadrotorPlant, CustomQuadrotorGeometry
import numpy as np

gate_locations = [
    np.array([1.0, 1.0, 1.0]),
    np.array([2.0, 1.0, 1.0]),
    np.array([1.0, 2.0, 1.0]),
]

class QuadrotorEnv:
    @staticmethod
    def build(meshcat, target_position: np.ndarray, target_velocity: np.ndarray, realtime_rate=-1, drone_type="5inch"):
        # State is x ,y , z, roll, pitch, yaw + velocities.
        builder = DiagramBuilder()

        plant = builder.AddSystem(CustomQuadrotorPlant(drone_type=drone_type))


        PLANT_INPUT_NAME = "plant_input"
        PLANT_OUTPUT_NAME = "state"
        builder.ExportInput(plant.get_input_port(0), PLANT_INPUT_NAME)
        builder.ExportOutput(plant.get_output_port(0), PLANT_OUTPUT_NAME)


        # Set up visualization in MeshCat
        scene_graph = builder.AddSystem(SceneGraph())
        CustomQuadrotorGeometry.AddToBuilder(builder, plant.get_output_port(0), scene_graph)
        if meshcat is not None:
            meshcat.Delete()
            meshcat.ResetRenderMode()
            meshcat.SetProperty("/Background", "visible", False)
            MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
            input("Press enter once meshcat has finished initializing and you're ready")
        # end setup for visualization

        diagram = builder.Build()

        # Set up a simulator to run this diagram
        simulator = Simulator(diagram)
        simulator.set_target_realtime_rate(realtime_rate)


        def reward_fn(system: System, context: Context):
            state = system.get_output_port(0).Eval(context) 
            # print(state.shape, state[:3])
            current_waypoint = np.array([1.0, 1.0, 1.0])  # Replace with dynamic waypoint logic later
            waypoint_err = np.linalg.norm(state[:3] - current_waypoint)
            
            # Example: reward is negative distance to waypoint
            return -waypoint_err
        
        def monitor_fn(context: Context) -> EventStatus:
            system = simulator.get_system()
            steps = simulator.get_num_steps_taken()
            if steps > 5000:
                # TODO: add a wrapper around Drake's gym env helper to turn on the truncated flag
                # currently there is no way of differentiating this from termination.
                return EventStatus.ReachedTermination(system, 'over 5000 timesteps in episode')
            state = system.get_output_port(0).Eval(context)
            if np.linalg.norm(state[:3]) > 3:
                return EventStatus.ReachedTermination(system, 'too far from origin')
            if state[3]<-1:
                return EventStatus.ReachedTermination(system, 'height too low')
            return EventStatus.DidNothing()

        simulator.set_monitor(monitor_fn)

        env = DrakeGymEnv(
            simulator, 
            1/100, 
            gym.spaces.Box(low=0, high=10, shape=(4,)),
            gym.spaces.Box(low=-300, high=300, shape=(12,)),
            reward_fn,
            PLANT_INPUT_NAME,
            PLANT_OUTPUT_NAME,
        )
        
        # Store plant reference for accessing parameters
        env.plant = plant
        
        return env