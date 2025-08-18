"""
Custom QuadrotorPlant and QuadrotorGeometry that matches our training parameters.

This is a modified version of Drake's quadrotor plant with parameters tuned to 
match the 3-inch and 5-inch racing drones used in training.
"""

import numpy as np
from pydrake.all import (
    LeafSystem, BasicVector, DiagramBuilder, SceneGraph, 
    Box, RigidTransform, RotationMatrix, RollPitchYaw,
    IllustrationProperties, GeometryInstance, GeometryFrame, Rgba,
    FramePoseVector, Value
)

class CustomQuadrotorPlant(LeafSystem):
    """
    Custom quadrotor plant with parameters tuned for racing drones.
    Based on Drake's QuadrotorPlant but with racing drone parameters.
    """
    
    def __init__(self, drone_type="5inch"):
        LeafSystem.__init__(self)
        
        # Set parameters based on drone type to match training data
        if drone_type == "3inch":
            self._m = 0.150  # 150g for 3-inch racing quad
            self._L = 0.076  # 76mm motor-to-motor distance
            self._kF = 7.0   # Adjusted force constant for lighter quad
            self._kM = 0.015 # Moment constant
            # Inertia for small racing quad
            self._I = np.array([
                [0.0008, 0.0, 0.0],
                [0.0, 0.0008, 0.0], 
                [0.0, 0.0, 0.0012]
            ])
        else:  # 5inch (default)
            self._m = 0.400  # 400g for 5-inch racing quad  
            self._L = 0.127  # 127mm motor-to-motor distance
            self._kF = 6.0   # Force constant for heavier quad
            self._kM = 0.025 # Moment constant 
            # Inertia for 5-inch racing quad
            self._I = np.array([
                [0.0025, 0.0, 0.0],
                [0.0, 0.0025, 0.0],
                [0.0, 0.0, 0.0035]
            ])
        
        self._g = 9.81  # Gravity
        
        # Four inputs -- one for each propeller (following Drake's structure)
        self.DeclareVectorInputPort("propeller_force", BasicVector(4))
        
        # State is x, y, z, roll, pitch, yaw + velocities (12D)
        state_index = self.DeclareContinuousState(12)
        self.DeclareStateOutputPort("state", state_index)
        
        print(f"Custom {drone_type} quadrotor plant created:")
        print(f"  Mass: {self._m:.3f} kg")
        print(f"  Length: {self._L:.3f} m") 
        print(f"  Force constant: {self._kF:.2f}")
        print(f"  Moment constant: {self._kM:.3f}")
    
    def DoCalcTimeDerivatives(self, context, derivatives):
        """Compute the time derivatives of the state (following Drake's implementation)."""
        # Get input (propeller forces)
        u_vec = self.EvalVectorInput(context, 0)
        u = u_vec.CopyToVector() if u_vec is not None else np.zeros(4)
        
        # Extract state
        state = context.get_continuous_state_vector().CopyToVector()
        xyz = state[0:3]
        rpy = state[3:6]
        xyz_dot = state[6:9]
        rpy_dot = state[9:12]
        
        # For each rotor, calculate the Bz measure of its aerodynamic force
        uF_Bz = self._kF * u
        
        # Compute net aerodynamic force on body, expressed in body frame
        Faero_B = np.array([0, 0, np.sum(uF_Bz)])
        
        # Compute moments about body center from rotor forces
        # Following Drake's rotor layout:
        # Motor 0: front-right, Motor 1: front-left, Motor 2: rear-left, Motor 3: rear-right
        Mx = self._L * (uF_Bz[1] - uF_Bz[3])  # Roll moment
        My = self._L * (uF_Bz[2] - uF_Bz[0])  # Pitch moment
        
        # Yaw moment from rotor torques (rotors 0,2 spin one way, 1,3 the other)
        uTau_Bz = self._kM * u
        Mz = uTau_Bz[0] - uTau_Bz[1] + uTau_Bz[2] - uTau_Bz[3]
        
        Tau_B = np.array([Mx, My, Mz])
        
        # Gravity force in world frame
        Fgravity_N = np.array([0, 0, -self._m * self._g])
        
        # Convert body forces to world frame
        rpy_obj = RollPitchYaw(rpy[0], rpy[1], rpy[2])
        R_NB = RotationMatrix(rpy_obj).matrix()
        
        # Net force in world frame
        Fnet_N = Fgravity_N + R_NB @ Faero_B
        xyz_ddot = Fnet_N / self._m
        
        # Angular dynamics
        # Calculate angular velocity in body frame from rpy rates
        w_NB_B = rpy_obj.CalcAngularVelocityInChildFromRpyDt(rpy_dot)
        
        # Solve Euler's equation: τ = I*α + ω × (I*ω)
        wIw = np.cross(w_NB_B, self._I @ w_NB_B)
        alpha_NB_B = np.linalg.solve(self._I, Tau_B - wIw)
        alpha_NB_N = R_NB @ alpha_NB_B
        
        # Calculate second derivative of rpy
        rpy_ddot = rpy_obj.CalcRpyDDtFromRpyDtAndAngularAccelInParent(rpy_dot, alpha_NB_N)
        
        # Pack derivatives: [xyz_dot, rpy_dot, xyz_ddot, rpy_ddot]
        xDt = np.concatenate([xyz_dot, rpy_dot, xyz_ddot, rpy_ddot])
        derivatives.get_mutable_vector().SetFromVector(xDt)
    
    # Property accessors (for compatibility with Drake's QuadrotorPlant)
    def m(self):
        return self._m
    
    def length(self):
        return self._L
    
    def force_constant(self):
        return self._kF
    
    def moment_constant(self):
        return self._kM
    
    def inertia(self):
        return self._I
    
    def g(self):
        return self._g


class CustomQuadrotorGeometry(LeafSystem):
    """Custom quadrotor geometry system that follows Drake's pattern."""
    
    @staticmethod
    def AddToBuilder(builder, quadrotor_state_port, scene_graph):
        """Add quadrotor geometry to the scene graph."""
        quadrotor_geometry = builder.AddSystem(CustomQuadrotorGeometry(scene_graph))
        builder.Connect(quadrotor_state_port, quadrotor_geometry.get_input_port(0))
        builder.Connect(
            quadrotor_geometry.get_output_port(0),
            scene_graph.get_source_pose_port(quadrotor_geometry.source_id_)
        )
        return quadrotor_geometry
    
    def __init__(self, scene_graph):
        LeafSystem.__init__(self)
        
        # Register source and add geometry
        self.source_id_ = scene_graph.RegisterSource("quadrotor")
        
        # Simple box geometry for racing drone
        body_shape = Box(0.15, 0.15, 0.05)  # Racing frame dimensions
        body_instance = GeometryInstance(RigidTransform(), body_shape, "body")
        
        # Set appearance
        props = IllustrationProperties()
        props.AddProperty("phong", "diffuse", Rgba(0.2, 0.2, 0.8, 1.0))  # Blue racing quad
        body_instance.set_illustration_properties(props)
        
        # Register frame and geometry
        self.frame_id_ = scene_graph.RegisterFrame(
            self.source_id_, GeometryFrame("quadrotor_frame")
        )
        scene_graph.RegisterGeometry(self.source_id_, self.frame_id_, body_instance)
        
        # Declare input/output ports
        self.DeclareVectorInputPort("state", BasicVector(12))
        self.DeclareAbstractOutputPort(
            "geometry_pose", 
            lambda: Value(FramePoseVector()),
            self.OutputGeometryPose
        )
    
    def OutputGeometryPose(self, context, poses):
        """Output the geometry pose based on quadrotor state."""
        state = self.get_input_port(0).Eval(context)
        
        # Extract position and orientation from state
        xyz = state[0:3]
        rpy = state[3:6]
        
        # Create rigid transform
        pose = RigidTransform(
            RollPitchYaw(rpy[0], rpy[1], rpy[2]),
            xyz
        )
        
        # Create FramePoseVector and set the pose
        frame_poses = FramePoseVector()
        frame_poses.set_value(self.frame_id_, pose)
        poses.set_value(frame_poses)
