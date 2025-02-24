import pybullet as p
import pybullet_data
import numpy as np
import time

class KukaIiwa7DigitalTwin:
    def __init__(self, urdf_path):
        """
        Initialize the digital twin for the KUKA iiwa 7 robotic manipulator.

        Parameters:
        - urdf_path: Path to the URDF file.
        """
        self.urdf_path = urdf_path
        self.physics_client = None
        self.robot_ids = []
        self.block_ids = []  # Store block IDs
        self.num_joints = 0  # Number of joints in the manipulator
        self.initial_target_position = [0.5, 0.3, 1.0]  # Initial target point (height of 1m)
        self.target_position = np.array(self.initial_target_position)  # Target position
        self.angle = 0  # Initial angle for circular motion
        self.target_radius = 0.2  # Radius of the circle for target movement
        self.ground_id = None  # ID for the ground

    # define a control barrier function to ensure that the end-effectors of the robots are at least a certain distance apart
    def control_barrier_function(self, robot_positions, min_distance):
        """
        Compute the control barrier function to ensure a minimum distance between robot end-effectors.

        Parameters:
        - robot_positions: List of robot end-effector positions.
        - min_distance: Minimum distance between end-effectors.

        Returns:
        - Control barrier function value.
        """
        num_robots = len(robot_positions)
        cbf_value = 0

        for i in range(num_robots):
            for j in range(i + 1, num_robots):
                distance = np.linalg.norm(robot_positions[i] - robot_positions[j])
                cbf_value += max(0, min_distance - distance)

        return cbf_value
    


    def initialize_simulation(self):
        """Initialize the PyBullet simulation environment and add three robots."""
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Add a ground with collision but no friction
        self.add_ground()

        # Define the positions of the three robots at the corners of a triangle
        block_positions = [
            [0, 0, 0.25],        # Block 1 (centered at the base of Robot 1)
            [1, 0, 0.25],        # Block 2 (centered at the base of Robot 2)
            [0.5, np.sqrt(3)/2, 0.25]  # Block 3 (centered at the base of Robot 3)
        ]

        # Add movable blocks at the base of each robot
        self.add_movable_blocks(block_positions)

        # Load manipulators and attach them to the blocks
        for i, block_pos in enumerate(block_positions):
            robot_id = p.loadURDF(
                self.urdf_path,
                basePosition=[block_pos[0], block_pos[1], block_pos[2] + 0.25],  # Offset manipulator base to top of block
                useFixedBase=False  # Manipulators are not fixed to the world
            )
            self.robot_ids.append(robot_id)

            # Create a fixed constraint between the block and the manipulator base
            #self.create_fixed_constraint(self.block_ids[i], robot_id)

        self.num_joints = p.getNumJoints(self.robot_ids[0])
        
        # Enable camera controls and remove PyBullet's sidebar
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0.5, 0])
        
        # Add debug parameters for camera control
        self.camera_distance = p.addUserDebugParameter("Camera Distance", 1, 5, 2.5)
        self.camera_yaw = p.addUserDebugParameter("Camera Yaw", -180, 180, 50)
        self.camera_pitch = p.addUserDebugParameter("Camera Pitch", -90, 90, -35)
        self.camera_target_x = p.addUserDebugParameter("Camera Target X", -5, 5, 0.5)
        self.camera_target_y = p.addUserDebugParameter("Camera Target Y", -5, 5, 0.5)
        self.camera_target_z = p.addUserDebugParameter("Camera Target Z", -5, 5, 1.0)

        # Add buttons for play, stop, manual time control, and termination
        self.play_button = p.addUserDebugParameter("Play", 1, 0, 0)
        self.stop_button = p.addUserDebugParameter("Stop", 1, 0, 0)
        self.time_slider = p.addUserDebugParameter("Time Step", 0.001, 0.1, 0.01)
        self.terminate_button = p.addUserDebugParameter("Terminate", 1, 0, 0)

        # Add a marker for the target point
        self.marker_id = None

    def add_ground(self):
        """Add a ground plane with collision and no friction."""
        # Create a plane collision shape
        ground_shape = p.createCollisionShape(p.GEOM_PLANE)

        # Create the ground body
        self.ground_id = p.createMultiBody(
            baseMass=0,  # Static plane
            baseCollisionShapeIndex=ground_shape,
            baseVisualShapeIndex=-1  # No visual shape for the ground
        )

        # Set the friction of the ground to zero
        p.changeDynamics(self.ground_id, -1, lateralFriction=0.0)

    def add_movable_blocks(self, positions):
        """
        Add movable blocks at specified positions.

        Parameters:
        - positions: List of [x, y, z] positions for each block.
        """
        block_size = [0.5, 0.5, 0.5]  # Dimensions: 0.5m x 0.5m x 0.5m
        block_mass = 10.0  # Nonzero mass to make the blocks movable
        for pos in positions:
            # Create a visual and collision shape for the block
            collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[d / 2 for d in block_size])
            visual_shape_id = p.createVisualShape(
                p.GEOM_BOX, 
                halfExtents=[d / 2 for d in block_size], 
                rgbaColor=[0.5, 0.5, 0.5, 1]  # Grey color
            )

            # Create the block as a movable object
            block_id = p.createMultiBody(
                baseMass=block_mass,  # Nonzero mass makes it movable
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=pos
            )
            self.block_ids.append(block_id)

    def create_fixed_constraint(self, block_id, robot_id):
        """
        Create a fixed constraint between a block and a manipulator.

        Parameters:
        - block_id: ID of the block.
        - robot_id: ID of the manipulator.
        """
        constraint_id = p.createConstraint(
            parentBodyUniqueId=block_id,
            parentLinkIndex=-1,  # Base link of the block
            childBodyUniqueId=robot_id,
            childLinkIndex=-1,  # Base link of the manipulator
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.25],  # Offset at the top of the block
            childFramePosition=[0, 0, 0]       # Bottom of the manipulator
        )
        self.constraints.append(constraint_id)

    def update_marker(self, target_point):
        """Update the marker to indicate the target point."""
        if self.marker_id is not None:
            p.removeBody(self.marker_id)
        visual_shape_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0, 1, 0, 1])  # Green sphere
        self.marker_id = p.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=target_point)

    def real_time_consensus(self):
        """
        Perform real-time trajectory generation and movement to a consensus point.
        """
        convergence_threshold = 0.01  # Threshold for convergence
        min_distance = 0.2  # Minimum distance between robots for collision avoidance

        while True:
            # Update the camera view based on user inputs
            cam_dist = p.readUserDebugParameter(self.camera_distance)
            cam_yaw = p.readUserDebugParameter(self.camera_yaw)
            cam_pitch = p.readUserDebugParameter(self.camera_pitch)
            cam_target_x = p.readUserDebugParameter(self.camera_target_x)
            cam_target_y = p.readUserDebugParameter(self.camera_target_y)
            cam_target_z = p.readUserDebugParameter(self.camera_target_z)
            p.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, [cam_target_x, cam_target_y, cam_target_z])

            play = p.readUserDebugParameter(self.play_button)
            stop = p.readUserDebugParameter(self.stop_button)
            terminate = p.readUserDebugParameter(self.terminate_button)
            time_step = p.readUserDebugParameter(self.time_slider)

            if terminate > 0:
                print("Simulation terminated.")
                break

            if stop > 0:
                time.sleep(0.01)
                continue

            if play > 0:
                all_converged = True
                for robot_id in self.robot_ids:
                    ee_position = np.array(p.getLinkState(robot_id, self.num_joints - 1)[0])
                    direction = np.array(self.target_position) - ee_position
                    distance = np.linalg.norm(direction)

                    if distance > convergence_threshold:
                        all_converged = False
                        step = direction / distance * min(distance, 1 * distance)  # Small step towards target
                        new_position = ee_position + step

                        # Only care about position, not orientation
                        joint_positions = p.calculateInverseKinematics(
                            robot_id, self.num_joints - 1, new_position
                        )

                        for i in range(self.num_joints):
                            p.setJointMotorControl2(
                                bodyUniqueId=robot_id,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=joint_positions[i]
                            )

                    # Avoid collisions with other robots
                    for j, robot_id_2 in enumerate(self.robot_ids):
                        if i != j:  # Avoid self-collision
                            ee_position_2 = np.array(p.getLinkState(robot_id_2, self.num_joints - 1)[0])
                            distance_between_robots = np.linalg.norm(ee_position - ee_position_2)

                            if distance_between_robots < min_distance:
                                # Calculate a repulsive force or corrective velocity
                                repulsive_vector = ee_position - ee_position_2
                                repulsive_distance = min_distance - distance_between_robots
                                repulsive_velocity = repulsive_vector / np.linalg.norm(repulsive_vector) * repulsive_distance

                                # Apply corrective movement
                                new_position += repulsive_velocity

                                # Recalculate the joint positions after corrective movement
                                joint_positions = p.calculateInverseKinematics(
                                    robot_id, self.num_joints - 1, new_position
                                )

                                for k in range(self.num_joints):
                                    p.setJointMotorControl2(
                                        bodyUniqueId=robot_id,
                                        jointIndex=k,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_positions[k]
                                    )

                # Move the target point along a circle with radius 0.2 meters
                self.update_target_position()

                # Update the marker to the new target position
                self.update_marker(self.target_position)

                p.stepSimulation()
                time.sleep(time_step)

                if all_converged:
                    print("All robots have converged to the target point.")
                    break

    def update_target_position(self):
        """Move the target point smoothly along a circle with radius 0.2 meters."""
        angular_velocity = 10  # Angular velocity (radians per second)
        self.angle += angular_velocity * 0.01  # Increment angle by the angular velocity * time step

        # Define the circular trajectory in the X-Y plane, keeping the Z constant at 1.0m
        target_x = self.initial_target_position[0] + self.target_radius * np.cos(self.angle)
        target_y = self.initial_target_position[1] + self.target_radius * np.sin(self.angle)
        target_z = 1.0  # Height of 1m

        self.target_position = np.array([target_x, target_y, target_z])

    def run_simulation(self):
        """Run the simulation indefinitely."""
        print("Running simulation. Use the GUI to control play, stop, and time step.")
        while True:
            terminate = p.readUserDebugParameter(self.terminate_button)
            if terminate > 0:
                print("Simulation terminated.")
                break
            p.stepSimulation()
            time.sleep(0.01)

# Main execution file (main.py)
if __name__ == "__main__":
    urdf_path = "kuka_iiwa/model.urdf"  # Update with the actual URDF path
    twin = KukaIiwa7DigitalTwin(urdf_path)
    twin.initialize_simulation()

    # Example usage: Real-time consensus trajectory generation
    twin.real_time_consensus()

    # Keep simulation running
    twin.run_simulation()