import pybullet as pybullet
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
        self.constraints = []  # Store constraint IDs
        self.num_joints = 0  # Number of joints in the manipulator
        self.initial_target_position = [0.5, 0.3, 1.0]  # Initial target point (height of 1m)
        self.target_position = np.array(self.initial_target_position)  # Target position
        self.angle = 0  # Initial angle for circular motion
        self.target_radius = 0.2  # Radius of the circle for target movement
        self.ground_id = None  # ID for the ground

    def initialize_simulation(self):
        """Initialize the PyBullet simulation environment and add three robots."""
        self.physics_client = pybullet.connect(pybullet.GUI)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.81)

        # Add a ground with collision but no friction
        self.add_ground()

        # Define the positions of the three robots at the corners of a triangle
        block_positions = [
            [0, 0, 0.25],        # Block 1 (centered at the base of Robot 1)
            [1, 0, 0.25],        # Block 2 (centered at the base of Robot 2)
            [0, 1, 0.25],  # Block 3 (centered at the base of Robot 3)
            [1, 1, 0.25] # Block 4 (centered at the base of Robot 4)
        ]

        # Add movable blocks at the base of each robot
        self.add_movable_blocks(block_positions)

        # Load manipulators and attach them to the blocks
        for i, block_pos in enumerate(block_positions):
            robot_id = pybullet.loadURDF(
                self.urdf_path,
                basePosition=[block_pos[0], block_pos[1], block_pos[2] + 0.25],  # Offset manipulator base to top of block
                useFixedBase=False  # Manipulators are not fixed to the world
            )
            self.robot_ids.append(robot_id)

            
        self.num_joints = pybullet.getNumJoints(self.robot_ids[0])
        
        # Enable camera controls and remove PyBullet's sidebar
        pybullet.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=[0.5, 0.5, 0])
        
        # Add debug parameters for camera control
        self.camera_distance = pybullet.addUserDebugParameter("Camera Distance", 2, 10, 5)
        self.camera_yaw = pybullet.addUserDebugParameter("Camera Yaw", -180, 180, 50)
        self.camera_pitch = pybullet.addUserDebugParameter("Camera Pitch", -90, 90, -35)
        self.camera_target_x = pybullet.addUserDebugParameter("Camera Target X", -5, 5, 0.5)
        self.camera_target_y = pybullet.addUserDebugParameter("Camera Target Y", -5, 5, 0.5)
        self.camera_target_z = pybullet.addUserDebugParameter("Camera Target Z", -5, 5, 1.0)

        # Add buttons for play, stop, manual time control, and termination
        self.play_button = pybullet.addUserDebugParameter("Play", 1, 0, 0)
        self.stop_button = pybullet.addUserDebugParameter("Stop", 1, 0, 0)
        self.time_slider = pybullet.addUserDebugParameter("Time Step", 0.001, 0.1, 0.01)
        self.terminate_button = pybullet.addUserDebugParameter("Terminate", 1, 0, 0)

        # Add a marker for the target point
        self.marker_id = None

    def add_ground(self):
        """Add a ground plane with collision and no friction."""
        # Create a plane collision shape
        ground_shape = pybullet.createCollisionShape(pybullet.GEOM_PLANE)

        # Create the ground body
        self.ground_id = pybullet.createMultiBody(
            baseMass=0,  # Static plane
            baseCollisionShapeIndex=ground_shape,
            baseVisualShapeIndex=-1  # No visual shape for the ground
        )

        # Set the friction of the ground to zero
        pybullet.changeDynamics(self.ground_id, -1, lateralFriction=0.0)

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
            collision_shape_id = pybullet.createCollisionShape(pybullet.GEOM_BOX, halfExtents=[d / 2 for d in block_size])
            visual_shape_id = pybullet.createVisualShape(
                pybullet.GEOM_BOX, 
                halfExtents=[d / 2 for d in block_size], 
                rgbaColor=[0.5, 0.5, 0.5, 1]  # Grey color
            )

            # Create the block as a movable object
            block_id = pybullet.createMultiBody(
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
        constraint_id = pybullet.createConstraint(
            parentBodyUniqueId=block_id,
            parentLinkIndex=-1,  # Base link of the block
            childBodyUniqueId=robot_id,
            childLinkIndex=-1,  # Base link of the manipulator
            jointType=pybullet.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0.25],  # Offset at the top of the block
            childFramePosition=[0, 0, 0]       # Bottom of the manipulator
        )
        self.constraints.append(constraint_id)

    def update_marker(self, target_point):
        """Update the marker to indicate the target point."""
        #if self.marker_id is not None:
        #    pybullet.removeBody(self.marker_id)
        #visual_shape_id = pybullet.createVisualShape(pybullet.GEOM_BOX, halfExtents=[0.01 / 2, 0.01 / 2, 2 / 2], rgbaColor=[1, 0, 0, 1])  # Green sphere
        #self.marker_id = pybullet.createMultiBody(baseVisualShapeIndex=visual_shape_id, basePosition=target_point)


        # remove the previous marker
        if self.marker_id is not None:
            pybullet.removeUserDebugItem(self.target_marker_id)        

        # Add a new marker at the target point
        self.target_marker_id = pybullet.addUserDebugText(
            text=".",  
            textPosition=target_point,
            textColorRGB=[1, 0, 0],  # Red color
            textSize=1.5
        )


    def real_time_consensus(self):
        """
        Perform real-time trajectory generation and movement to a consensus point.
        """
        convergence_threshold = 0.01  # Threshold for convergence

        while True:
            # Update the camera view based on user inputs
            cam_dist = pybullet.readUserDebugParameter(self.camera_distance)
            cam_yaw = pybullet.readUserDebugParameter(self.camera_yaw)
            cam_pitch = pybullet.readUserDebugParameter(self.camera_pitch)
            cam_target_x = pybullet.readUserDebugParameter(self.camera_target_x)
            cam_target_y = pybullet.readUserDebugParameter(self.camera_target_y)
            cam_target_z = pybullet.readUserDebugParameter(self.camera_target_z)
            pybullet.resetDebugVisualizerCamera(cam_dist, cam_yaw, cam_pitch, [cam_target_x, cam_target_y, cam_target_z])

            play = pybullet.readUserDebugParameter(self.play_button)
            stop = pybullet.readUserDebugParameter(self.stop_button)
            terminate = pybullet.readUserDebugParameter(self.terminate_button)
            time_step = pybullet.readUserDebugParameter(self.time_slider)

            if terminate > 0:
                print("Simulation terminated.")
                break

            if stop > 0:
                time.sleep(0.01)
                continue

            if play > 0:
                all_converged = True
                for robot_id in self.robot_ids:
                    ee_position = np.array(pybullet.getLinkState(robot_id, self.num_joints - 1)[0])
                    direction = np.array(self.target_position) - ee_position
                    distance = np.linalg.norm(direction)

                    if distance > convergence_threshold:
                        all_converged = False
                        step = direction / distance * min(distance, 1 * distance)  # Small step towards target
                        new_position = ee_position + step

                        # Only care about position, not orientation
                        joint_positions = pybullet.calculateInverseKinematics(
                            robot_id, self.num_joints - 1, new_position
                        )

                        for i in range(self.num_joints):
                            pybullet.setJointMotorControl2(
                                bodyUniqueId=robot_id,
                                jointIndex=i,
                                controlMode=pybullet.POSITION_CONTROL,
                                targetPosition=joint_positions[i]
                            )

                # Move the target point along a circle with radius 0.2 meters
                self.update_target_position()

                # Update the marker to the new target position
                self.update_marker(self.target_position)

                # Update the positions of the blocks
                self.update_blocks_positions()

                self.move_robots_to_blocks()

                pybullet.stepSimulation()
                time.sleep(time_step)

                if all_converged:
                    print("All robots have converged to the target point.")
                    break

    def update_target_position(self):
        """Move the target point smoothly along a circle with radius 0.2 meters."""
        angular_velocity = 0.1  # Angular velocity (radians per second)
        linear_velocity = 0.5  # Linear velocity in the x-direction
        self.angle += angular_velocity * 0.01  # Increment angle by the angular velocity * time step

        # Define the circular trajectory in the X-Y plane, keeping the Z constant at 1.0m
        self.initial_target_position[0] += linear_velocity * 0.01  # Move forward in x
        target_x = self.initial_target_position[0] + self.target_radius * np.cos(self.angle)
        target_y = self.initial_target_position[1] + self.target_radius * np.sin(self.angle)
        target_z = 1.0  # Height of 1m

        self.target_position = np.array([target_x, target_y, target_z])
    
    def update_blocks_positions(self):
        base_radius = 0.75
        for i, block_id in enumerate(self.block_ids):
            angle_offset = i *  np.pi / 2
            block_x = self.target_position[0] + base_radius * np.cos(self.angle + angle_offset)
            block_y = self.target_position[1] + base_radius * np.sin(self.angle + angle_offset)
            pybullet.resetBasePositionAndOrientation(block_id, [block_x, block_y, 0.25], [0, 0, 0, 1])

            # Move the corresponding robot base with the block
            #robot_pos = pybullet.getBasePositionAndOrientation(self.robot_ids[i])[0]
            #pybullet.resetBasePositionAndOrientation(self.robot_ids[i], [block_x, block_y, robot_pos[2]], [0, 0, 0, 1])

        

    # make the base of each robot move on the posision of the block
    def move_robots_to_blocks(self):
        for i, block_id in enumerate(self.block_ids):
            block_pos, _ = pybullet.getBasePositionAndOrientation(block_id)
            pybullet.resetBasePositionAndOrientation(self.robot_ids[i], [block_pos[0], block_pos[1], block_pos[2] + 0.25], [0, 0, 0, 1])

    

    def run_simulation(self):
        """Run the simulation indefinitely."""
        print("Running simulation. Use the GUI to control play, stop, and time step.")
        while True:
            terminate = pybullet.readUserDebugParameter(self.terminate_button)
            if terminate > 0:
                print("Simulation terminated.")
                break
            pybullet.stepSimulation()
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