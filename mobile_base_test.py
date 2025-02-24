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
        self.target_position = np.array([0.5, 0.3, 1.0])  # Target position (height of 1m)
        self.target_velocity = np.array([0.0005, 0.00, 0.0])  # Velocity of the target (movement in XY)
        self.angle = 0  # Initial angle for circular motion
        self.target_radius = 0.2  # Radius of the circle for target movement
        self.ground_id = None  # ID for the ground
        self.target_visual = None  # Visualization ID for the target point

    def initialize_simulation(self):
        """Initialize the PyBullet simulation environment and add robots and blocks."""
        self.physics_client = pybullet.connect(pybullet.GUI)
        pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
        pybullet.setGravity(0, 0, -9.81)

        # Add a ground with collision but no friction
        self.add_ground()

        # Define the positions of the blocks
        block_positions = [
            [0, 0, 0.25],        # Block 1
            [1, 0, 0.25],        # Block 2
            [0, 1, 0.25],        # Block 3
            [1, 1, 0.25]         # Block 4
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

        # Add the constraints to make the robots move with the blocks
        self.create_planar_constraints()

        # Add the target point visual marker
        self.add_target_visual_marker()

    def add_ground(self):
        """Add a ground plane with collision and no friction."""
        ground_shape = pybullet.createCollisionShape(pybullet.GEOM_PLANE)
        self.ground_id = pybullet.createMultiBody(
            baseMass=0,  # Static plane
            baseCollisionShapeIndex=ground_shape,
            baseVisualShapeIndex=-1  # No visual shape for the ground
        )
        pybullet.changeDynamics(self.ground_id, -1, lateralFriction=0.0)

    def add_movable_blocks(self, positions):
        """Add movable blocks at specified positions."""
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
                baseMass=block_mass,
                baseCollisionShapeIndex=collision_shape_id,
                baseVisualShapeIndex=visual_shape_id,
                basePosition=pos
            )
            self.block_ids.append(block_id)

    def create_planar_constraints(self):
        """Create a planar joint for each block to allow movement in the X-Y plane."""
        for block_id in self.block_ids:
            constraint_id = pybullet.createConstraint(
                parentBodyUniqueId=block_id,
                parentLinkIndex=-1,  # Base link of the block
                childBodyUniqueId=block_id,
                childLinkIndex=-1,  # Base link of the manipulator
                jointType=pybullet.JOINT_PRISMATIC,
                jointAxis=[1, 0, 0],  # Planar motion along the X-axis
                parentFramePosition=[0, 0, 0.25],  # Offset at the top of the block
                childFramePosition=[0, 0, 0]  # Bottom of the manipulator
            )
            self.constraints.append(constraint_id)
            
            # Adding a second constraint for motion along the Y-axis
            constraint_id_y = pybullet.createConstraint(
                parentBodyUniqueId=block_id,
                parentLinkIndex=-1,
                childBodyUniqueId=block_id,
                childLinkIndex=-1,
                jointType=pybullet.JOINT_PRISMATIC,
                jointAxis=[0, 1, 0],  # Planar motion along the Y-axis
                parentFramePosition=[0, 0, 0.25],
                childFramePosition=[0, 0, 0]
            )
            self.constraints.append(constraint_id_y)

    def add_target_visual_marker(self):
        """Add a visual marker for the target point (just a sphere)"""
        target_radius = 0.05  # Small radius for the target visualization
        target_visual = pybullet.createVisualShape(
            shapeType=pybullet.GEOM_SPHERE,
            radius=target_radius,
            rgbaColor=[1, 0, 0, 1]  # Red color for the target
        )
        self.target_visual = pybullet.createMultiBody(
            baseMass=0,  # No mass, just for visualization
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=target_visual,
            basePosition=self.target_position.tolist()
        )

    def move_target_in_xy(self):
        """Move the target point in the XY plane."""
        self.target_position += self.target_velocity  # Update position in XY
        pybullet.resetBasePositionAndOrientation(self.target_visual, self.target_position.tolist(), [0, 0, 0, 1])

    def move_blocks_on_plane(self):
        """Move blocks in a circular formation around the moving target."""
        velocity = 0.1  # Movement speed
        time_step = 0.01  # Simulation time step
        k_p = 0.2  # Proportional control gain
        r = 1  # Radius of formation

        while True:
            # Update the target movement
            self.move_target_in_xy()

            # Compute center of mass of the blocks
            com_x = np.mean([pybullet.getBasePositionAndOrientation(block_id)[0][0] for block_id in self.block_ids])
            com_y = np.mean([pybullet.getBasePositionAndOrientation(block_id)[0][1] for block_id in self.block_ids])

            # Compute desired positions for the blocks
            for i, block_id in enumerate(self.block_ids):
                theta_i = i * (np.pi / 2)  # Phase difference for each block
                theta_t = np.arctan2(self.target_position[1] - com_y, self.target_position[0] - com_x)

                x_des = self.target_position[0] + r * np.cos(theta_t + theta_i)
                y_des = self.target_position[1] + r * np.sin(theta_t + theta_i)

                # Get current position
                block_pos, _ = pybullet.getBasePositionAndOrientation(block_id)

                # Compute velocity using proportional control
                new_x = block_pos[0] + k_p * (x_des - block_pos[0]) * time_step
                new_y = block_pos[1] + k_p * (y_des - block_pos[1]) * time_step

                # Update the block's position
                pybullet.resetBasePositionAndOrientation(block_id, [new_x, new_y, block_pos[2]], [0, 0, 0, 1])

                # Move the corresponding robot base with the block
                robot_pos = pybullet.getBasePositionAndOrientation(self.robot_ids[i])[0]
                pybullet.resetBasePositionAndOrientation(self.robot_ids[i], [new_x, new_y, robot_pos[2]], [0, 0, 0, 1])

            pybullet.stepSimulation()
            time.sleep(time_step)

    def move_blocks_on_plane_old(self):
        """Move blocks along a defined path on the X-Y plane."""
        velocity = 0.5  # Movement velocity
        time_step = 0.01  # Time step for the simulation

        while True:
            for i, block_id in enumerate(self.block_ids):
                # Calculate new position based on velocity (move along X and Y)
                block_pos, block_orientation = pybullet.getBasePositionAndOrientation(block_id)
                new_pos = [block_pos[0] + velocity * time_step, block_pos[1], block_pos[2]]

                # Update the block's position
                pybullet.resetBasePositionAndOrientation(block_id, new_pos, [0, 0, 0, 1])

                # Move the corresponding robot base with the block
                robot_pos = pybullet.getBasePositionAndOrientation(self.robot_ids[i])[0]
                pybullet.resetBasePositionAndOrientation(self.robot_ids[i], [new_pos[0], new_pos[1], robot_pos[2]], [0, 0, 0, 1])

            # Move the target point
            self.move_target_in_xy()

            pybullet.stepSimulation()
            time.sleep(time_step)


    def run_simulation(self):
        """Run the simulation indefinitely."""
        print("Running simulation. Blocks will move on the X-Y plane and target will move in XY.")
        self.move_blocks_on_plane()


# Main execution file (main.py)
if __name__ == "__main__":
    urdf_path = "kuka_iiwa/model.urdf"  # Update with the actual URDF path
    twin = KukaIiwa7DigitalTwin(urdf_path)
    twin.initialize_simulation()

    # Run the simulation where the blocks move on a plane
    twin.run_simulation()
