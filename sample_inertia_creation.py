import pybullet as p
import pybullet_data
import time

# Connect to PyBullet and set up the environment
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load a simple plane
plane_id = p.loadURDF("plane.urdf")

# Define the block properties
block_mass = 1.0
block_size = [0.5, 0.5, 0.5]
block_position = [0, 0, 0.5]
block_orientation = [0, 0, 0, 1]  # Quaternion

# Create collision and visual shapes for the block
block_collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[s / 2 for s in block_size])
block_visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in block_size], rgbaColor=[0.8, 0.3, 0.3, 1])

# Create the block body
block_body_id = p.createMultiBody(
    baseMass=block_mass,
    baseCollisionShapeIndex=block_collision_shape,
    baseVisualShapeIndex=block_visual_shape,
    basePosition=block_position,
    baseOrientation=block_orientation
)

# Enable the default sidebar and set the initial camera view
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
camera_distance = 2.5
camera_yaw = 50
camera_pitch = -35
camera_target_position = [0.5, 0.5, 0]
p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=camera_yaw, cameraPitch=camera_pitch, cameraTargetPosition=camera_target_position)

# Function to update the camera view
def update_camera():
    p.resetDebugVisualizerCamera(cameraDistance=camera_distance, cameraYaw=camera_yaw, cameraPitch=camera_pitch, cameraTargetPosition=camera_target_position)

# Run the simulation with smooth keyboard controls for the camera
while True:
    keys = p.getKeyboardEvents()
    
    if ord('w') in keys and keys[ord('w')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
        camera_pitch += 0.1
    if ord('s') in keys and keys[ord('s')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
        camera_pitch -= 0.1
    if ord('a') in keys and keys[ord('a')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
        camera_yaw -= 0.1
    if ord('d') in keys and keys[ord('d')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
        camera_yaw += 0.1
    if ord('q') in keys and keys[ord('q')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
        camera_distance += 0.01
    if ord('e') in keys and keys[ord('e')] & (p.KEY_IS_DOWN | p.KEY_WAS_TRIGGERED):
        camera_distance -= 0.01

    update_camera()
    p.stepSimulation()
    time.sleep(0.01)