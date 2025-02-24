import pybullet as p
import pybullet_data
import time

# Connect to PyBullet and set up the environment
physics_client = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Load a simple plane and a robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("kuka_iiwa/model.urdf", basePosition=[0, 0, 0], useFixedBase=True)

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