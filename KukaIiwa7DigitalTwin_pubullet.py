import pybullet as p
import pybullet_data
import numpy as np
import time
import json
import threading

class KukaIiwa7DigitalTwin:
    def __init__(self, urdf_path):
        """
        Initialize the digital twin for the KUKA iiwa 7 robotic manipulator.

        Parameters:
        - urdf_path: Path to the URDF file.
        """
        self.urdf_path = urdf_path
        self.physics_client = None
        self.robot_id = None

    def initialize_simulation(self):
        """Initialize the PyBullet simulation environment."""
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.robot_id = p.loadURDF(self.urdf_path, useFixedBase=True)
        self.num_joints = p.getNumJoints(self.robot_id)

    def forward_kinematics(self):
        """Compute the forward kinematics of the robot."""
        joint_states = p.getJointStates(self.robot_id, range(self.num_joints))
        joint_positions = [state[0] for state in joint_states]

        transforms = []
        for i in range(self.num_joints):
            link_state = p.getLinkState(self.robot_id, i, computeForwardKinematics=True)
            transforms.append(link_state[4])  # World position of the link

        return transforms

    def set_joint_trajectory(self, trajectory, time_step=0.01):
        """
        Execute a joint trajectory.

        Parameters:
        - trajectory: List of joint configurations over time.
        - time_step: Time step between trajectory points (default: 0.01 seconds).
        """
        for q_target in trajectory:
            for i in range(self.num_joints):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=q_target[i]
                )
            p.stepSimulation()
            time.sleep(time_step)

    def set_end_effector_trajectory(self, trajectory, time_step=0.01):
        """
        Execute an end-effector trajectory using inverse kinematics.

        Parameters:
        - trajectory: List of end-effector poses (positions and orientations).
        - time_step: Time step between trajectory points (default: 0.01 seconds).
        """
        for pose_target in trajectory:
            position, orientation = pose_target[:3], pose_target[3:]
            joint_positions = p.calculateInverseKinematics(
                self.robot_id, self.num_joints - 1, position, orientation
            )

            for i in range(self.num_joints):
                p.setJointMotorControl2(
                    bodyUniqueId=self.robot_id,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=joint_positions[i]
                )
            p.stepSimulation()
            time.sleep(time_step)

    def apply_torque_control(self, torques, time_step=0.01):
        """
        Apply torque control to the robot.

        Parameters:
        - torques: Joint torques to apply.
        - time_step: Simulation time step (default: 0.01 seconds).
        """
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                bodyUniqueId=self.robot_id,
                jointIndex=i,
                controlMode=p.TORQUE_CONTROL,
                force=torques[i]
            )
        p.stepSimulation()
        time.sleep(time_step)

    def start_control_server(self, host="localhost", port=5000):
        """
        Start a control server to receive external commands.

        Parameters:
        - host: Hostname for the server (default: "localhost").
        - port: Port for the server (default: 5000).
        """
        def handle_client(client_socket):
            while True:
                try:
                    data = client_socket.recv(1024)
                    if not data:
                        break

                    command = json.loads(data.decode("utf-8"))
                    if command["type"] == "joint_trajectory":
                        self.set_joint_trajectory(command["trajectory"])
                    elif command["type"] == "end_effector_trajectory":
                        self.set_end_effector_trajectory(command["trajectory"])
                    elif command["type"] == "torque":
                        self.apply_torque_control(command["torques"])
                except Exception as e:
                    print(f"Error handling client: {e}")
                    break

            client_socket.close()

        import socket

        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind((host, port))
        server.listen(1)
        print(f"Control server started on {host}:{port}")

        while True:
            client_socket, addr = server.accept()
            print(f"Connection from {addr}")
            threading.Thread(target=handle_client, args=(client_socket,)).start()

# Main execution file (main.py)
if __name__ == "__main__":
    urdf_path = "kuka_iiwa/model.urdf"  # Update with the actual URDF path
    twin = KukaIiwa7DigitalTwin(urdf_path)
    twin.initialize_simulation()

    # Example usage: Set joint trajectory
    trajectory = [np.zeros(7), np.ones(7) * 0.5]
    twin.set_joint_trajectory(trajectory, time_step=0.05)

    # Example usage: Start control server
    threading.Thread(target=twin.start_control_server, daemon=True).start()

    # Keep the simulation running
    while True:
        time.sleep(1)
