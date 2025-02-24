import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp

# Define the 2-DOF planar manipulator parameters
l1, l2 = 1.0, 1.0  # Link lengths

def forward_kinematics(q):
    """ Compute end-effector position given joint angles q."""
    x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
    y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    return np.array([x, y])

def jacobian(q):
    """ Compute the Jacobian matrix of the 2-DOF arm."""
    J = np.array([[-l1*np.sin(q[0]) - l2*np.sin(q[0] + q[1]), -l2*np.sin(q[0] + q[1])],
                  [ l1*np.cos(q[0]) + l2*np.cos(q[0] + q[1]),  l2*np.cos(q[0] + q[1])]])
    return J

def min_singular_value(J):
    """ Compute the smallest singular value of J."""
    return np.min(np.linalg.svd(J, compute_uv=False))

def cbf_qp(q, v_d, sigma_safe=0.1, gamma=1.0):
    """ Solve the CBF-QP for singularity avoidance."""
    J = jacobian(q)
    sigma_min = min_singular_value(J)
    
    # Compute gradient of singular value w.r.t. q (approximation)
    eps = 1e-5
    grad_sigma = np.zeros(2)
    for i in range(2):
        dq = np.zeros(2)
        dq[i] = eps
        grad_sigma[i] = (min_singular_value(jacobian(q + dq)) - sigma_min) / eps
    
    # Define optimization variables
    dq = cp.Variable(2)
    
    # Objective: minimize tracking error ||J dq - v_d||^2
    objective = cp.Minimize(cp.norm(J @ dq - v_d, 2))
    
    # CBF constraint: grad_sigma @ dq + gamma * (sigma_min - sigma_safe) >= 0
    constraints = [grad_sigma @ dq + gamma * (sigma_min - sigma_safe) >= 0]
    
    # Solve QP
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return dq.value if prob.status == cp.OPTIMAL else np.zeros(2)

# Simulation parameters
T = 50  # Simulation steps
dt = 0.1
q = np.array([0.5, -0.5])  # Initial joint configuration
traj = []

# Desired end-effector velocity (constant)
v_d = np.array([0.1, 0.0])

# Simulate
for t in range(T):
    traj.append(forward_kinematics(q))
    dq = cbf_qp(q, v_d)
    q += dq * dt  # Update joint positions

# Plot trajectory
traj = np.array(traj)
plt.plot(traj[:, 0], traj[:, 1], 'b-', label='End-effector trajectory')
plt.scatter(traj[0, 0], traj[0, 1], c='g', marker='o', label='Start')
plt.scatter(traj[-1, 0], traj[-1, 1], c='r', marker='x', label='End')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("2-DOF Manipulator with CBF-based Singularity Avoidance")
plt.legend()
plt.grid()
plt.show()
