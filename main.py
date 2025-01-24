import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import networkx as nx
import time
from matplotlib.patches import Rectangle

# Initialize parameters
n_agents = 4  # Number of robots
space_dim = 2  # 2D space
duration = 20  # Duration of simulation in seconds
timestep = 0.1  # Time step in seconds (increased to slow down agents)
iterations = int(duration / timestep)  # Total iterations
boundary = [-2, 2]  # Square boundary (-2 to 2)

# Initial positions of robots (random within the square)
np.random.seed()  # For randomization of initial positions
initial_positions = np.random.rand(n_agents, space_dim) * (boundary[1] - boundary[0]) + boundary[0]
positions = initial_positions.copy()
trajectory = [positions.copy()]  # Store positions over time

# Define target positions for each agent (random within the square)
target_positions = np.random.rand(n_agents, space_dim) * (boundary[1] - boundary[0]) + boundary[0]

# Define the communication graph
G = nx.Graph()
G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])  # A cyclic graph
weights = {edge: 0.1 for edge in G.edges()}  # Uniform weights

# Consensus algorithm with boundary enforcement and target positions
def consensus_step(positions, G, weights, boundary, target_positions):
    new_positions = positions.copy()
    for i in range(len(positions)):
        neighbors = list(G.neighbors(i))
        weighted_sum = positions[i]
        total_weight = 1  # Include the agent's own weight
        for neighbor in neighbors:
            weight = weights.get((i, neighbor), weights.get((neighbor, i), 0))
            weighted_sum += weight * positions[neighbor]
            total_weight += weight
        # Move towards target p"?
        # sition
        target_weight = 0.2  # Increase this weight to give more influence to the target position
        weighted_sum += target_weight * target_positions[i]
        total_weight += target_weight
        new_positions[i] = weighted_sum / total_weight
        # Enforce boundary conditions
        new_positions[i] = np.clip(new_positions[i], boundary[0], boundary[1])
    return new_positions

# Simulate and record all positions
for t in range(iterations):
    positions = consensus_step(positions, G, weights, boundary, target_positions)
    trajectory.append(positions.copy())

# Visualization with slider and play button
fig, ax = plt.subplots(figsize=(8, 6))
plt.subplots_adjust(bottom=0.3)

# Draw boundary
boundary_rect = Rectangle((boundary[0], boundary[0]), boundary[1] - boundary[0], boundary[1] - boundary[0],
                          linewidth=2, edgecolor='blue', facecolor='none')
ax.add_patch(boundary_rect)

# Initial plot setup
sc = ax.scatter(trajectory[0][:, 0], trajectory[0][:, 1], c='red', label='Robot Positions')
initial_sc = ax.scatter(initial_positions[:, 0], initial_positions[:, 1], c='green', marker='x', label='Initial Positions')
final_sc = ax.scatter(trajectory[-1][:, 0], trajectory[-1][:, 1], c='purple', marker='s', label='Target Positions')
labels = [
    ax.text(trajectory[0][i, 0], trajectory[0][i, 1], f"{i}", fontsize=10, color='black', ha='center', va='center')
    for i in range(n_agents)
]
lines = [
    ax.plot([], [], lw=1, color='gray')[0] for _ in range(n_agents)
]  # Initialize trajectory lines
time_text = ax.text(0.02, 1.05, f"Time: 0.0s", transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# Fixed plot bounds
ax.set_xlim(boundary[0], boundary[1])
ax.set_ylim(boundary[0], boundary[1])
ax.legend()

# Show grid
ax.grid(True)

# Add slider
ax_slider = plt.axes([0.2, 0.2, 0.6, 0.03])
time_slider = Slider(ax_slider, 'Time', 0, duration, valinit=0, valstep=timestep)

# Add play button
ax_button = plt.axes([0.8, 0.05, 0.1, 0.075])
play_button = Button(ax_button, 'Play')

# Globals for play functionality
is_playing = False

# Update function for slider
def update(val):
    t = int(val / timestep)
    sc.set_offsets(trajectory[t])
    for i, label in enumerate(labels):
        label.set_position((trajectory[t][i, 0], trajectory[t][i, 1]))
    for i, line in enumerate(lines):
        line.set_data([traj[i, 0] for traj in trajectory[:t + 1]],
                      [traj[i, 1] for traj in trajectory[:t + 1]])  # Update trajectory
    time_text.set_text(f"Time: {val:.1f}s")
    fig.canvas.draw_idle()

time_slider.on_changed(update)

# Play button callback
def play(event):
    global is_playing
    is_playing = not is_playing
    if is_playing:
        play_button.label.set_text("Pause")
        start_time = time.time()
        current_time = time_slider.val
        while is_playing and current_time < duration:
            elapsed_time = time.time() - start_time
            current_time = min(current_time + elapsed_time, duration)
            time_slider.set_val(current_time)
            start_time = time.time()  # Reset start time to avoid cumulative lag
            plt.pause(timestep)
    else:
        play_button.label.set_text("Play")

play_button.on_clicked(play)

plt.show()
