from packet_handler import*
import socket
import threading
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib
matplotlib.use('qt5agg')  # Or 'qt5agg', 'wxagg', depending on your system
import matplotlib.pyplot as plt
from scipy.spatial import distance
handler = PacketHandler()

connect = False

packet_dispatcher = PacketDispatcher()
data = packet_dispatcher.register_DataPacket_callback(lambda x: print(x))
if connect:
    valid = True
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("mainrobotTeensy.local", 80))
    except socket.error as msg:
        print(msg)
        valid = False

    def packet_handler(c_client):
        disconnected = False
        while c_client and not disconnected:
            data = c_client.recv(1024)
            if not data or len(data) == 0:
                disconnected = True
            else:
                handler.receive_data(data)
                [a, b] = handler.check_packet()
                while a == EXECUTED_PACKET:
                    packet_dispatcher.dispatch_packet(b)
                    [a, b] = handler.check_packet()


    if(valid):
        t = threading.Thread(target=packet_handler, args=(client,))
        t.start()

print("Hello world")

# --- STEP 1: Tangential potential field planner ---
class TangentialPotentialFieldPlanner:
    def __init__(self, goal, obstacles, k_att=1.0, k_rep=100.0, rho_0=1.0, k_tan=1.0):
        self.goal = np.array(goal)
        self.obstacles = obstacles  # List of dicts: {"position": (x, y), "multiplier": float}
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.k_tan = k_tan

    def attractive_force(self, q):
        return -self.k_att * (q - self.goal)

    def repulsive_force(self, q):
        force = np.zeros(2)
        for obs in self.obstacles:
            obs_pos = np.array(obs["position"])
            multiplier = obs.get("multiplier", 1.0)

            delta = q - obs_pos
            rho = np.linalg.norm(delta)
            if 1e-5 < rho < self.rho_0:
                grad_rho = delta / rho
                rep_force = self.k_rep * (1/rho - 1/self.rho_0) * (1/rho**2) * grad_rho
                perp = np.array([-grad_rho[1], grad_rho[0]])
                tan_force = self.k_tan * (1 / rho) * perp
                force += multiplier * (rep_force + tan_force)
        return force

    def total_force(self, q):
        return self.attractive_force(q) + self.repulsive_force(q)

# --- STEP 2: Simulate path with strong perpendicular escape ---
def simulate_path_with_strong_perpendicular_escape(planner, start, step_size=0.2, max_steps=300, threshold=0.1, min_noise_ratio=1.5, stuck_limit=5):
    path = [start]
    q = np.array(start[:2])
    stuck_counter = 0

    for _ in range(max_steps):
        F = planner.total_force(q)
        norm = np.linalg.norm(F)

        if norm < threshold:
            stuck_counter += 1
        else:
            stuck_counter = 0

        if stuck_counter >= stuck_limit:
            if norm > 1e-5:
                direction = F / norm
                perp = np.array([-direction[1], direction[0]])
                noise_magnitude = min_noise_ratio * norm
                F += perp * noise_magnitude * (2 * np.random.rand() - 1)
            else:
                F = np.random.randn(2)
            norm = np.linalg.norm(F)

        direction = F / norm if norm > 1e-5 else np.zeros_like(F)
        q_next = q + step_size * direction
        theta = np.arctan2(direction[1], direction[0])
        path.append((q_next[0], q_next[1], theta))
        q = q_next

        if np.linalg.norm(q - planner.goal) < 0.3:
            break

    return path

# --- STEP 3: Custom RDP simplification with heading ---
def rdp_custom(points, epsilon):
    def point_line_distance(point, start, end):
        if np.allclose(start, end):
            return np.linalg.norm(point - start)
        return np.abs(np.cross(end - start, start - point)) / np.linalg.norm(end - start)

    def rdp_rec(start_idx, end_idx):
        dmax = 0.0
        index = start_idx
        for i in range(start_idx + 1, end_idx):
            d = point_line_distance(points[i], points[start_idx], points[end_idx])
            if d > dmax:
                index = i
                dmax = d
        if dmax >= epsilon:
            results1 = rdp_rec(start_idx, index)
            results2 = rdp_rec(index, end_idx)
            return results1[:-1] + results2
        else:
            return [points[start_idx], points[end_idx]]

    return rdp_rec(0, len(points) - 1)

def simplify_path_rdp_manual_with_heading(path, epsilon=0.3):
    points = np.array([[p[0], p[1]] for p in path])
    simplified_pts = rdp_custom(points, epsilon)

    simplified_path = []
    for i in range(len(simplified_pts)):
        x, y = simplified_pts[i]
        if i < len(simplified_pts) - 1:
            dx, dy = simplified_pts[i + 1] - simplified_pts[i]
        else:
            dx, dy = simplified_pts[i] - simplified_pts[i - 1]
        theta = np.arctan2(dy, dx)
        simplified_path.append((x, y, theta))
    return simplified_path

# --- STEP 4: Length computation ---
def compute_path_length(path):
    return sum(distance.euclidean(path[i][:2], path[i+1][:2]) for i in range(len(path)-1))

# --- Usage Example ---
goal = (8, 8)
obstacles = [
    {"position": (3, 3), "multiplier": 2.0},
    {"position": (6, 4), "multiplier": 0.5},
    {"position": (4, 7), "multiplier": 1.5}
]
start_pose = np.array((0.5, 0.5, 0.0))

planner = TangentialPotentialFieldPlanner(goal, obstacles, k_att=1.0, k_rep=200.0, rho_0=2.0, k_tan=2.5)
raw_path = simulate_path_with_strong_perpendicular_escape(planner, start_pose)
simplified_path = simplify_path_rdp_manual_with_heading(raw_path, epsilon=0.3)

# `simplified_path` now contains a clean sequence of (x, y, theta) poses ready for clothoid fitting.
def plot_potential_field_path_comparison(raw_path, simplified_path, goal, obstacles, title="Path Simplification Visualization"):
    x_raw = [p[0] for p in raw_path]
    y_raw = [p[1] for p in raw_path]
    x_simp = [p[0] for p in simplified_path]
    y_simp = [p[1] for p in simplified_path]

    plt.figure(figsize=(8, 8))

    # Plot obstacles
    for obs in obstacles:
        pos = obs["position"]
        multiplier = obs.get("multiplier", 1.0)
        circle = plt.Circle(pos, 0.2 * multiplier, color='red', alpha=0.5)
        plt.gca().add_patch(circle)
        plt.plot(pos[0], pos[1], 'rx')

    # Plot raw path
    plt.plot(x_raw, y_raw, 'gray', label=f'Raw Path ({compute_path_length(raw_path):.2f}m)')

    # Plot simplified path
    plt.plot(x_simp, y_simp, 'r.-', label=f'Simplified Path ({compute_path_length(simplified_path):.2f}m)')

    # Plot headings on simplified path
    for x, y, theta in simplified_path:
        plt.arrow(x, y, 0.3 * np.cos(theta), 0.3 * np.sin(theta), head_width=0.1, color='black')

    # Start and goal
    plt.plot(raw_path[0][0], raw_path[0][1], 'bo', label='Start')
    plt.plot(goal[0], goal[1], 'go', label='Goal')

    plt.title(title)
    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.gca().set_aspect('equal')
    plt.grid(True)
    plt.legend()
    plt.show()

# Use the plotter with current paths
plot_potential_field_path_comparison(
    raw_path=raw_path,
    simplified_path=simplified_path,
    goal=goal,
    obstacles=obstacles
)
import heapq

def astar_on_potential_field(planner, start, goal, grid_resolution=0.2, max_iters=5000):
    """
    Performs A* over a discretized grid where cost is based on potential field.
    """
    start = np.array(start[:2])
    goal = np.array(goal)
    start_node = tuple(np.round(start[:2] / grid_resolution).astype(int))
    goal_node = tuple(np.round(np.array(goal) / grid_resolution).astype(int))

    open_set = []
    heapq.heappush(open_set, (0, start_node))
    came_from = {}
    cost_so_far = {start_node: 0}

    def neighbors(node):
        x, y = node
        return [(x + dx, y + dy) for dx in [-1, 0, 1] for dy in [-1, 0, 1] if (dx != 0 or dy != 0)]

    def heuristic(a, b):
        return np.linalg.norm((np.array(a) - np.array(b)) * grid_resolution)

    def potential_cost(node):
        pos = np.array(node) * grid_resolution
        F = planner.total_force(pos)
        return np.linalg.norm(F)

    for _ in range(max_iters):
        if not open_set:
            break
        _, current = heapq.heappop(open_set)

        if current == goal_node:
            break

        for neighbor in neighbors(current):
            pos = np.array(neighbor) * grid_resolution
            move_cost = np.linalg.norm(np.array(neighbor) - np.array(current)) * grid_resolution
            pot_cost = potential_cost(neighbor)
            new_cost = cost_so_far[current] + move_cost + 0.5 * pot_cost  # Weighted combination

            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal_node)
                heapq.heappush(open_set, (priority, neighbor))
                came_from[neighbor] = current

    # Reconstruct path
    path = []
    current = goal_node
    while current != start_node:
        pos = np.array(current) * grid_resolution
        path.append((pos[0], pos[1]))
        current = came_from.get(current)
        if current is None:
            break
    path.append(np.array(start_node) * grid_resolution)
    path.reverse()
    return path
from mpl_toolkits.mplot3d import Axes3D

def compute_potential_field(planner, x_range, y_range, resolution=0.1):
    X, Y = np.meshgrid(
        np.arange(x_range[0], x_range[1], resolution),
        np.arange(y_range[0], y_range[1], resolution)
    )
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            q = np.array([X[i, j], Y[i, j]])
            # Attractive potential
            U_att = 0.5 * planner.k_att * np.linalg.norm(q - planner.goal) ** 2

            # Repulsive potential
            U_rep = 0.0
            for obs in planner.obstacles:
                obs_pos = np.array(obs["position"])
                multiplier = obs.get("multiplier", 1.0)
                delta = q - obs_pos
                rho = np.linalg.norm(delta)
                if rho < planner.rho_0 and rho > 1e-5:
                    U_rep += multiplier * 0.5 * planner.k_rep * (1/rho - 1/planner.rho_0) ** 2

            Z[i, j] = U_att + U_rep
    return X, Y, Z

def plot_potential_surface(planner, x_range=(0, 10), y_range=(0, 10), resolution=0.1):
    X, Y, Z = compute_potential_field(planner, x_range, y_range, resolution)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)

    # Mark obstacles and goal
    for obs in planner.obstacles:
        ox, oy = obs["position"]
        ax.scatter(ox, oy, 0, color='red', s=50)
    ax.scatter(planner.goal[0], planner.goal[1], 0, color='green', s=60)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Potential")
    ax.set_title("Artificial Potential Field Surface")
    plt.tight_layout()
    plt.show()

# Run A* over the potential field
astar_path = astar_on_potential_field(planner, start_pose, goal)

# Plot A* result
x_astar = [p[0] for p in astar_path]
y_astar = [p[1] for p in astar_path]

plt.figure(figsize=(8, 8))
plt.plot(x_astar, y_astar, 'r.-', label='A* Path on Potential Field')
plt.plot(goal[0], goal[1], 'go', label='Goal')
plt.plot(start_pose[0], start_pose[1], 'bo', label='Start')

# Obstacles
for obs in obstacles:
    pos = obs["position"]
    multiplier = obs.get("multiplier", 1.0)
    circle = plt.Circle(pos, 0.2 * multiplier, color='red', alpha=0.5)
    plt.gca().add_patch(circle)
    plt.plot(pos[0], pos[1], 'rx')

plt.title("A* Navigation on Potential Field")
plt.grid(True)
plt.xlim(0, 10)
plt.ylim(0, 10)
plt.legend()
plt.gca().set_aspect('equal')
plt.show()
plot_potential_surface(planner)