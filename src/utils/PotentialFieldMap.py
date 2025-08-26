import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.utils.HarmonicSolver import HarmonicSolver
from src.utils.path_following import get_parametrized_path, get_spline_for_parametrized_path, fit_clothoids, \
    curve_list_to_path


class PotentialFieldMap:
    def __init__(self, width_m, height_m, resolution_m):
        self.width = width_m
        self.height = height_m
        self.resolution = resolution_m
        self.nx = int(width_m / resolution_m)
        self.ny = int(height_m / resolution_m)
        self.grid = np.ones((self.ny, self.nx))  # Initialize with 1s (boundary condition)
        self.mask = np.zeros_like(self.grid, dtype=bool)  # True for obstacles
        self.goal = None

    def set_goal(self, x_m, y_m):
        i, j = self._to_grid_coords(x_m, y_m)
        self.grid[i, j] = 0  # Potential at goal = 0
        self.goal = (i, j)

    def add_obstacle(self, x_m, y_m, radius_m):
        i_center, j_center = self._to_grid_coords(x_m, y_m)
        r = int(radius_m / self.resolution)
        for i in range(max(0, i_center - r), min(self.ny, i_center + r)):
            for j in range(max(0, j_center - r), min(self.nx, j_center + r)):
                if np.linalg.norm([i - i_center, j - j_center]) <= r:
                    self.mask[i, j] = True
    def add_border(self):
        self.mask[:, 0] = True
        self.mask[:, -1] = True
        self.mask[0, :] = True
        self.mask[-1, :] = True


    def _to_grid_coords(self, x_m, y_m):
        return int(y_m / self.resolution), int(x_m / self.resolution)

    def visualize_powered(self):
        X, Y = np.meshgrid(
            np.linspace(0, self.width, self.nx),
            np.linspace(0, self.height, self.ny)
        )
        Z = 1-np.power(1 - self.grid, 0.3)  # enhances low values

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
        ax.view_init(elev=45, azim=-120)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Potential with (1-(1-z)^0.3)')
        ax.set_title('Harmonic Potential Field')

        fig.colorbar(surf, shrink=0.5, aspect=10, label='Log Potential')
        plt.tight_layout()

    def visualize(self):
        X, Y = np.meshgrid(
            np.linspace(0, self.width, self.nx),
            np.linspace(0, self.height, self.ny)
        )
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, self.grid, cmap='viridis')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Potential')
        ax.set_title('Harmonic Potential Field')

    def get_path(self, start_xy, step_size=0.05, max_steps=10000, stop_thresh=0.05):
        path = [start_xy]
        x, y = start_xy
        ny, nx = self.grid.shape
        res = self.resolution

        for _ in range(max_steps):
            i, j = int(y / res), int(x / res)

            # Stop if out of bounds or close to goal
            if i <= 1 or j <= 1 or i >= ny - 2 or j >= nx - 2:
                break
            if self.goal and np.linalg.norm(np.array((i, j)) - np.array(self.goal)) < 2:
                break

            # Compute gradient using central differences
            dU_dx = (self.grid[i, j + 1] - self.grid[i, j - 1]) / (2 * res)
            dU_dy = (self.grid[i + 1, j] - self.grid[i - 1, j]) / (2 * res)

            grad = np.array([dU_dx, dU_dy])
            norm = np.linalg.norm(grad)
            if norm < 1e-10:
                break

            # Move opposite to the gradient
            delta = -step_size * grad / norm
            x += delta[0]
            y += delta[1]
            path.append((x, y))
        return path
    def get_as_curve_list(self, start_xy, step_size=0.05, max_steps=10000, stop_thresh=0.05, s=0.001, step=0.1):
        path = self.get_path(start_xy, step_size, max_steps, stop_thresh)
        param_path = get_parametrized_path(path)
        x_k,y_k = get_spline_for_parametrized_path(param_path, s=s)
        curve_list= fit_clothoids(x_k, y_k, param_path, step)
        clothoid_path = curve_list_to_path(curve_list)
        return curve_list, clothoid_path
    def solve(self, omega=1.7, max_iter=500, tol=1e-10):
        solver = HarmonicSolver(self)
        solver.solve_SOR(omega=omega, max_iter=max_iter, tol=tol)


