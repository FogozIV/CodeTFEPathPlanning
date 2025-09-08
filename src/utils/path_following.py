from math import sqrt

import numpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import splrep, splev
from curve_library import *


def get_parametrized_path(path):
    parametrized_path = [(path[0][0], path[0][1], 0.0)]
    for i in range(1, len(path)):
        parametrized_path.append((path[i][0], path[i][1], parametrized_path[-1][2] + sqrt(
            (path[i][0] - path[i - 1][0]) ** 2 + (path[i][1] - path[i - 1][1]) ** 2)))
    return parametrized_path

def get_spline_for_parametrized_path(parametrized_path, s=0):
    tck_x = splrep([p[2] for p in parametrized_path], [p[0] for p in parametrized_path], s=s)
    tck_y = splrep([p[2] for p in parametrized_path], [p[1] for p in parametrized_path], s=s)
    return tck_x, tck_y

def show_spline_path(tck_x, tck_y, parametrized_path):
    # --- 1. Generate query points along the spline ---
    s_values = np.linspace(parametrized_path[0][2], parametrized_path[-1][2], int((parametrized_path[-1][2] - parametrized_path[0][2])/0.01))
    # --- 2. Evaluate the spline ---
    x_spline = splev(s_values, tck_x)
    y_spline = splev(s_values, tck_y)

    # --- 3. Extract your original discrete points for comparison ---
    x_points = [p[0] for p in parametrized_path]
    y_points = [p[1] for p in parametrized_path]

    # --- 4. Plot ---
    plt.figure(figsize=(10, 5))
    plt.plot(x_spline, y_spline, label="Spline fit", linewidth=2)
    plt.scatter(x_points, y_points, color='red', label="Original points", s=30)
    plt.axis("equal")
    plt.legend()
    plt.title("Path smoothed with B-spline")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")


def get_orientation_and_curvature(tck_x, tck_y, s_query):
    x_s = splev(s_query, tck_x, der=1) #if not type(s_query) is numpy.ndarray else [splev(s_query[i], tck_x[i], der=1) for i in range(len(s_query))]
    y_s = splev(s_query, tck_y, der=1) #if not type(s_query) is numpy.ndarray else [splev(s_query[i], tck_y[i], der=1) for i in range(len(s_query))]

    x_ss = splev(s_query, tck_x, der=2) #if not type(s_query) is numpy.ndarray else [splev(s_query[i], tck_x[i], der=2) for i in range(len(s_query))]
    y_ss = splev(s_query, tck_y, der=2) #if not type(s_query) is numpy.ndarray else [splev(s_query[i], tck_y[i], der=2) for i in range(len(s_query))]

    theta = np.arctan2(y_s, x_s)
    kappa = (x_s * y_ss - y_s * x_ss) / (x_s**2 + y_s**2) ** 1.5
    return theta, kappa

def curvature_profile_spline(tck_x, tck_y, parametrized_path, step=0.001):
    s_values = np.linspace(parametrized_path[0][2], parametrized_path[-1][2], int((parametrized_path[-1][2] - parametrized_path[0][2])/step))
    theta_spline, kappa_spline = get_orientation_and_curvature(tck_x, tck_y, s_values)
    plt.figure(figsize=(10, 5))
    plt.plot(s_values, kappa_spline, label="Curvature")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Arc length s (m)")
    plt.ylabel("Curvature κ [1/m]")
    plt.title("Curvature Profile of Spline")
    plt.legend()
    plt.grid(True)


def get_clothoid_from_path(path,start_position, end_position):
    curve_list = CurveList()
    previous = []


def fit_clothoids(tck_x, tck_y, parametrized_path, step=0.01):
    s_values = np.linspace(parametrized_path[0][2], parametrized_path[-1][2], int((parametrized_path[-1][2] - parametrized_path[0][2])/step))
    x_spline = splev(s_values, tck_x)
    y_spline = splev(s_values, tck_y)
    theta_spline, kappa_spline = get_orientation_and_curvature(tck_x, tck_y, s_values)
    curve_list = CurveList()
    for i in range(1, len(s_values)):
        n_cl= ClothoidCurve.get_from_positions(Position(x_spline[i-1],y_spline[i-1],Angle.from_radians(theta_spline[i-1]), kappa_spline[i-1]), Position(x_spline[i],y_spline[i],Angle.from_radians(theta_spline[i]), kappa_spline[i]), 0.001)
        curve_list.add_curve_list(n_cl)
    return curve_list
def fit_clothoids_from_waypoints(anchors):
    """
    anchors: list of (x, y, theta, kappa) states
             - must include start and goal
             - may include intermediate waypoints (sparse)
    step: resolution for downstream sampling if needed
    """
    curve_list = CurveList()
    for i in range(1, len(anchors)):
        x0, y0, th0, k0 = anchors[i-1]
        x1, y1, th1, k1 = anchors[i]

        p0 = Position(x0, y0, Angle.from_radians(th0), k0)
        p1 = Position(x1, y1, Angle.from_radians(th1), k1)

        # Solve a G² clothoid interpolation between p0 and p1
        n_cl = ClothoidCurve.get_from_positions(p0, p1)
        curve_list.add_curve_list(n_cl)

    return curve_list
def curve_list_to_path(curve_list: CurveList, step=0.001):
    length = curve_list.getMaxValue()
    s_values = np.linspace(0, length, int(length/step))
    path = [curve_list.getPosition(s, step) for s in s_values]
    path = [(p.x, p.y) for p in path]
    return path



def simulate_gradient_descent_path(field_map, start_xy, step_size=0.05, max_steps=10000, stop_thresh=0.05):
    path = [start_xy]
    x, y = start_xy
    ny, nx = field_map.grid.shape
    res = field_map.resolution

    for _ in range(max_steps):
        i, j = int(y / res), int(x / res)

        # Stop if out of bounds or close to goal
        if i <= 1 or j <= 1 or i >= ny-2 or j >= nx-2:
            break
        if field_map.goal and np.linalg.norm(np.array((i, j)) - np.array(field_map.goal)) < 2:
            break

        # Compute gradient using central differences
        dU_dx = (field_map.grid[i, j+1] - field_map.grid[i, j-1]) / (2 * res)
        dU_dy = (field_map.grid[i+1, j] - field_map.grid[i-1, j]) / (2 * res)

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
def plot_path_on_field(field_map, path):
    X, Y = np.meshgrid(
        np.linspace(0, field_map.width, field_map.nx),
        np.linspace(0, field_map.height, field_map.ny)
    )

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, field_map.grid, levels=50, cmap='viridis')
    plt.colorbar(label="Potential")
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'r.-', label="Robot Path")
    plt.plot(path[0, 0], path[0, 1], 'bo', label="Start")
    gx, gy = field_map.goal[1] * field_map.resolution, field_map.goal[0] * field_map.resolution
    plt.plot(gx, gy, 'go', label="Goal")
    plt.legend()
    plt.title("Gradient Descent Path on Harmonic Field")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)

def plot_path(path):
    plt.figure(figsize=(10, 8))
    plt.plot(path[:, 0], path[:, 1], 'r.-', label="Robot Path")
    plt.plot(path[0, 0], path[0, 1], 'bo', label="Start")
    plt.title("Gradient Descent Path on Harmonic Field")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)


def plot_clothoid_path_on_field(field_map, clothoid_path, path):
    X, Y = np.meshgrid(
        np.linspace(0, field_map.width, field_map.nx),
        np.linspace(0, field_map.height, field_map.ny)
    )

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, 1-(1-field_map.grid)**0.1, levels=50, cmap='viridis')
    plt.colorbar(label="Potential")
    path = np.array(path)
    clothoid_path = np.array(clothoid_path)
    plt.plot(path[:, 0], path[:, 1], 'r.-', label="Robot Path")
    plt.plot(path[0, 0], path[0, 1], 'bo', label="Start")
    plt.plot(clothoid_path[:, 0], clothoid_path[:, 1], 'g.-', label="Robot Path with clothoid")
    gx, gy = field_map.goal[1] * field_map.resolution, field_map.goal[0] * field_map.resolution
    plt.plot(gx, gy, 'go', label="Goal")
    plt.legend()
    plt.title("Gradient Descent Path on Harmonic Field")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)

def plot_clothoid_path_on_distorted_field(field_map, clothoid_path):
    X, Y = np.meshgrid(
        np.linspace(0, field_map.width, field_map.nx),
        np.linspace(0, field_map.height, field_map.ny)
    )

    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, 1 - (1 - field_map.grid) ** 0.1, levels=50, cmap='viridis')
    plt.colorbar(label="Potential")
    clothoid_path = np.array(clothoid_path)
    plt.plot(clothoid_path[:, 0], clothoid_path[:, 1], 'g.-', label="Robot Path with clothoid")
    gx, gy = field_map.goal[1] * field_map.resolution, field_map.goal[0] * field_map.resolution
    plt.plot(gx, gy, 'go', label="Goal")
    plt.legend()
    plt.title("Gradient Descent Path on Harmonic Field")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True)

def plot_clothoid_curvature(curve_list: CurveList, step=0.001):
    length = curve_list.getMaxValue()
    s_values = np.linspace(0, length, int(length/step))
    print(len(s_values))
    path = [curve_list.getPosition(s, step) for s in s_values]
    plt.figure(figsize=(10, 5))
    plt.plot(s_values, [p.curvature for p in path], label="Curvature")
    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("Arc length s (m)")
    plt.ylabel("Curvature κ [1/m]")
    plt.title("Curvature Profile of Clothoid")
    plt.legend()
    plt.grid(True)
