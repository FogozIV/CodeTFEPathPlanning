
import matplotlib
from math import sqrt

from src.utils.SaveCurveListToFile import save_curve_list

matplotlib.use('qt5agg')
from src.utils.path_following import *
from src.utils.MultiGridSolver import MultigridSolver
from utils.PotentialFieldMap import PotentialFieldMap
from utils.HarmonicSolver import HarmonicSolver
from scipy.interpolate import splrep, splev
from simplification.cutil import simplify_coords


# Create and solve a potential field
field = PotentialFieldMap(width_m=3.0, height_m=2.0, resolution_m=0.05)
field.set_goal(2.7, 1)
field.add_obstacle(1.0, 1.0, 0.2)
field.add_obstacle(2.0, 0.5, 0.15)
#field.add_border()

solver = HarmonicSolver(field)
solver.solve_SOR(max_iter=500)
field.visualize_powered()
field.visualize()
start_pos = (0.1, 1)
start_xy = start_pos
field_map = field
i, j = int(start_xy[1] / field_map.resolution), int(start_xy[0] / field_map.resolution)
print("Gradient at start:", (
    field_map.grid[i, j+1] - field_map.grid[i, j-1],
    field_map.grid[i+1, j] - field_map.grid[i-1, j]
))
"""
path = simulate_gradient_descent_path(field, start_pos)
plot_path_on_field(field, path)
simplified_path = simplify_coords(path, epsilon=0.01)
#print(len(simplified_path))
#plot_path_on_field(field, simplified_path)
param_path = get_parametrized_path(path)
x_k,y_k = get_spline_for_parametrized_path(param_path, s=0.001)
print(x_k, y_k)
show_spline_path(x_k, y_k, param_path)
curvature_profile_spline(x_k, y_k, param_path, step=0.00001)
get_clothoid_from_path(simplified_path[:-1], start_pos,[2.5,1.5])
curve_list= fit_clothoids(x_k, y_k, param_path, 0.1)
clothoid_path = curve_list_to_path(curve_list)
print(curve_list.get_curve_count())
plot_clothoid_path_on_field(field, clothoid_path, path)
plot_clothoid_curvature(curve_list, step=0.00001)
plt.show()
"""

curve_list, path = field.get_as_curve_list(start_pos)
print(curve_list.get_curve_count())
plot_path_on_field(field, path)
plt.show()

save_curve_list("test.bin", curve_list)