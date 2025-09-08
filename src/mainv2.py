
import matplotlib
from math import sqrt

from src.utils.SaveCurveListToFile import save_curve_list
from src.utils.gpt_path_improvement import harmonic_guided_clothoids, clothoid_with_obstacle_anchors

matplotlib.use('qt5agg')
from src.utils.path_following import *
from src.utils.MultiGridSolver import MultigridSolver
from utils.PotentialFieldMap import PotentialFieldMap
from utils.HarmonicSolver import HarmonicSolver
from scipy.interpolate import splrep, splev
from simplification.cutil import simplify_coords


# Create and solve a potential field
field = PotentialFieldMap(width_m=3.0*1000, height_m=2.0*1000, resolution_m=0.05*1000)
field.set_goal(2.7*1000, 1*1000)
field.add_obstacle(0.98*1000, 1.0*1000, 0.3*1000)
field.add_obstacle(0.98*1000, 2.0*1000, 0.5*1000)
field.add_obstacle(0.5*1000, 0.5*1000, 0.4*1000)
field.add_obstacle(1.88*1000, 1.5*1000, 0.5*1000)
#field.add_border()

solver = HarmonicSolver(field)
solver.solve_SOR(max_iter=500)
field.visualize_powered()
"""
field.visualize_powered()
field.visualize()
plt.show()
start_pos = (0.1*1000, 1*1000)
start_xy = start_pos
field_map = field
i, j = int(start_xy[1] / field_map.resolution), int(start_xy[0] / field_map.resolution)
print("Gradient at start:", (
    field_map.grid[i, j+1] - field_map.grid[i, j-1],
    field_map.grid[i+1, j] - field_map.grid[i-1, j]
))
"""


"""
path = simulate_gradient_descent_path(field, start_pos)
simplified_path = simplify_coords(path, epsilon=0.01)
plot_path_on_field(field, path)
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
"""
curve_list, path = field.get_as_curve_list(start_pos)
print(curve_list.get_curve_count())
plot_path_on_field(field, path)
plot_clothoid_curvature(curve_list, step=0.01)
plt.show()
save_curve_list("test.bin", curve_list)
"""

start_pos = (100,1000,0,0)
end_pos = (2700, 1000, 0,0)

data, path = harmonic_guided_clothoids(field, start_pos, end_pos)
plot_path_on_field(field, path)
plot_clothoid_curvature(data)
plt.show()

save_curve_list("curves.bin", data)

positions = [start_pos, (523,1310, 45, 0), (940, 1430, 0, 0), (1500,1010, -45, 0), (2032,695, 0, 0), end_pos]

positions = [Position(p[0], p[1], Angle.from_degrees(p[2]), p[3]) for p in positions]

curve_list = CurveList()
for i in range(len(positions) - 1):
    curve_list.add_curve(ClothoidCurve.get_from_positions(positions[i], positions[i+1]))

path = curve_list_to_path(curve_list)
plot_clothoid_path_on_distorted_field(field, path)
plt.show()
save_curve_list("hand_made.bin", curve_list)