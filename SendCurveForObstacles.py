from matplotlib import pyplot as plt

from src.utils.PotentialFieldMap import PotentialFieldMap
from src.utils.path_following import plot_path_on_field

field = PotentialFieldMap(width_m=3.0, height_m=2.0, resolution_m=0.05)
field.set_goal(2.5, 1.75)
field.add_obstacle(1.0, 1.0, 0.5)
field.solve()
field.visualize_powered()
plt.show()
curve_list, path = field.get_as_curve_list((0.25, 0.25))

print(curve_list.get_curve_count())
plot_path_on_field(field, path)
plt.show()