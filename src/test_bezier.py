from itertools import accumulate

from curve_library import Position, HermiteSplineCurve, Angle, ClothoidCurve
import numpy as np
from matplotlib import pyplot as plt


start = Position(0,0,Angle.from_degrees(0), 0)
end = Position(500,500,Angle.from_degrees(180), 0)
def handle_spline_data():
    spline = HermiteSplineCurve.build(start, end, 1000)
    t = np.linspace(0,1,200)
    spline_positions = [spline.eval(index) for index in t]
    positions = np.array([[p[0].x, p[0].y] for p in spline_positions])
    first_derivatives = np.array([[p[1].x, p[1].y] for p in spline_positions])
    second_derivatives = np.array([[p[2].x, p[2].y] for p in spline_positions])

    kappa = (first_derivatives[:, 0] * second_derivatives[:, 1] - first_derivatives[:, 1] * second_derivatives[:, 0])/(first_derivatives[:, 1]**2 + first_derivatives[:, 0]**2)**1.5
    length = np.array(list(accumulate(np.sqrt(first_derivatives[:, 1]**2 + first_derivatives[:, 0]**2) * (1-0)/200)))
    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 1])
    plt.title("Position found by the spline")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)

    plt.figure(figsize=(10, 8))
    plt.plot(first_derivatives[:, 0], first_derivatives[:, 1], )
    plt.title("Derivatives found by the spline")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)

    plt.figure(figsize=(10, 8))
    plt.plot(t, length)
    plt.legend()
    plt.title("Spline's Length")
    plt.xlabel("Parameter value")
    plt.ylabel("Length")
    plt.grid(True)
    plt.figure(figsize=(10, 8))
    plt.plot(length, kappa)
    plt.title("Spline's Curvature")
    plt.xlabel("Length (mm)")
    plt.ylabel("Curvature (1/mm)")
    plt.grid(True)


    print(spline.getLength()," & ", length[-1])

def handle_clothoid_data():
    clothoid = ClothoidCurve.get_from_positions(start, end)
    t = np.linspace(0, clothoid.getMaxValue(), 200)
    clothoid_pos = [clothoid.getPosition(i, 0.01) for i in t]
    clothoid_derivative = [clothoid.get_derivative(i) for i in t]
    positions = np.array([[p.x, p.y] for p in clothoid_pos])
    first_derivatives = np.array([[p.x, p.y] for p in clothoid_derivative])
    kappa = [p.curvature for p in clothoid_pos]

    plt.figure(figsize=(10, 8))
    plt.plot(positions[:, 0], positions[:, 1])
    plt.title("Position found by the clothoid")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)

    plt.figure(figsize=(10, 8))
    plt.plot(first_derivatives[:, 0], first_derivatives[:, 1], )
    plt.title("Derivatives found by the clothoid")
    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.axis("equal")
    plt.grid(True)

    plt.figure(figsize=(10, 8))
    plt.plot(t, kappa)
    plt.title("clothoid's Curvature")
    plt.xlabel("Length (mm)")
    plt.ylabel("Curvature (1/mm)")
    plt.grid(True)


handle_spline_data()
handle_clothoid_data()


plt.show()

