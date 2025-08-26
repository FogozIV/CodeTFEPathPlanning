import numpy as np
import struct
from matplotlib import pyplot as plt
from packet_handler import *
from curve_library import *

from src.utils.SaveCurveListToFile import save_curve_list

spi = SocketPacketInterface("mainrobotteensy.local")

positions = [(0,0, 0), (1000,-200,45), (1200,400,180), (800,400, 225), (400,0, 180), (0,0,180)]
positions = [Position(p[0], p[1], Angle.from_degrees(p[2])) for p in positions]

curve_list = CurveList()
for i in range(len(positions) -1 ):
    curve_list.add_curve(QuinticHermiteSpline.build(positions[i], positions[i+1], (positions[i+1] - positions[i]).get_distance() * 1.1))

min_value = curve_list.getMinValue()
max_value = curve_list.getMaxValue()
t = np.linspace(min_value, max_value, 1000)

plot_pos = [curve_list.getPosition(i, 0.01) for i in t]

positions = np.array([[p.x, p.y] for p in plot_pos])

plt.figure(figsize=(10, 8))
plt.plot(positions[:, 0], positions[:, 1])
plt.title("Position found by the spline")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.axis("equal")
plt.grid(True)
plt.show()

a = input("Do you want to send the curve? (y/n)")
if a.lower() == "y":
    spi.dispatcher.register_PingPacket_callback(lambda x: print(x))
    spi.connect()
    spi.start()
    print("Sent the packet")
    print(curve_list.getFullCurve())
    spi.send_packet(SendTrajectoryPacket(curve_list.getFullCurve(), 100,200,0,0,0))
    spi.stop()

else:
    a = input("Do you want to save the curve? (y/n)")
    if a.lower() == "y":
        save_curve_list("curve_list.bin", curve_list)
print("Bye bye")
