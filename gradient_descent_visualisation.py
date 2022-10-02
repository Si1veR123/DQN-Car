import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})


# Setting the axes properties
ax.set_xlim3d([-10.0, 10.0])
ax.set_xlabel('X')

ax.set_ylim3d([-10.0, 10.0])
ax.set_ylabel('Y')

ax.set_zlim3d([-100.0, 100.0])
ax.set_zlabel('Z')


def data_generator(x, y, derivative=False):
    return (y, x) if derivative else x*y


# Make data.
X = np.arange(-10, 10, 0.5)
Y = np.arange(-10, 10, 0.5)

Z = []
for x in X:
    row = []
    for y in Y:
        row.append(data_generator(x, y))
    Z.append(row)
Z = np.array(Z)

X, Y = np.meshgrid(X, Y)


# Plot the surface.
surf = ax.plot_wireframe(X, Y, Z, linewidth=0.05, antialiased=False)


m = (int(bool(max(np.random.random()-0.5, 0)))-0.5)*2

line_x = [m*(9 - np.random.random()*2-1)]
line_y = [m*(9 - np.random.random()*2-1)]
line_z = [data_generator(line_x[0], line_y[0])+0.1]
lr = 0.01
rand = 0.2
line, = ax.plot(line_x, line_y, line_z, "red", lw=3, zorder=5)
while True:
    if not (abs(line_x[-1]) > 10 or abs(line_y[-1]) > 10):
        dx, dy = data_generator(line_x[-1], line_y[-1], derivative=True)
        line_x.append(line_x[-1] - dx*lr + rand*(np.random.random()-0.5))
        line_y.append(line_y[-1] - dy*lr + rand*(np.random.random()-0.5))
        line_z.append(data_generator(line_x[-1], line_y[-1])+0.1)

    line.set_data(line_x, line_y)
    line.set_3d_properties(line_z)
    plt.draw()
    plt.pause(0.01)
