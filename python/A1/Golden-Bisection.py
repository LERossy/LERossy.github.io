import math
import matplotlib.pyplot as plt
import numpy as np

import Plotter

# ---------
# Algorithm
# ---------

def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    def dist_sq(x): return (x - x0)**2 + (f(x) - y0)**2
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    while abs(b - a) > tolerance:
        if f_x1 < f_x2:
            b = x2
            x2 = x1
            f_x2 = f_x1
            x1 = a + resphi * (b - a)
            f_x1 = dist_sq(x1)
        else:
            a = x1
            x1 = x2
            f_x1 = f_x2
            x2 = b - resphi * (b - a)
            f_x2 = dist_sq(x2)
    best_x = (a + b) / 2
    return math.sqrt(dist_sq(best_x)), best_x


# ----------------------------
# Example Points and Functions
# ----------------------------

points = [(0,0), (-4,0), (-8,0), (2,0), (6,0), (2,2), (-2.5,6), (0,5.4), (0,-1.3), (-5,-4)]

def f(x):
    return (x**2 + 5)

def g(x):
    return (3 * x**3 - 13 * x**2 + x - 6)

def e(x):
    return (-1.5 * np.exp(2*x) + 0.5)

def c(x):
    return (2 * np.cos(x-1) + 1)

def l(x):
    return (0.5 * np.log(x))

# -----------------
# Results and Plots
# -----------------

x = np.linspace(-10, 10, 5000)
solutions = [0,0,0,0,0,0,0,0,0,0]
plotter = Plotter()

# f(x)
for i in range(len(points)):
    d, solutions[i] = golden_section_search(points[i][0], points[i][1], f, -2, 2)
    print ((float(d), float(solutions[i])))
    
plotter.plot_shortest_distances_for_function(x, f, points, solutions)
    
print()

# g(x)
sections = [(4,5), (-0.5,0.5), (-0.5,0.5), (4,5), (4,5), (4,5), (-0.5,0.5), (4,5), (4,5), (-0.5,0.5)]
for i in range(len(points)):
    d, solutions[i] = golden_section_search(points[i][0], points[i][1], g, sections[i][0], sections[i][1])
    print ((float(d), float(solutions[i])))
    
plotter.plot_shortest_distances_for_function(x, g, points, solutions, y_bounds=(-10, 10))

print()

# e(x)
for i in range(len(points)):
    d, solutions[i] = golden_section_search(points[i][0], points[i][1], e, -10, 1)
    print ((float(d), float(solutions[i])))
    
plotter.plot_shortest_distances_for_function(x, e, points, solutions, y_bounds=(-5, 8))

print()

# c(x)
sections = [(-1,0), (-4.5,-3), (-8,-7), (2,3), (5,6.5), (1.5,2.5), (-6,-4), (0,1.5), (-1,0), (-3,-2)]
for i in range(len(points)):
    d, solutions[i] = golden_section_search(points[i][0], points[i][1], c, sections[i][0], sections[i][1])
    print ((float(d), float(solutions[i])))

plotter.plot_shortest_distances_for_function(x, c, points, solutions, y_bounds=(-6, 8))
    
print()

# l(x)
for i in range(len(points)):
    d, solutions[i] = golden_section_search(points[i][0], points[i][1], l, 0, 6.5)
    print ((float(d), float(solutions[i])))
    
plotter.plot_shortest_distances_for_function(x, l, points, solutions, y_bounds=(-8, 8))
    
plt.show()