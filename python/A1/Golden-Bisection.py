import math
import matplotlib.pyplot as plt
import numpy as np

import Plotter

# ---------
# Algorithm
# ---------

def golden_section_search(x0, y0, f, a, b, tolerance=1e-7):
    history = []
    
    phi = (1 + math.sqrt(5)) / 2
    resphi = 2 - phi
    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)
    def dist_sq(x): return (x - x0)**2 + (f(x) - y0)**2
    f_x1 = dist_sq(x1)
    f_x2 = dist_sq(x2)
    history.append((a,b))
    # print(a, b, x1, x2)
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
        history.append((a,b))
        # print(a, b, x1, x2)
    best_x = (a + b) / 2
    return math.sqrt(dist_sq(best_x)), best_x, history


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
plotter = Plotter.Plotter()

# f(x)
for i in range(len(points)):
    d, solutions[i], hist = golden_section_search(points[i][0], points[i][1], f, -2, 2)
    print (points[i], [-2,2], float(solutions[i]), float(d), len(hist))
    
plot_title = "Shortest Line Segments Between Points and f(x) = x^2 + 5 Following Golden Section Search"
plotter.plot_shortest_distances_for_function(x, f, points, solutions, title=plot_title, function_name="f(x)")
    
print()

# g(x)
sections = [(4,5), (-0.5,0.5), (-0.5,0.5), (4,5), (4,5), (4,5), (-0.5,0.5), (4,5), (4,5), (-0.5,0.5)]
for i in range(len(points)):
    d, solutions[i], hist = golden_section_search(points[i][0], points[i][1], g, sections[i][0], sections[i][1])
    print (points[i], [sections[i][0], sections[i][1]], float(solutions[i]), float(d), len(hist))
    
plot_title = "Shortest Line Segments Between Points and g(x) = 3x^3 - 13x^2 + x - 6 Following Golden Section Search"
plotter.plot_shortest_distances_for_function(x, g, points, solutions, y_bounds=(-10, 10), title=plot_title, function_name="g(x)")

print()

# e(x)
for i in range(len(points)):
    d, solutions[i], hist = golden_section_search(points[i][0], points[i][1], e, -10, 1)
    print (points[i], [-10, 1], float(solutions[i]), float(d), len(hist))
    
plot_title = "Shortest Line Segments Between Points and e(x) = -1.5e^(2x) + 0.5 Following Golden Section Search"
plotter.plot_shortest_distances_for_function(x, e, points, solutions, y_bounds=(-5, 8), title=plot_title, function_name="e(x)")

print()

# c(x)
sections = [(-1,0), (-4.5,-3), (-8,-7), (2,3), (5,6.5), (1.5,2.5), (-6,-4), (0,1.5), (-1,0), (-3,-2)]
for i in range(len(points)):
    d, solutions[i], hist  = golden_section_search(points[i][0], points[i][1], c, sections[i][0], sections[i][1])
    
    if points[i] == (0, 5.4):
        # Close-up of Biection Search iterations
        plot_title = "Golden Section Iterations of Finding Shortest Distance Between (0, 5.4) and c(x) = 2cos(x - 1) + 1"
        plotter.plot_bisection_steps( x, c, points[i][0], points[i][1], hist, title=plot_title, function_name="c(x)")
        print()
        for k in range(len(hist)):
            print(hist[k])
        print()
        
    print (points[i], [sections[i][0], sections[i][1]], float(solutions[i]), float(d), len(hist))

plot_title = "Shortest Line Segments Between Points and c(x) = 2cos(x - 1) + 1 Following Golden Section Search"
plotter.plot_shortest_distances_for_function(x, c, points, solutions, y_bounds=(-6, 8), title=plot_title, function_name="c(x)")
    
print()

# l(x)
for i in range(len(points)):
    d, solutions[i], hist = golden_section_search(points[i][0], points[i][1], l, 0, 6.5)
    print (points[i], [0, 6.5], float(solutions[i]), float(d), len(hist))
    
plot_title = "Shortest Line Segments Between Points and l(x) = 0.5ln(x) Following Golden Section Search"
plotter.plot_shortest_distances_for_function(x, l, points, solutions, y_bounds=(-8, 8), title=plot_title, function_name="l(x)")
    
plt.show()