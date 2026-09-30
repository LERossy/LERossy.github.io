import matplotlib.pyplot as plt
import numpy as np

import Plotter

# ---------
# Algorithm
# ---------

def find_distance_newton(x0, y0, f, df, ddf, initial_guess=0.0, tolerance=1e-7, max_iter=100, domain=(-np.inf, np.inf)):
    x = initial_guess
    history = []
    
    # Make sure current x is in the domain
    if not (domain[0] < x < domain[1]):
        return None, None, history
    
    for i in range(max_iter):
        history.append(x)
            
        D_prime = 2 * (x - x0) + 2 * (f(x) - y0) * df(x)
        D_double_prime = 2 + 2 * (df(x)**2) + 2 * (f(x) - y0) * ddf(x)
        
        next_x = x - D_prime / D_double_prime
        
        # print(x, D_prime, D_double_prime, D_prime/D_double_prime, next_x)
        
        # Stop if next guess is outside domain
        if not (domain[0] < next_x < domain[1]):
            print("outside domain")
            break
        
        if abs(next_x - x) < tolerance:
            x = next_x
            break
        
        x = next_x
        
    shortest_distance = ((x - x0)**2 + (f(x) - y0)**2)**0.5
    
    history.append(x)
    
    return shortest_distance, x, history

# ----------------------------
# Example Points and Functions
# ----------------------------

points = [(0,0), (-4,0), (-8,0), (2,0), (6,0), (2,2), (-2.5,6), (0,5.4), (0,-1.3), (-5,-4)]

# y = x^2 + 5
def f(x):
    return (x**2 + 5)
def df(x):
    return 2 * x
def ddf(x):
    return 2

# y = 3x^3 - 13x^2 + x - 6
def g(x):
    return (3 * x**3 - 13 * x**2 + x - 6)
def dg(x):
    return(9 * x**2 - 26 * x + 1)
def ddg(x):
    return(18 * x - 26)

# y = -1.5e^(2x) + 0.5
def e(x):
    return (-1.5 * np.exp(2*x) + 0.5)
def de(x):
    return (-3 * np.exp(2*x))
def dde(x):
    return (-6 * np.exp(2*x))

# y = 2cos(x - 1) + 1
def c(x):
    return (2 * np.cos(x-1) + 1)
def dc(x):
    return (-2 * np.sin(x-1))
def ddc(x):
    return (-2 * np.cos(x-1))

# y = 0.5ln(x)
def l(x):
    return (0.5 * np.log(x))
def dl(x):
    return (0.5/x)
def ddl(x):
    return (-0.5/x**2)

# -----------------
# Results and Plots
# -----------------

x = np.linspace(-10, 10, 5000)
solutions = [0,0,0,0,0,0,0,0,0,0]
plotter = Plotter.Plotter()

# f(x)
for i in range(len(points)):
    d, solutions[i], hist = find_distance_newton(points[i][0], points[i][1], f, df, ddf)
    print (points[i], float(solutions[i]), float(d), len(hist))
    
plot_title = "Shortest Line Segments Between Points and f(x) = x^2 + 5"
    
plotter.plot_shortest_distances_for_function(x, f, points, solutions, title=plot_title, function_name="f(x)")
    
print()

# g(x)
init_guesses = [5, 0, 0, 5, 5, 5, 0, 5, 4, 0]
for i in range(len(points)):
    d, solutions[i], hist = find_distance_newton(points[i][0], points[i][1], g, dg, ddg, initial_guess=init_guesses[i])
    print (points[i], float(solutions[i]), float(d), len(hist))
    
plot_title = "Shortest Line Segments Between Points and g(x) = 3x^3 - 13x^2 + x - 6"
    
plotter.plot_shortest_distances_for_function(x, g, points, solutions, y_bounds=(-10, 10), title=plot_title, function_name="g(x)")
    
print()

# e(x)
for i in range(len(points)):
    d, solutions[i], hist = find_distance_newton(points[i][0], points[i][1], e, de, dde, initial_guess=points[i][0])
    print (points[i], float(solutions[i]), float(d), len(hist))
    
plot_title = "Shortest Line Segments Between Points and e(x) = -1.5e^(2x) + 0.5"

plotter.plot_shortest_distances_for_function(x, e, points, solutions, y_bounds=(-5, 8), title=plot_title, function_name="e(x)")
    
print()

# c(x)
init_guesses = [0, -4, -7.5, 2, 6, 2, -5, 0, 0, -2.5]
for i in range(len(points)):
    hist = []
    
    # Special case: point (0, 5.4)
    if points[i] == (0, 5.4):
        d, solutions[i], hist = find_distance_newton(points[i][0], points[i][1], c, dc, ddc, initial_guess=init_guesses[i])

        # Close-up of Newton's iterations
        plot_title = "Iterations of Finding Shortest Distance Between (0, 5.4) and c(x) = 2cos(x - 1) + 1"
        plotter.plot_newton_steps( x, c, points[i][0], points[i][1], hist, title=plot_title, function_name="c(x)")
    else:
        d, solutions[i], hist = find_distance_newton(points[i][0],points[i][1],c, dc, ddc, initial_guess=init_guesses[i])
    print (points[i], float(solutions[i]), float(d), len(hist))

plot_title = "Shortest Line Segments Between Points and c(x) = 2cos(x - 1) + 1"

plotter.plot_shortest_distances_for_function(x, c, points, solutions, y_bounds=(-6, 8), title=plot_title, function_name="c(x)")
    
print()

# l(x)
init_guesses = [1, 0.1, 0.1, 1, 1, 1, 1, 1, 0.1, 0.00001]
for i in range(len(points)):
    d, solutions[i], hist = find_distance_newton(points[i][0], points[i][1], l, dl, ddl, initial_guess=init_guesses[i], domain=(0, np.inf))
    print (points[i], float(solutions[i]), float(d), len(hist))
    
plot_title = "Shortest Line Segments Between Points and l(x) = 0.5ln(x)"

plotter.plot_shortest_distances_for_function(x, l, points, solutions, y_bounds=(-8, 8), title=plot_title, function_name="l(x)")
    
print()

plt.show()