import matplotlib.pyplot as plt
import numpy as np
import Plotter

def fit_curve_newton(points, initial_guesses, degree=1, tolerance=1e-7, max_iter=100):
    if len(initial_guesses) != degree+1:
        print("length of initial guess for C incorrect")
        return
    
    J = np.zeros((len(points),degree+1))
    
    for i in range(len(points)):
        for j in range (degree+1):
            J[i][j] = points[i][0]**j
            
    J_pseudo_inverse = np.linalg.inv(J.T @ J) @ J.T
    
    C = np.array(initial_guesses, dtype=float).reshape(-1, 1)
    
    history = [C]
    
    for i in range(max_iter):
        F = np.zeros((len(points),1))
        for m in range(len(points)):
            for n in range(degree+1):
                F[m][0] += C[n][0] * points[m][0]**n
            F[m][0] -= points[m][1]
            
        C_next = C - J_pseudo_inverse @ F
        history.append(C_next)
        
        if np.max(np.abs(C_next - C)) < tolerance:
            C = C_next
            break

        C = C_next
            
    return C, history





pts = [(0,0.5),(1, 1.5),(2,3.5),(3,7.5)]

init_guess1 = [1,1]
soln1, hist = fit_curve_newton(pts, init_guess1, degree=1)

print(soln1)

init_guess2 = [1,1,1]
soln2, hist = fit_curve_newton(pts, init_guess2, degree=2)

print(soln2)

def init_line(x):
    return (init_guess1[0] + init_guess1[1] * x)

def line(x):
    return (soln1[0][0] + soln1[1][0] * x)

def init_parabola(x):
    return (init_guess2[0] + init_guess2[1] * x + init_guess2[2] * x**2)

def parabola(x):
    return (soln2[0][0] + soln2[1][0] * x + soln2[2][0] * x**2)


x = np.linspace(-2, 10, 1000)
plotter = Plotter.Plotter()

plot_title = "Best fit Line and Parabola for Set of Points"
plotter.plot_points_and_two_functions(x, pts, line, parabola, title=plot_title, f1_name="y = 2.3x - 0.2", f2_name="y = 0.75x^2 + 0.05x + 0.55")

plot_title = "Newton-Raphson Multivariate Iterations for Line Fitting"
plotter.plot_points_and_two_functions(x, pts, init_line, line, title=plot_title, f1_name="y = x + 1", f2_name="y = 2.3x - 0.2")

plot_title = "Newton-Raphson Multivariate Iterations for Parabola Fitting"
plotter.plot_points_and_two_functions(x, pts, init_parabola, parabola, title=plot_title, f1_name="y = x^2 + x + 1", f2_name="y = 0.75x^2 + 0.05x + 0.55")

plt.show()
