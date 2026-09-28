import numpy as np


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
init_guess = [1,1,1]

soln, hist = fit_curve_newton(pts, init_guess, degree=2)

print(soln)
print(len(hist))
print(hist)