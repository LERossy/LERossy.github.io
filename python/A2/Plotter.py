import matplotlib.pyplot as plt
import numpy as np

# ------------------
# Plotting Functions
# ------------------

class Plotter:
    
    def plot_RMSEs(self, RMSEs, degree=1, data_type="training"):
        fig, ax = plt.subplots()
        polynomial_degrees = np.linspace(1, degree, degree)
        
        # Function
        ax.plot(polynomial_degrees, RMSEs, label="RMSE vs Polynomial Degree")
        
        # Draw x- and y-axes
        
        ax.set_xlim(0, degree+1)
        # ax.axhline(0, linewidth=1, color='black')
        # ax.axvline(0, linewidth=1, color='black')
        ax.set_xlabel("Polynomial Degree")
        ax.set_ylabel("RMSE")
        
        ax.set_title(f"RMSE vs Polynomial Degree for {data_type} data")

        ax.grid()
        ax.legend()
        
    def plot_RMSEs(self, RMSEs, K=[5], data_type="training"):
            fig, ax = plt.subplots()
            
            # Function
            ax.plot(K, RMSEs, label="RMSE vs K")
            
            # Draw x- and y-axes
            
            ax.set_xlim(0, K[-1] + 1)
            # ax.axhline(0, linewidth=1, color='black')
            # ax.axvline(0, linewidth=1, color='black')
            ax.set_xlabel("K")
            ax.set_ylabel("RMSE")
            
            ax.set_title(f"RMSE vs Number of Gaussian Basis Functions K for {data_type} data")
    
            ax.grid()
            ax.legend()
        
    def plot_1d_training_model_test(self, training_data_x, training_data_y, model_weights, test_data_x, test_data_y, degree, dimension="x"):
        fig, ax = plt.subplots()
        x = np.linspace(0, 100, 2000)
        
        def model(x):
            y = 0
            for i in range(degree+1):
                y += model_weights[i] * x**i
            return y
        
        ax.scatter(training_data_x, training_data_y, label="Training Data", color='green')
        ax.plot(x, model(x), label="Learned Polynomial")
        ax.scatter(test_data_x, test_data_y, label="Test Data", color='orange')
        
        ax.set_xlabel(dimension)
        ax.set_ylabel("y")

        ax.grid()
        ax.legend()

    def plot_validation_errors(self, lambdas, errors):
        fig, ax = plt.subplots()

        # 0 can't be shown on a log scale, so lambda = 0 becomes a horizontal baseline instead
        lambdas = np.array(lambdas, dtype=float)
        errors = np.array(errors, dtype=float)
        nonzero = lambdas > 0

        ax.semilogx(lambdas[nonzero], errors[nonzero], marker='o', label="Average Validation Error")

        if np.any(~nonzero):
            baseline = errors[~nonzero][0]
            ax.axhline(baseline, linestyle='--', color='red', label=f"No Regularization (λ = 0): {baseline:.4g}")

        ax.set_xlabel("Regularization Constant λ")
        ax.set_ylabel("Average Validation Error")

        ax.set_title("Average Validation Error vs Regularization Constant")

        ax.grid(which='both')
        ax.legend()