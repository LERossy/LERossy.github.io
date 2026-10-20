import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import Plotter

def generate_PHI(data, degree=1):
    if data.ndim == 1:                    # a single variable: treat it as one column
        data = data.reshape(-1, 1)
    indp_vars = data.shape[1]

    PHI = np.ones((degree * indp_vars + 1, len(data)))
    
    for n in range (len(data)):
        for k in range(0, degree):
            for l in range (indp_vars):
                PHI[k * indp_vars + l + 1][n] = data[n][l]**(k+1)
                
    return PHI
    

# Assuming data is formatted as each row corresponding to one datapoint of N entries, 
# where the first N-1 entries are the independent variables, and the Nth entry is the dependent variable
def polynomial_regression(data, degree=1):
    X_raw = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(np.float64)

    keep = X_raw.std(axis=0) > 0              # drop columns that never change
    X = X_raw[:, keep]
    # X = (X - X.mean(axis=0)) / X.std(axis=0)
    indp_vars = X.shape[1]

    PHI = generate_PHI(X, degree)
            
    PHI_pseudo_inv = np.linalg.inv(PHI @ PHI.T) @ PHI
    
    weights = PHI_pseudo_inv @ y.T

    # put the weights back in their original positions, leaving 0 for dropped columns
    total_vars = len(keep)
    kept_cols = np.flatnonzero(keep)
    full_weights = np.zeros(degree * total_vars + 1)
    full_weights[0] = weights[0]

    for k in range(0, degree):
        for l in range (indp_vars):
            full_weights[k * total_vars + kept_cols[l] + 1] = weights[k * indp_vars + l + 1]

    return full_weights, X_raw, y


# Perform polynomial regression on 1 dimension of training data only
def polynomial_regression_1d_vis(data, dimension, degree=1):
    if dimension > len(data[0])-1 or dimension <= 0:
        print("Invalid dimension selected")
        return
    
    X = data[:, dimension-1].astype(np.float64)
    y = data[:, -1].astype(np.float64)

    PHI = generate_PHI(X, degree)
            
    PHI_pseudo_inv = np.linalg.inv(PHI @ PHI.T) @ PHI
    
    weights = PHI_pseudo_inv @ y.T

    return weights, X, y

# Perform L2-regularized (Ridge) polynomial regression on 1 dimension of training data
def L2_1d_regularized_regression(data, y, degree=1, reg_constant=0):
    PHI = generate_PHI(data, degree)
            
    PHI_pseudo_inv_reg = np.linalg.inv(PHI @ PHI.T + reg_constant * np.eye(PHI.shape[0])) @ PHI
    
    weights = PHI_pseudo_inv_reg @ y.T

    return weights

# Perform cross-validation 
def cross_validate(data, y, degree=1, folds=10, reg_const=0):
    pts = data.shape[0]
    fold_size = int(pts / folds)
    avg_rmse = 0.0
    
    for i in range(folds):
        test = data[i * fold_size : (i+1) * fold_size]
        test_y = y[i * fold_size : (i+1) * fold_size]
        training = np.concatenate((data[:i * fold_size], data[(i+1) * fold_size:]))
        training_y = np.concatenate((y[:i * fold_size], y[(i+1) * fold_size:]))
        
        w = L2_1d_regularized_regression(training, training_y, degree, reg_const)
        
        phi = generate_PHI(test, degree)
        y_hat = phi.T @ w
        squared_errors = (y_hat - test_y)**2
        rmse = np.sqrt(np.mean(squared_errors))
        avg_rmse += rmse
        
    avg_rmse /= folds
    return avg_rmse
        

    


TRAINING_DATA_SIZE = 100
MAX_DEGREE = 7

training_data = []
test_data = []
i = 0

with open(Path(__file__).parent / "assn2 material/assn2 material/housing.data") as d:
  for line in d:
    if i < TRAINING_DATA_SIZE:
        training_data.append(line.split())
    else:
        test_data.append(line.split())
    i += 1
    
training_data = np.array(training_data, dtype=np.float32)
test_data = np.array(test_data, dtype=np.float32)

test_data_X = test_data[:, :-1].astype(np.float64)
test_data_y = test_data[:, -1].astype(np.float64)

test_RMSEs = []
training_RMSEs = []

# Part a
for i in range(1, MAX_DEGREE+1):
    weights, training_data_X, training_data_y = polynomial_regression(training_data, i)
    # print(weights)
    
    # Part b
    if i == 1:
        print(weights)

    phi = generate_PHI(training_data_X, i)
    y_hat = phi.T @ weights
    squared_errors = (y_hat - training_data_y)**2
    rmse = np.sqrt(np.mean(squared_errors))
    training_RMSEs.append(rmse)
    
    phi = generate_PHI(test_data_X, i)
    y_hat = phi.T @ weights
    squared_errors = (y_hat - test_data_y)**2
    rmse = np.sqrt(np.mean(squared_errors))
    test_RMSEs.append(rmse)
    
    print(f"{training_RMSEs[i-1]:.4g} {test_RMSEs[i-1]:.4g}")
    
    
plotter = Plotter.Plotter()

# Part a
plotter.plot_RMSEs(training_RMSEs, degree=MAX_DEGREE, data_type="training")
plotter.plot_RMSEs(test_RMSEs, degree=MAX_DEGREE, data_type="test")

print()

# Part c
FEATURE = 2

for i in range(1, MAX_DEGREE+1):
    weights, training_data_X, training_data_y = polynomial_regression_1d_vis(training_data, FEATURE, i)
    print(weights)
    
    plotter.plot_1d_training_model_test(training_data_X, training_data_y, weights, test_data_X[:, FEATURE-1], test_data_y, i, dimension="proportion of residential land zoned for lots over 25,000 sq.ft.")

print()
    
# Part d
FEATURE = 2
POLYNOMIAL = 8
FOLDS = 10
lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
avg_validation_set_errors = []

X = training_data[:, FEATURE-1].astype(np.float64)
y = training_data[:, -1].astype(np.float64)

for reg_const in lambdas:
    avg_error = cross_validate(X, y, POLYNOMIAL, FOLDS, reg_const)
    avg_validation_set_errors.append(avg_error)
    print(avg_error)

plotter.plot_validation_errors(lambdas, avg_validation_set_errors)

plt.show()
