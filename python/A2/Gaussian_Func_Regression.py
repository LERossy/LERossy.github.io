import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import Plotter


TRAINING_DATA_SIZE = 100
OVERLAP_FACTOR = 0.75


# For all sets of K
# shuffle data and pick K cKs
# For each dimension
#   Calculate avg delta cs
#   sigma = avg * OVERLAP_FACTOR; a = -1/2sigma^2
# Generate Phi matrix
# Business as usual

def generate_PHI(X, ck, a, K):
    pts = X.shape[0]

    PHI = np.ones((K, pts))

    for n in range(pts):
        for k in range(K-1):
            PHI[k, n] = np.exp( np.sum(a * (X[n] - ck[k])**2) )

    return PHI

def gaussian_regression(data, K):
    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(np.float64)

    indp_vars = X.shape[1]
    
    rng = np.random.default_rng()       # the 0 makes the shuffle the same every run
    ck = rng.permutation(X)[:K-1]
    a_values = []
    
    for l in range(indp_vars):
        ckl = np.sort(ck[:, l])
        avg_delta_c = np.mean(np.diff(ckl))
        if avg_delta_c == 0:
            a_values.append(0.0)
        else:
            sigma = avg_delta_c * OVERLAP_FACTOR
            a = -1 / (2 * sigma**2)
            a_values.append(a)
        
    PHI = generate_PHI(X, ck, a_values, K)
    
    PHI_pseudo_inv = np.linalg.inv(PHI @ PHI.T) @ PHI
        
    weights = PHI_pseudo_inv @ y.T

    return weights, PHI, y, ck, a_values


def L2_regularized_gaussian_regression(data, K, reg_const):
    X = data[:, :-1].astype(np.float64)
    y = data[:, -1].astype(np.float64)

    indp_vars = X.shape[1]
    
    rng = np.random.default_rng()       # the 0 makes the shuffle the same every run
    ck = rng.permutation(X)[:K-1]
    a_values = []
    
    for l in range(indp_vars):
        ckl = np.sort(ck[:, l])
        avg_delta_c = np.mean(np.diff(ckl))
        if avg_delta_c == 0:
            a_values.append(0.0)
        else:
            sigma = avg_delta_c * OVERLAP_FACTOR
            a = -1 / (2 * sigma**2)
            a_values.append(a)
        
    PHI = generate_PHI(X, ck, a_values, K)
    
    PHI_pseudo_inv_reg = np.linalg.inv(PHI @ PHI.T + reg_const * np.eye(PHI.shape[0])) @ PHI
        
    weights = PHI_pseudo_inv_reg @ y.T

    return weights, PHI, y, ck, a_values

# Perform cross-validation 
def cross_validate(data, K, folds=10, reg_const=0):
    pts = data.shape[0]
    fold_size = int(pts / folds)
    avg_rmse = 0.0
    
    for i in range(folds):
        test = data[i * fold_size : (i+1) * fold_size, : -1].astype(np.float64)
        test_y = data[i * fold_size : (i+1) * fold_size, -1].astype(np.float64)
        training = np.concatenate((data[:i * fold_size], data[(i+1) * fold_size:]))
        
        w, _, _, ck, a = L2_regularized_gaussian_regression(training, K, reg_const)
        
        phi = generate_PHI(test, ck, a, K)
        y_hat = phi.T @ w
        squared_errors = (y_hat - test_y)**2
        rmse = np.sqrt(np.mean(squared_errors))
        avg_rmse += rmse
        
    avg_rmse /= folds
    return avg_rmse 
    

Ks = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
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

for i in range(len(Ks)):
    weights, training_phi, training_data_y, ck, a = gaussian_regression(training_data, Ks[i])
    # print(weights)
    # print()
    
    y_hat = training_phi.T @ weights
    squared_errors = (y_hat - training_data_y)**2
    rmse = np.sqrt(np.mean(squared_errors))
    training_RMSEs.append(rmse)
    
    phi = generate_PHI(test_data_X, ck, a, Ks[i])
    y_hat = phi.T @ weights
    squared_errors = (y_hat - test_data_y)**2
    rmse = np.sqrt(np.mean(squared_errors))
    test_RMSEs.append(rmse)
    
    print(f"{training_RMSEs[i]:.4g} {test_RMSEs[i]:.4g}")
    
    
plotter = Plotter.Plotter()

# Part e
plotter.plot_gaussian_RMSEs(training_RMSEs, Ks, data_type="training")
plotter.plot_gaussian_RMSEs(test_RMSEs, Ks, data_type="test")

# Part f
K = 90
FOLDS = 10
lambdas = [0, 0.01, 0.1, 1, 10, 100, 1000]
avg_validation_set_errors = []

for reg_const in lambdas:
    avg_error = cross_validate(training_data, K, FOLDS, reg_const)
    avg_validation_set_errors.append(avg_error)
    print(avg_error)

plotter.plot_validation_errors(lambdas, avg_validation_set_errors)

plt.show()