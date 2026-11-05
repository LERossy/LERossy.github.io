import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from Plotter import Plotter
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap

STANDARD = 0
PC_ANALYSIS = 1
ISOMAP = 2

# ----------
# Parse Data
# ----------

data = []

with open(Path(__file__).parent / "iris.data") as d:
    for line in d:
        line = line.strip()
        if not line:
          continue
        entries = line.split(",")
        data.append([float(x) for x in entries[:4]] + [entries[4]])


# print(data)
features = np.array([row[:4] for row in data])   # shape (150, 4), dtype float64
labels   = np.array([row[4] for row in data])    # shape (150,), strings

plotter = Plotter()

# -------
# Part 2:
# -------

# 1D, 2D and 3D marginalization on normal axes
plotter.marginalize(features, 1, STANDARD)
plotter.marginalize(features, 2, STANDARD)
plotter.marginalize(features, 3, STANDARD)
    
# -------
# Part 3:
# -------

# Perform PCA for 1, 2 and 3 Principal Components
pca1 = PCA(n_components=1)
pca2 = PCA(n_components=2)
pca3 = PCA(n_components=3)

data_marginalized_PC1 = pca1.fit_transform(features)
data_marginalized_PC2 = pca2.fit_transform(features)
data_marginalized_PC3 = pca3.fit_transform(features)

# 1D, 2D and 3D marginalization on PC axes
plotter.marginalize(data_marginalized_PC1, 1, PC_ANALYSIS)
plotter.marginalize(data_marginalized_PC2, 2, PC_ANALYSIS)
plotter.marginalize(data_marginalized_PC3, 3, PC_ANALYSIS)

# Print PC vectors --> prints V^T
print(pca3.components_)

# -------
# Part 4:
# -------

# Print explained variance and variance ratio for each PC
print(pca3.explained_variance_)
print(pca3.explained_variance_ratio_)

# -------
# Part 5:
# -------

# Perform ISOMAP for 1, 2 and 3 Principal Components
isomap1 = Isomap(n_components=1, n_neighbors=3)
isomap2 = Isomap(n_components=2, n_neighbors=3)
isomap3 = Isomap(n_components=3, n_neighbors=3)

data_marginalized_isomap1 = isomap1.fit_transform(features)
data_marginalized_isomap2 = isomap2.fit_transform(features)
data_marginalized_isomap3 = isomap3.fit_transform(features)

# 1D, 2D and 3D marginalization on ISOMAP axes
plotter.marginalize(data_marginalized_isomap1, 1, ISOMAP)
plotter.marginalize(data_marginalized_isomap2, 2, ISOMAP)
plotter.marginalize(data_marginalized_isomap3, 3, ISOMAP)

# Print Reconstruction Errors
print(isomap1.reconstruction_error())
print(isomap2.reconstruction_error())
print(isomap3.reconstruction_error())


# Plot everything
plt.show()