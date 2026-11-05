from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np

STANDARD = 0
PC_ANALYSIS = 1
ISOMAP = 2

class Plotter:
    
    def marginalize(self, data, nr_features=1, axes=STANDARD):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if nr_features == 3 else None)
        
        # Split data up by class
        iris_setosa = data[:50]
        iris_versicolor = data[50:100]
        iris_virginica = data[100:]
        
        # Plot 1D
        if nr_features == 1:
            ax.scatter([pt[0] for pt in iris_setosa], [-0.1 for pt in iris_setosa], label="Iris-setosa", s=20, color='red')
            ax.scatter([pt[0] for pt in iris_versicolor], [0 for pt in iris_versicolor], label="Iris-versicolor", s=20, color='blue')
            ax.scatter([pt[0] for pt in iris_virginica], [0.1 for pt in iris_virginica], label="Iris-virginica", s=20, color='green')
            
            ax.set_ybound(-5, 5)
            
            if axes == ISOMAP:
                ax.set_xlabel("Component 1")
            elif axes == PC_ANALYSIS:
                ax.set_xlabel("Principal Component 1")
            else:
                ax.set_xlabel("sepal length in cm")
        # Plot 2D
        elif nr_features == 2:
            ax.scatter([pt[0] for pt in iris_setosa], [pt[1] for pt in iris_setosa], label="Iris-setosa", s=25, color='red')
            ax.scatter([pt[0] for pt in iris_versicolor], [pt[1] for pt in iris_versicolor], label="Iris-versicolor", s=25, color='blue')
            ax.scatter([pt[0] for pt in iris_virginica], [pt[1] for pt in iris_virginica], label="Iris-virginica", s=25, color='green')
            if axes == ISOMAP:
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
            elif axes == PC_ANALYSIS:
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
            else:
                ax.set_xlabel("sepal length in cm")
                ax.set_ylabel("sepal width in cm")
        # Plot 3D 
        elif nr_features == 3:
            ax.scatter3D([pt[0] for pt in iris_setosa], [pt[1] for pt in iris_setosa], [pt[2] for pt in iris_setosa], label="Iris-setosa", s=20, color='red')
            ax.scatter3D([pt[0] for pt in iris_versicolor], [pt[1] for pt in iris_versicolor], [pt[2] for pt in iris_versicolor], label="Iris-versicolor", s=20, color='blue')
            ax.scatter3D([pt[0] for pt in iris_virginica], [pt[1] for pt in iris_virginica], [pt[2] for pt in iris_virginica], label="Iris-virginica", s=20, color='green')
            if axes == ISOMAP:
                ax.set_xlabel("Component 1")
                ax.set_ylabel("Component 2")
                ax.set_zlabel("Component 3")
            elif axes == PC_ANALYSIS:
                ax.set_xlabel("Principal Component 1")
                ax.set_ylabel("Principal Component 2")
                ax.set_zlabel("Principal Component 3")
            else:
                ax.set_xlabel("sepal length in cm")
                ax.set_ylabel("sepal width in cm")
                ax.set_zlabel("petal length in cm")

        # Remaining Plot setup
        if axes == ISOMAP:
            ax.set_title("Iris data marginalization onto " + str(nr_features) + " Components using ISOMAP data compression")
        elif axes == PC_ANALYSIS:
            ax.set_title("Iris data marginalization onto " + str(nr_features) + " Principal Components")
        else:
            ax.set_title("Iris data marginalization onto standard Cartesian Axes")

        ax.grid()
        ax.legend()