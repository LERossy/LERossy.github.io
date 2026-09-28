import matplotlib.pyplot as plt
import numpy as np

# ------------------
# Plotting Functions
# ------------------

class Plotter:
    
    def plot_shortest_distances_for_function(self, x_series, f, pts, solns, x_bounds=(-10,10), y_bounds=(-6,10), title=None, function_name="f(x)"):
        fig, ax = plt.subplots()
        ax.plot(x_series, f(x_series), label=function_name)
        
        # Draw x- and y-axes
        ax.axhline(0, linewidth=1, color='black')
        ax.axvline(0, linewidth=1, color='black')

        # Set viewing window
        ax.set_xlim(x_bounds[0], x_bounds[1])
        ax.set_ylim(y_bounds[0], y_bounds[1])

        # Make x and y units have the same physical scale
        ax.set_aspect('equal', adjustable='box')
        
        for i in range(len(pts)):
            p0 = (pts[i][0], pts[i][1])
            p1 = (solns[i], f(solns[i]))
            # Only label the first of each so the legend has one entry per kind
            first = (i == 0)
            ax.scatter(p0[0], p0[1], color='green', zorder=5,
                       label="Points" if first else None)
            ax.scatter(p1[0], p1[1], color='blue', zorder=5,
                       label="Closest points on f(x)" if first else None)
            ax.plot([p0[0],p1[0]], [p0[1],p1[1]], color='red',
                    label="Shortest distance" if first else None)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)

        ax.grid()
        ax.legend()

    def plot_newton_steps(self, x_series, f, x0, y0, x_history, title=None, function_name="f(x)"):
        fig, ax = plt.subplots()

        # Function
        ax.plot(x_series, f(x_series), label=function_name)

        # Original point
        ax.scatter(x0, y0, color='green', s=60, zorder=5, label="Starting point")

        # Newton points on the curve
        y_history = [f(x) for x in x_history]

        ax.scatter(x_history, y_history, color='orange', s=40, zorder=5, label="Newton iterates")

        # Axes
        ax.axhline(0, linewidth=1, color='black')
        ax.axvline(0, linewidth=1, color='black')

        # Automatically make the close-up around the Newton iterations
        x_min = min(x_history + [x0])
        x_max = max(x_history + [x0])

        padding = 0.5 * (x_max - x_min)

        if padding == 0:
            padding = 1

        ax.set_xlim(x_min - padding, x_max + padding)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        if title:
            ax.set_title(title)

        ax.grid()
        # ax.set_aspect('equal', adjustable='box')
        ax.legend()

        # Label the starting point and each Newton point with its coordinates
        ax.annotate(f"({x0:.4g}, {y0:.4g})", (x0, y0), xytext=(5, 5), textcoords="offset points")
        for i, (x, y) in enumerate(zip(x_history, y_history)):
            ax.annotate(
                f"c{i} ({x:.4g}, {y:.4g})",
                (x, y),
                xytext=(5, 5),
                textcoords="offset points"
            )