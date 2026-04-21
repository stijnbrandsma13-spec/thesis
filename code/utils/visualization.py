import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",   # Computer Modern (LaTeX-like)
    "axes.grid": True,
})

def plot_cdf_3D(cdf: callable, x_range = (-2,2), y_range = (-2,2), n_grid = 50):
    x = np.linspace(*x_range, n_grid)
    y = np.linspace(*y_range, n_grid)
    X, Y = np.meshgrid(x, y)

    Z = np.zeros_like(X)

    for i in range(n_grid):
        for j in range(n_grid):
            Z[i, j] = cdf(np.array([X[i, j], Y[i, j]]))

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(
        X, Y, Z,
        cmap='jet',
        edgecolor='k',
        linewidth=0.5,
        antialiased=True,
        shade=True
    )

    ax.set_xlabel(r'$\beta_1$')
    ax.set_ylabel(r'$\beta_2$')
    ax.set_zlabel('CDF')

    ax.set_xlim(*x_range)
    ax.invert_xaxis() # This is a nicer direction
    ax.set_ylim(*y_range)
    ax.set_zlim(0, 1)

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    beta_support = np.array([
        [0, 0],
        [-1,  1],
        [ 1, 0],
        [ -2,  1]
    ])

    theta = np.array([0.2, 0.3, 0.3, 0.2])

    def example_cdf(beta, boolean):
        if boolean:
            beta = np.asarray(beta)
            mask = (beta_support <= beta).all(axis=1)
            return theta[mask].sum()
        
    plot_cdf_3D(lambda beta: example_cdf(beta, True))