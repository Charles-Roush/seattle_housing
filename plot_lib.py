import re
from pathlib import Path
import math

import numpy as np
import matplotx  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set style
style = matplotx.styles.duftify(matplotx.styles.dracula)
mpl.style.use(style)

# Set style
font = {'weight' : 'bold'}

mpl.rc('font', **font)

mpl.rcParams['text.color'] = 'white'
mpl.rcParams['axes.labelcolor'] = 'white'
mpl.rcParams['xtick.color'] = 'white'
mpl.rcParams['ytick.color'] = 'white'
mpl.rcParams['axes.edgecolor'] = 'white'

def safe_filename(s: str) -> str:
    """Remove characters that are invalid in filenames."""
    return re.sub(r'[<>:"/\\|?*\n\r]+', '_', s).strip()



def fit_poly(x_vals, y_vals, degree=1):
    """
    Fit polynomial of given degree using normal equations.
    Returns coefficients in increasing-power order: [c0, c1, c2, ...]
    (so y = c0 + c1*x + c2*x**2 + ...)
    """
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    m = degree + 1

    # Build the normal-equation matrix A and right-hand side C.
    # A[i, j] = sum(x ** (i + j))
    A = np.array([[np.sum(x ** (i + j)) for j in range(m)] for i in range(m)], dtype=float)
    C = np.array([np.sum(y * (x ** i)) for i in range(m)], dtype=float)

    coeffs = np.linalg.solve(A, C)
    return coeffs


def eval_poly(coeffs, xs):
    """
    Evaluate polynomial with coeffs in increasing-power order.
    Uses np.polyval by reversing coefficients to high->low order.
    """
    coeffs = np.array(coeffs, dtype=float)
    xs = np.array(xs, dtype=float)
    return np.polyval(coeffs[::-1], xs)


# -------------------------
# Plot classes
# -------------------------
class Plot:
    def __init__(self, x_vals, y_vals=None, title="", x_title="", y_title="", label=None):
        self.x_vals = x_vals
        self.y_vals = y_vals
        self.title = title
        self.x_title = x_title
        self.y_title = y_title
        self.label = label


    def plot(self, ax):
        """Set labels and call the actual plot function"""
        if self.title:
            ax.set_title(self.title)
        if self.x_title:
            ax.set_xlabel(self.x_title)
        if self.y_title:
            ax.set_ylabel(self.y_title)
        self._plot(ax)

    def _plot(self, ax):
        raise NotImplementedError("Subclasses must implement this method")


class LinePlot(Plot):
    def _plot(self, ax):
        if self.x_vals is not None:
            ax.plot(self.x_vals, self.y_vals, label=self.label)
        else:
            ax.plot(self.y_vals, label=self.label)


class ScatterPlot(Plot):
    def _plot(self, ax):
        if self.y_vals is not None:
            ax.scatter(self.x_vals, self.y_vals, label=self.label)
        else:
            ax.scatter(range(len(self.x_vals)), self.x_vals, label=self.label)


class HistPlot(Plot):
    def __init__(self, x_vals, bins=10, **kwargs):
        super().__init__(x_vals, **kwargs)
        self.bins = bins

    def _plot(self, ax):
        ax.hist(self.x_vals, bins=self.bins, label=self.label)


class PiePlot(Plot):
    def __init__(self, x_vals, labels=None, **kwargs):
        super().__init__(x_vals, **kwargs)
        self.labels = labels

    def _plot(self, ax):
        ax.pie(self.x_vals, labels=self.labels, label=self.label)


class BoxPlot(Plot):
    def _plot(self, ax):
        ax.boxplot(self.x_vals, label=self.label)


class PolyBestFitPlot(Plot):
    """
    Plot a polynomial best-fit curve.
    coeffs: optional. If not provided, the class will compute them using fit_poly.
    coeffs format: increasing-power order [c0, c1, c2, ...]
    """
    def __init__(self, x_vals, y_vals, degree=1, coeffs=None, **kwargs):
        super().__init__(x_vals, y_vals, **kwargs)
        self.degree = degree
        self.coeffs = None if coeffs is None else np.array(coeffs, dtype=float)

    def _plot(self, ax):
        if self.x_vals is None or self.y_vals is None:
            return

        if self.coeffs is None:
            self.coeffs = fit_poly(self.x_vals, self.y_vals, degree=self.degree)

        x1, x2 = float(min(self.x_vals)), float(max(self.x_vals))
        xs = np.linspace(x1, x2, 300)
        ys = eval_poly(self.coeffs, xs)

        ax.plot(xs, ys, linestyle='--', label=self.label)


class PolyBestFitStdPlot(Plot):
    """
    Plot polynomial best-fit with two offset lines at +/- std (simple, line offsets).
    std is an additive offset to y (not a rigorous confidence interval).
    """
    def __init__(self, x_vals, y_vals, degree=1, coeffs=None, std=0.0, **kwargs):
        super().__init__(x_vals, y_vals, **kwargs)
        self.degree = degree
        self.coeffs = None if coeffs is None else np.array(coeffs, dtype=float)
        self.std = float(std)

    def _plot(self, ax):
        if self.x_vals is None or self.y_vals is None:
            return

        if self.coeffs is None:
            self.coeffs = fit_poly(self.x_vals, self.y_vals, degree=self.degree)

        x1, x2 = float(min(self.x_vals)), float(max(self.x_vals))
        xs = np.linspace(x1, x2, 300)
        ys = eval_poly(self.coeffs, xs)

        ax.plot(xs, ys, linestyle='--')
        ax.plot(xs, ys + self.std, linestyle=':', color='gray')
        ax.plot(xs, ys - self.std, linestyle=':', color='gray', label=self.label)


# -------------------------
# SubPlot manager
# -------------------------
class SubPlot:
    def __init__(self, plot_groups, rows=None, columns=None, size=(16, 12)):
        """
        plot_groups: list of lists of Plot objects.
        Each inner list is drawn on the same subplot.
        """
        if not isinstance(plot_groups, (list, tuple)):
            raise TypeError("plot_groups must be a list of lists of Plot objects")
        self.plot_groups = plot_groups
        self.n = len(plot_groups)

        # Determine rows and columns if not provided
        if rows is None and columns is None:
            self.rows = int(math.sqrt(self.n)) or 1
            self.columns = math.ceil(self.n / self.rows)
        elif rows is not None and columns is None:
            self.rows = rows
            self.columns = math.ceil(self.n / rows)
        elif rows is None and columns is not None:
            self.columns = columns
            self.rows = math.ceil(self.n / columns)
        else:
            self.rows = rows
            self.columns = columns

        self.fig, self.axes = plt.subplots(self.rows, self.columns, squeeze=False, figsize=size)

    def plot(self, save_dir="plots"):
        # Ensure output directory exists
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        axes_flat = self.axes.flatten()
        for ax, plot_group in zip(axes_flat, self.plot_groups):
            for plot in plot_group:
                plot.plot(ax)

        # Hide unused axes
        for ax in axes_flat[len(self.plot_groups):]:
            self.fig.delaxes(ax)

        plt.tight_layout()
        plt.legend()
        
        file_title = 'no_title'
        if self.plot_groups and self.plot_groups[0] and getattr(self.plot_groups[0][0], "title", ""):
            file_title = safe_filename(self.plot_groups[0][0].title)
        path = Path(save_dir) / f'{file_title}.png'
        self.fig.savefig(path, bbox_inches="tight")
        plt.show()
        plt.close(self.fig)


def plot_lobf(x_vals, y_vals, quadratic=False):
    import stats_lib as sl
    import numpy as np
    import matplotlib.pyplot as plt

    # get coefficients
    if quadratic:
        a, b, c = sl.get_lobf_quad(x_vals, y_vals)
    else:
        a, b = sl.get_lobf_lin(x_vals, y_vals)

    # make smooth xs for plotting
    xs = np.linspace(min(x_vals), max(x_vals), 300)

    # compute ys
    if quadratic:
        ys = a*xs**2 + b*xs + c
    else:
        ys = a*xs + b

    # plot
    plt.scatter(x_vals, y_vals)
    plt.plot(xs, ys, '--')
    plt.show()

def plot_lobf_with_std(x_vals, y_vals, quadratic=False, n_std=2):
    import stats_lib as sl
    import numpy as np
    import matplotlib.pyplot as plt

    x_vals = np.array(x_vals, dtype=float)
    y_vals = np.array(y_vals, dtype=float)

    if quadratic:
        a, b, c = sl.get_lobf_quad(x_vals, y_vals)
        y_fit = a*x_vals**2 + b*x_vals + c
        xs = np.linspace(min(x_vals), max(x_vals), 300)
        ys_fit = a*xs**2 + b*xs + c
    else:
        a, b = sl.get_lobf_lin(x_vals, y_vals)
        y_fit = a*x_vals + b
        xs = np.linspace(min(x_vals), max(x_vals), 300)
        ys_fit = a*xs + b

    residuals = y_vals - y_fit
    std = np.std(residuals)

    ys_upper = ys_fit + n_std*std
    ys_lower = ys_fit - n_std*std

    plt.scatter(x_vals, y_vals)
    plt.plot(xs, ys_fit, '--', label='Best fit')
    plt.plot(xs, ys_upper, ':', color='gray', label=f'+{n_std} std')
    plt.plot(xs, ys_lower, ':', color='gray', label=f'-{n_std} std')
    plt.legend()
    plt.show()

def plot_centroid_groups(df, x_column_name, y_column_name):
    plots = []
    for idx in df['centroid_idx'].unique():
        rows = df.query(f'centroid_idx=={idx}')
        x_vals = rows[x_column_name]
        y_vals = rows[y_column_name]
        sp = ScatterPlot(x_vals, y_vals)
        plots.append(sp)
    SubPlot([plots]).plot()
    