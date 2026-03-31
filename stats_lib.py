import numpy as np
import pandas as pd
from math import sqrt, pi, exp

def get_basic_stats(data):
    stats = {
            'mean': np.mean(data),
            'median': np.median(data),
            'var': np.var(data),
            'std': np.std(data),
            'min': min(data),
            'max': max(data),
            'len': len(data)
    }
    return stats

def get_gauss(x, mean, std):
    t1 = 1/(std*sqrt(2 * pi))
    t2 = exp((-1/2)*((x-mean)/std)**2)
    return t1 * t2

def get_quartiles(data):
    return np.percentile(data, [25, 50, 75])

def get_correlation(x_vals, y_vals):
    n = len(x_vals)
    sum_of_xy = sum([x * y for x, y in zip(x_vals, y_vals)])
    sum_x = sum(x_vals)
    sum_y = sum(y_vals)
    sum_x_squared = sum([x**2 for x in x_vals])
    sum_y_squared = sum([y**2 for y in y_vals])
    
    numerator = n * sum_of_xy - (sum_x * sum_y)
    denominator = sqrt((n * sum_x_squared - sum_x**2) * (n * sum_y_squared - sum_y**2))
    
    r = numerator / denominator
    return r

def get_lobf_lin(x_vals, y_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)

    A = np.array([
        [np.sum(x*x), np.sum(x)],
        [np.sum(x),   len(x)]
    ], dtype=float)

    C = np.array([np.sum(x*y), np.sum(y)], dtype=float)

    # Solve A * B = C
    a, b = np.linalg.solve(A, C)
    return a, b

def get_lobf_quad(x_vals, y_vals):
    x = np.array(x_vals, dtype=float)
    y = np.array(y_vals, dtype=float)
    A = np.array([
        [np.sum(x**4), np.sum(x**3), np.sum(x**2)],
        [np.sum(x**3), np.sum(x**2), np.sum(x)],
        [np.sum(x**2), np.sum(x), len(x)]
        ], dtype=float)
    C = np.array([
        np.sum(y * x**2), 
        np.sum(y*x), 
        np.sum(y)
        ], dtype=float)
    a, b, c = np.linalg.solve(A, C)
    return a, b, c

def get_y(x, slope, intercept):
    return slope * float(x) + intercept

def get_residuals(x_vals, y_vals):
    """ 
    Returns a list of residuals, takes in x_vals and y_vals
    """

    slope, intercept = get_lobf_lin(x_vals, y_vals)
    residuals = [y - get_y(x, slope, intercept) for x, y in zip(x_vals, y_vals)]
    return residuals

def get_outliers(x_vals, y_vals, n=2):
    """
    Return (outlier_residuals, outlier_indices) where an outlier is:
      abs(residual - mean_residual) > n * std_residual
    """
    residuals = get_residuals(x_vals, y_vals)
    stats = get_basic_stats(residuals)
    mean_r = stats['mean']
    std_r = stats['std']
    if std_r == 0:
        return [], []
    indices = []
    for i, r in enumerate(residuals):
        if abs(r - mean_r) > n * std_r:
            indices.append(i)
    return indices

def remove_outliers(x_vals, y_vals, n=2):
    """
    Remove points whose residual is more than n * std away from mean residual.
    Accepts pandas Series or array-like. Returns (x_filtered, y_filtered) as pandas Series.
    """
    x = pd.Series(x_vals).reset_index(drop=True)
    y = pd.Series(y_vals).reset_index(drop=True)

    indices = get_outliers(x, y, n=n)
    if not indices:
        return x, y

    mask = pd.Series(True, index=x.index)
    mask.iloc[indices] = False  
    return x[mask].reset_index(drop=True), y[mask].reset_index(drop=True)



def get_rmse(x_vals, y_vals):
    """
    Returns the root mean squared error
    """
    residuals = get_residuals(x_vals, y_vals)
    rmse = sqrt(sum(r**2 for r in residuals) / len(residuals))
    return rmse


def weighted_moving_average(data, weights = [1, 2, 3, 2, 1]):
    w_len = len(weights)
    w_sum = sum(weights)
    result = []

    for start in range(len(data) - w_len + 1):
        window = data[start:(start + w_len)]
        weighted_sum = sum(w * v for w, v in zip(weights, window))
        result.append(weighted_sum / w_sum)

    return result

def weighted_moving_average_2d(x_vals, y_vals, weights = [1, 2, 3, 2, 1]):
    x_vals = weighted_moving_average(x_vals, weights)
    y_vals = weighted_moving_average(y_vals, weights)
    return x_vals, y_vals


def fading_moving_average(data, weights=[0.5, 0.5]):
    w0, w1 = weights
    last_value = data[0]
    result = [last_value]

    for current in data[1:]:
        new_value = last_value * w0 + current * w1
        result.append(new_value)
        last_value = new_value

    return result

def new_point(value, mean, std):
    return (value-mean)/std

def normalize_list(list):
    mean = np.mean(list)
    std = np.std(list)
    new_list = [new_point(val, mean, std) for val in list]
    return new_list

def get_distance(x1, y1, x2, y2):
    return sqrt(((x2-x1)**2)+((y2-y1)**2))

def get_distances_for_multiple_points(x_vals, y_vals, points_x_vals, points_y_vals):
    return [
        [
            get_distance(x1, y1, x2, y2)
            for x1, y1 in zip(points_x_vals, points_y_vals)
        ]
        for x2, y2 in zip(x_vals, y_vals)
    ]

def apply_k_means_to_df(df, column_x_name, column_y_name):
    n_column_x_name = f'normalized_{column_x_name}'
    n_column_y_name = f'normalized_{column_y_name}'
    centroid_x_vals = []
    centroid_y_vals = []
    for centroid_idx in df['centroid_idx'].unique():
        rows = df.query(f'centroid_idx == {centroid_idx}')
        centroid_x_vals.append(rows[n_column_x_name].mean())
        centroid_y_vals.append(rows[n_column_y_name].mean())
    return centroid_x_vals, centroid_y_vals

def apply_centroids_to_df(df, column_x_name, column_y_name, centroid_x_vals, centroid_y_vals):
    centroid_counts = len(centroid_x_vals)
    n_column_x_name = f'normalized_{column_x_name}'
    n_column_y_name = f'normalized_{column_y_name}'
    distances_to_centroids = get_distances_for_multiple_points(df[n_column_x_name], df[n_column_y_name], centroid_x_vals, centroid_y_vals)
    df['distances_to_centroids'] = distances_to_centroids
    df['centroid_idx'] = df['distances_to_centroids'].apply(lambda x: x.index(min(x)))

def add_normalized_lists(df, column_x_name, column_y_name):
    n_column_x_name = f'normalized_{column_x_name}'
    n_column_y_name = f'normalized_{column_y_name}'
    df[n_column_x_name] = normalize_list(df[column_x_name])
    df[n_column_y_name] = normalize_list(df[column_y_name])

def full_k_means(df, column_x_name, column_y_name, centroid_x_vals, centroid_y_vals, iters=5):
    add_normalized_lists(df, column_x_name, column_y_name)
    apply_centroids_to_df(df, column_x_name, column_y_name, centroid_x_vals, centroid_y_vals)
    for iter in range(iters):
        centroid_x_vals, centroid_y_vals = apply_k_means_to_df(df, column_x_name, column_y_name)
        apply_centroids_to_df(df, column_x_name, column_y_name, centroid_x_vals, centroid_y_vals)

