import numpy as np
import matplotlib.pyplot as plt

class DTW:
    """
    A class to represent the Dynamic Time Warping (DTW) results.
    
    Attributes:
    -----------
    distance : float
        The DTW distance between the two time series.
    matrix : numpy.ndarray or None
        The accumulated cost matrix, if keep_matrix is True, otherwise None.
    path : list of tuples or None
        The optimal warping path, if keep_path was is True, otherwise None.
    """
    def __init__(self, distance, matrix=None, path=None):
        self.distance = distance
        self.matrix = matrix
        self.path = path

def __dtw(X, Y, bounds=None, keep_matrix=False, keep_path=False, plot_path=False) -> DTW:
    """
    Compute the Dynamic Time Warping (DTW) distance between two time series, X and Y, within specified bounds.
    Optionally, return the accumulated cost matrix and/or the warping path as attributes of a DTW object.

    Parameters:
    -----------
    X : numpy.ndarray
        First input time series (shape: (num_features, len_X)).
    Y : numpy.ndarray
        Second input time series (shape: (num_features, len_Y)).
    bounds : numpy.ndarray, optional
        2D array with lower and upper bounds for each column of the accumulated cost matrix (shape: (2, len_Y)).
        Otherwise, a full window is considered (default).
    keep_matrix : bool, optional
        If True, store the accumulated cost matrix in the returned DTW object (default: False).
    keep_path : bool, optional
        If True, store the warping path in the returned DTW object (default: False).

    Returns:
    --------
    result : DTW
        A DTW object with the following attributes:
        - distance: The DTW distance (float).
        - matrix: The accumulated cost matrix (numpy.ndarray), if keep_matrix is True, otherwise None.
        - path: The warping path (list of tuples), if keep_path is True, otherwise None.
    """
    # Initialize dimensions and bounds if none were provided
    N, M = X.shape[1], Y.shape[1]
    if bounds is None:
        bounds = np.ones((2, M), dtype=np.int32)
        bounds[1, :] = N

    # Initialize accumulated cost matrix and preallocate memory
    acc_cost = np.full((2, N+1), np.inf, dtype=np.float64)
    acc_cost[1, 0] = 0

    # Initialize accumulated matrix for later use if needed
    acc_mat = None
    if keep_matrix or keep_path:
        acc_mat = np.zeros((N, M), dtype=np.float64)

    # Calculate the accumulated cost for each cell within the given bounds
    for m, (lower, upper) in enumerate(bounds.T):
        acc_cost[0, :] = acc_cost[1, :]
        acc_cost[1, :] = np.inf

        # Calculate the squared Euclidean distance for each relevant row in current column
        costs = np.sum((X[:, lower-1:upper] - Y[:, m][:, np.newaxis])**2, axis=0)

        # Update the accumulated cost matrix with the minimum cost path
        for idx, n in enumerate(range(lower, upper+1)):
            acc_cost[1, n] = costs[idx] +\
                                min(acc_cost[0, n], 
                                    acc_cost[0, n - 1],
                                    acc_cost[1, n - 1])

        # Store the accumulated cost in the accumulated matrix if needed
        if keep_matrix or keep_path:
            acc_mat[:, m] = acc_cost[1, 1:]

    # Calculate the warping path if needed
    if keep_path:
        path = np.zeros(((N+M-1), 2), dtype=np.int32)
        i = N - 1
        j = M - 1

        idx = N+M-1
        while i > 0 or j > 0:
            idx -= 1
            path[idx] = [i, j]
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                min_idx = np.argmin([acc_mat[i-1, j], 
                                     acc_mat[i, j-1], 
                                     acc_mat[i-1, j-1]])
                if min_idx == 0:
                    i -= 1
                elif min_idx == 1:
                    j -= 1
                else:
                    i -= 1
                    j -= 1

        path = path[idx-1:]
        
    # Prepare return values
    distance = acc_cost[1, -1]
    matrix = acc_mat if keep_matrix else None
    path = path if keep_path else None

    if plot_path:
        fig, ax = plt.subplots()
        
        ys, xs = [p[0] + 0.5 for p in path], [p[1] + 0.5 for p in path]
        ax.matshow(acc_mat)
        ax.plot(xs, ys, color="black", linewidth=2)
        ax.set_aspect('equal')

        plt.show()
                

    # Return DTW object
    return DTW(distance, matrix, path)

def dtw(X, Y, bounds=None, keep_matrix=False, keep_path=False, multivar_method='independent', plot_path=False) -> DTW:
    """
    Compute the Dynamic Time Warping (DTW) distance between two time series, X and Y, within specified
    bounds. The method supports both univariate and multivariate time series. 
    Optionally, return the accumulated cost matrix and/or the warping path as attributes of a DTW object.
    For multivariate times series, either a dependent or independent method can be used.

    Parameters:
    -----------
    X : numpy.ndarray
        First input time series (shape: (num_features, len_X)).
    Y : numpy.ndarray
        Second input time series (shape: (num_features, len_Y)).
    bounds : numpy.ndarray, optional
        2D array with lower and upper bounds for each column of the accumulated cost matrix (shape: (2, len_Y)).
        Otherwise, a full window is considered (default).
    keep_matrix : bool, optional
        If True, store the accumulated cost matrix in the returned DTW object (default: False).
    keep_path : bool, optional
        If True, store the warping path in the returned DTW object (default: False).
    multivar_method : str, optional
        Method for handling multivariate time series: 'dependent' (default) or 'independent'.

    Returns:
    --------
    result : DTW
        A DTW object with the following attributes:
        - distance: The DTW distance (float).
        - matrix: The accumulated cost matrix (numpy.ndarray or list of numpy.ndarray if multivar_method = "independent"), if keep_matrix is True,
                  otherwise None.
        - path: The warping path (list of tuples or list of lists of tuples if multivar_method = "independent"), if keep_path is True, otherwise None.

    Raises:
    -------
    ValueError:
        If the input time series do not have the same number of dimensions, or if an unexpected value is provided for
        the multivar_method parameter.
    """
    # Ensure input time series have at least two dimensions
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)

    # Check if input time series have the same number of dimensions
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'Expected series with same number of dimensions, got {X.shape[0]} and {Y.shape[0]}')    

    # If time series have only one feature or method is 'dependent', compute DTW using __dtw function
    if X.shape[0] == Y.shape[0] == 1 or multivar_method == 'dependent':
        return __dtw(X=X, Y=Y, bounds=bounds, keep_matrix=keep_matrix, keep_path=keep_path, plot_path=plot_path)

    # If method is 'independent', compute DTW for each dimension independently and aggregate results
    elif multivar_method == 'independent':
        distance, matrix, path = 0, [], []

        for dim in range(X.shape[0]):
            res = __dtw(X = X[dim, :][np.newaxis, :], Y=Y[dim, :][np.newaxis, :], bounds=bounds, keep_matrix=keep_matrix, keep_path=keep_path)
            distance += res.distance
            matrix.append(res.matrix)
            path.append(res.path)

        return DTW(distance, matrix, path)

    # Raise an error if an unexpected value for multivar_method is provided
    else:
        raise ValueError(f"Unexpected value for multivar_method, expected 'dependent' or 'independent', got {multivar_method}")

def itakura(N, M, s):
    """
    Compute the Itakura parallelogram bounds for Dynamic Time Warping (DTW).

    Parameters:
    -----------
    N : int
        Length of the first time series.
    M : int
        Length of the second time series.
    s : float
        Slope of the Itakura parallelogram. Controls the allowed warping.

    Returns:
    --------
    bounds : numpy.ndarray
        The Itakura parallelogram bounds for DTW (shape: (2, M)).
    """
    # Compute the base indices for the second time series
    m = np.arange(M, dtype=np.int32)

    # Compute the lower and upper bounds for each column of the accumulated cost matrix
    lower_bound = np.maximum(s * (m - M) + N, (1 / s) * m)
    upper_bound = np.floor(np.minimum((1 / s) * (m - M) + N, (s * m) + 1))

    # Combine the lower and upper bounds into a single array and clip the values to the range [0, N]
    bounds = np.array([lower_bound, upper_bound], dtype=np.int32)
    bounds = np.clip(bounds, 0, N)

    return bounds

def sakoe_chiba(N, M, w):
    """
    Compute the Sakoe-Chiba band bounds for Dynamic Time Warping (DTW).

    Parameters:
    -----------
    N : int
        Length of the first time series.
    M : int
        Length of the second time series.
    w : int
        Window size of the Sakoe-Chiba band. Controls the allowed warping.

    Returns:
    --------
    bounds : numpy.ndarray
        The Sakoe-Chiba band bounds for DTW (shape: (2, M)).
    """
    # Compute the base indices for the second time series
    m = np.arange(M, dtype=np.int32)

    # Compute the coefficient based on the lengths of the time series
    coeff = N / M

    # Compute the lower and upper bounds for each column of the accumulated cost matrix
    lower_bound = (coeff * (m - w)).astype(np.int32)
    upper_bound = (coeff * (m + w)).astype(np.int32)

    # Combine the lower and upper bounds into a single array and clip the values to the range [0, N]
    bounds = np.array([lower_bound, upper_bound], dtype=np.int32)
    bounds = np.clip(bounds, 0, N)

    return bounds

def __cdtw(X, Y, method, window_size=np.inf, keep_matrix=False, keep_path=False, multivar_method='independent') -> DTW:
    """
    Compute the Constrained Dynamic Time Warping (CDTW) distance between two time series, X and Y, 
    using a specified method for generating warping constraints. The method supports both univariate 
    and multivariate time series. Optionally, return the accumulated cost matrix
    and/or the warping path as attributes of a DTW object.

    Parameters:
    -----------
    X : numpy.ndarray
        First input time series (shape: (num_features, len_X)).
    Y : numpy.ndarray
        Second input time series (shape: (num_features, len_Y)).
    method : function
        Function to compute the bounds (e.g., itakura, sakoe_chiba).
    window_size : int, optional
        The window size parameter for the method (default: np.inf).
    keep_matrix : bool, optional
        If True, store the accumulated cost matrix in the returned DTW object (default: False).
    keep_path : bool, optional
        If True, store the warping path in the returned DTW object (default: False).
    multivar_method : str, optional
        Method for handling multivariate time series: 'dependent' or 'independent' (default).

    Returns:
    --------
    result : DTW
        A DTW object with the following attributes:
        - distance: The CDTW distance (float).
        - matrix: The accumulated cost matrix (numpy.ndarray or list of numpy.ndarray if multivar_method = "independent"), 
            if keep_matrix is True, otherwise None.
        - path: The warping path (list of tuples or list of lists of tuples if multivar_method = "independent"), 
            if keep_path is True, otherwise None.
    """
    # Get the lengths of the input time series
    N, M = X.shape[1], Y.shape[1]

    # Compute the bounds using the provided method and window size, then clip the values to the range [1, N]
    bounds = method(N, M, window_size) + 1
    bounds = np.clip(bounds, 1, N)

    # Compute the DTW distance with the specified bounds and options
    return dtw(X, Y, bounds=bounds, keep_matrix=keep_matrix, keep_path=keep_path, multivar_method=multivar_method)

def cdtw(X, Y, method, window_size=None, keep_matrix=False, keep_path=False, multivar_method='independent') -> DTW:
    """
    Compute the Constrained Dynamic Time Warping (CDTW) distance between two time series, X and Y, 
    using a specified method for generating warping constraints. The method supports both univariate 
    and multivariate time series. Optionally, return the accumulated cost matrix
    and/or the warping path as attributes of a DTW object.

    Parameters:
    -----------
    X : numpy.ndarray
        First input time series (shape: (num_features, len_X)).
    Y : numpy.ndarray
        Second input time series (shape: (num_features, len_Y)).
    method : function
        Function to compute the bounds (e.g., itakura, sakoe_chiba).
    window_size : float or int, optional
        The window size parameter for the method (default: np.inf).
    keep_matrix : bool, optional
        If True, store the accumulated cost matrix in the returned DTW object (default: False).
    keep_path : bool, optional
        If True, store the warping path in the returned DTW object (default: False).
    multivar_method : str, optional
        Method for handling multivariate time series: 'dependent' (default) or 'independent'.
        
    Returns:
    --------
    result : DTW
        A DTW object with the following attributes:
        - distance: The CDTW distance (float).
        - matrix: The accumulated cost matrix (numpy.ndarray or list of numpy.ndarray if multivar_method = "independent"), 
            if keep_matrix is True, otherwise None.
        - path: The warping path (list of tuples or list of lists of tuples if multivar_method = "independent"), 
            if keep_path is True, otherwise None.
    """
    # Ensure the input time series are at least 2D arrays
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)

    # Check if both time series have the same number of dimensions
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'Expected series with same number of dimensions, got {X.shape[0]} and {Y.shape[0]}')

    # If no window size is given, run full DTW
    if window_size is None:
        window_size = max(X.shape[1], Y.shape[1])

    # Compute the CDTW distance with the specified bounds and options
    return __cdtw(X=X, Y=Y, method=method, window_size=window_size, keep_matrix=keep_matrix, keep_path=keep_path, multivar_method=multivar_method)

def project_path(path, scaling_factor, radius):
    """
    Project a warping path onto a larger grid to generate a bounding region for a lower-resolution time series
    alignment. Used as a helping function for the fastdtw method.

    Parameters:
    -----------
    path : list of tuples
        The warping path represented as a list of coordinate tuples (i, j).
    scaling_factor : int
        The factor by which the grid size should be scaled.
    radius : int
        The radius around each projected point that should be included in the bounds.

    Returns:
    --------
    bounds : numpy.ndarray
        The projected bounding region as a 2D numpy array of shape (2, scaling_factor * M), where the first row
        represents the lower bounds and the second row represents the upper bounds.
    """
    # Convert the path to a numpy array
    path = np.array(path, dtype=np.int32)

    # Calculate the dimensions of the projected grid
    height, width = (path[-1, 0] + 1) * scaling_factor - 1, (path[-1, 1] + 1) * scaling_factor - 1

    # Scale path
    path = path * scaling_factor

    # Calculate intermediate points and join in single array
    all_points = np.zeros((len(path)*2-1, 2), dtype=np.intp)
    all_points[::2, :] = path
    all_points[1::2, :] = path[:-1] + np.floor((path[1:] - path[:-1]) / 2).astype(np.int32)

    # Initialize lower and upper bounds with infinite and negative infinite values, respectively
    lower_bound = np.full((all_points.shape[0], width+1), height+1)
    upper_bound = np.full((all_points.shape[0], width+1), -1)

    # Iterate through all points
    for idx, (i, j) in enumerate(all_points):
        left = max(j - radius, 0)
        right = min(j + scaling_factor + radius, width+1)

        lower_bound[idx, left:right] = i - radius
        upper_bound[idx, left:right] = i + radius + 1

    # Combine the lower and upper bounds into a single numpy array
    bounds = np.array([np.min(lower_bound, axis=0), np.max(upper_bound, axis=0)])

    # Ensure the bounds are of integer type and clip them to the height range
    bounds = bounds.astype(np.int32)
    bounds = np.clip(bounds, 0, height)

    return bounds

def shrink(series):
    """
    Shrink the input time series by a factor of 2 using a simple averaging method. The function takes an
    input time series and computes the average of each pair of adjacent points to create a new time series
    with half the original length. Used as a helping function for the fastdtw method.

    Parameters:
    -----------
    series : array-like
        The input time series, which can be a list, numpy array, or other array-like object. The shape of
        the time series should be (num_dimensions, length).

    Returns:
    --------
    shrunk : numpy.ndarray
        The shrunken time series with half the original length, of shape (num_dimensions, ⌈ length / 2 ⌉).
    """
    # Convert the input series to a numpy array
    series = np.array(series)

    # If the number of columns (time series length) is odd, append the last column to make it even
    if series.shape[1] % 2 == 1:
        app = series[:, -1].reshape(series.shape[0], 1)
        series = np.hstack((series, app))

    # Compute the average of each pair of adjacent points to create the shrunken time series
    shrunk = ((series[:, 1::2] + series[:, ::2]) / 2.0)

    return shrunk

def __fastdtw(X, Y, radius, keep_matrix=False, keep_path=False, plot_path=False) -> DTW:
    """
    Calculate the approximate Dynamic Time Warping (DTW) distance between two time series X and Y using
    the FastDTW algorithm. The FastDTW algorithm uses a multi-scale approach to compute an approximate 
    path and then refines the path using the original time series.

    Parameters:
    -----------
    X : array-like
        The first input time series, of shape (num_dimensions, length_X).
    Y : array-like
        The second input time series, of shape (num_dimensions, length_Y).
    radius : int
        The size of the search window used in the FastDTW algorithm.
    keep_matrix : bool, optional
        If True, the accumulated cost matrix is included in the returned DTW object (default: False).
    keep_path : bool, optional
        If True, the optimal warping path is included in the returned DTW object (default: False).
        
    Returns:
    --------
    DTW
        A DTW object containing the DTW distance, accumulated cost matrix (if keep_matrix is True),
        and the warping path (if keep_path is True).
    """
    # Determine the length of the input time series
    N, M = X.shape[1], Y.shape[1]

    # Minimum size for the FastDTW algorithm to be applied
    min_size = radius + 2

    # If either time series is too short, use the standard DTW algorithm
    if N <= min_size or M <= min_size:
        return dtw(X, Y, keep_matrix=keep_matrix, keep_path=True, multivar_method='dependent', plot_path=plot_path)

    # Shrink the input time series by a factor of 2
    shrunk_X, shrunk_Y = shrink(X), shrink(Y)

    # Recursively apply the FastDTW algorithm on the shrunken time series
    res = __fastdtw(shrunk_X, shrunk_Y, radius, keep_matrix=keep_matrix, keep_path=True, plot_path=plot_path)

    # Project the approximate warping path from the shrunken time series onto the original time series
    bounds = project_path(res.path, scaling_factor=2, radius=radius)

    # Adjust and clip the bounds to the correct size
    bounds = bounds[:, :M] + 1
    bounds = np.clip(bounds, 0, N)

    # Use the standard DTW algorithm on the original time series, with the projected bounds
    return dtw(X, Y, bounds, keep_matrix=keep_matrix, keep_path=keep_path, multivar_method='dependent', plot_path=plot_path)

def fastdtw(X, Y, radius, keep_matrix=False, keep_path=False, multivar_method='independent', plot_path=False) -> DTW:
    """
    Calculate the approximate Dynamic Time Warping (DTW) distance between two time series X and Y using
    the FastDTW algorithm. The FastDTW algorithm uses a multi-scale approach to compute an approximate
    path and then refines the path using the original time series.
    This function supports both univariate and multivariate time series.

    Parameters:
    -----------
    X : array-like
        The first input time series, of shape (num_dimensions, length_X).
    Y : array-like
        The second input time series, of shape (num_dimensions, length_Y).
    radius : int
        The size of the search window used in the FastDTW algorithm.
    keep_matrix : bool, optional
        If True, the accumulated cost matrix is included in the returned DTW object (default: False).
    keep_path : bool, optional
        If True, the optimal warping path is included in the returned DTW object (default: False).
    multivar_method : str, optional
        Specifies the method for handling multivariate time series (default: 'independent').
        'dependent': Treat the dimensions as dependent, and calculate a single DTW distance.
        'independent': Treat the dimensions as independent, and calculate a separate DTW distance for each dimension.
        
    Returns:
    --------
    DTW
        A DTW object containing the DTW distance, accumulated cost matrix (if keep_matrix is True),
        and the warping path (if keep_path is True).
    """
    # Ensure the input time series are at least 2-dimensional arrays
    X, Y = np.atleast_2d(X), np.atleast_2d(Y)

    # Check if the input time series have the same number of dimensions
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f'Expected series with same number of dimensions, got {X.shape[0]} and {Y.shape[0]}')

    # If univariate time series or 'dependent' method is specified, call fastdtw_method
    if X.shape[0] == Y.shape[0] == 1 or multivar_method == 'dependent':
        return __fastdtw(X=X, Y=Y, radius=radius, keep_matrix=keep_matrix, keep_path=keep_path, plot_path=plot_path)

    # If 'independent' method is specified, calculate DTW distance for each dimension separately
    elif multivar_method == 'independent':
        distance, matrix, path = 0, [], []

        # Iterate over dimensions and call fastdtw_method for each dimension
        for dim in range(X.shape[0]):
            res = __fastdtw(X = X[dim, :][np.newaxis, :], Y=Y[dim, :][np.newaxis, :], radius=radius, keep_matrix=keep_matrix, keep_path=keep_path)
            distance += res.distance
            matrix.append(res.matrix)
            path.append(res.path)

        # Return a DTW object with the accumulated results
        return DTW(distance, matrix, path)

    # Raise an error if an unsupported value for multivar_method is specified
    else:
        raise ValueError(f"Unexpected value for multivar_method, expected 'dependent' or 'independent', got {multivar_method}")