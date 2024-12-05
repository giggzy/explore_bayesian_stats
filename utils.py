from matplotlib import pyplot as plt
import numpy as np

def simulate_2_parameter_bayesian_learning_grid_approximation(
    x_obs,
    y_obs,
    param_a_grid,
    param_b_grid,
    true_param_a,
    true_param_b,
    model_func,
    posterior_func,
    n_posterior_samples=3,
    param_labels=None,
    data_range_x=None,
    data_range_y=None,
):
    """General function for simulating Bayesian learning in a 2-parameter model
    using grid approximation.

    Parameters
    ----------
    x_obs : np.ndarray
        The observed x values
    y_obs : np.ndarray
        The observed y values
    param_a_grid: np.ndarray
        The range of values the first model parameter in the model can take.
        Note: should have same length as param_b_grid.
    param_b_grid: np.ndarray
        The range of values the second model parameter in the model can take.
        Note: should have same length as param_a_grid.
    true_param_a: float
        The true value of the first model parameter, used for visualizing ground
        truth
    true_param_b: float
        The true value of the second model parameter, used for visualizing ground
        truth
    model_func: Callable
        A function `f` of the form `f(x, param_a, param_b)`. Evaluates the model
        given at data points x, given the current state of parameters, `param_a`
        and `param_b`. Returns a scalar output for the `y` associated with input
        `x`.
    posterior_func: Callable
        A function `f` of the form `f(x_obs, y_obs, param_grid_a, param_grid_b)
        that returns the posterior probability given the observed data and the
        range of parameters defined by `param_grid_a` and `param_grid_b`.
    n_posterior_samples: int
        The number of model functions sampled from the 2D posterior
    param_labels: Optional[list[str, str]]
        For visualization, the names of `param_a` and `param_b`, respectively
    data_range_x: Optional len-2 float sequence
        For visualization, the upper and lower bounds of the domain used for model
        evaluation
    data_range_y: Optional len-2 float sequence
        For visualization, the upper and lower bounds of the range used for model
        evaluation.
    """
    param_labels = param_labels if param_labels is not None else ["param_a", "param_b"]
    data_range_x = (x_obs.min(), x_obs.max()) if data_range_x is None else data_range_x
    data_range_y = (y_obs.min(), y_obs.max()) if data_range_y is None else data_range_y

    # NOTE: assume square parameter grid
    resolution = len(param_a_grid)

    param_a_grid, param_b_grid = np.meshgrid(param_a_grid, param_b_grid)
    param_a_grid = param_a_grid.ravel()
    param_b_grid = param_b_grid.ravel()

    posterior = posterior_func(x_obs, y_obs, param_a_grid, param_b_grid)

    # Visualization
    _, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot Posterior over intercept and slope params
    plt.sca(axs[0])
    plt.contour(
        param_a_grid.reshape(resolution, resolution),
        param_b_grid.reshape(resolution, resolution),
        posterior.reshape(resolution, resolution),
        cmap="gray_r",
    )

    # Sample locations in parameter space according to posterior
    sample_idx = np.random.choice(
        np.arange(len(posterior)),
        p=posterior / posterior.sum(),
        size=n_posterior_samples,
    )

    param_a_list = []
    param_b_list = []
    for ii, idx in enumerate(sample_idx):
        param_a = param_a_grid[idx]
        param_b = param_b_grid[idx]
        param_a_list.append(param_a)
        param_b_list.append(param_b)

        # Add sampled parameters to posterior
        plt.scatter(param_a, param_b, s=60, c=f"C{ii}", alpha=0.75, zorder=20)

    # Add the true params to the plot for reference
    plt.scatter(
        true_param_a, true_param_b, color="k", marker="x", s=60, label="true parameters"
    )

    plt.xlabel(param_labels[0])
    plt.ylabel(param_labels[1])

    # Plot the current training data and model trends sampled from posterior
    plt.sca(axs[1])
    plt.scatter(x_obs, y_obs, s=60, c="k", alpha=0.5)

    # Plot the resulting model functions sampled from posterior
    xs = np.linspace(data_range_x[0], data_range_x[1], 100)
    for ii, (param_a, param_b) in enumerate(zip(param_a_list, param_b_list)):
        ys = model_func(xs, param_a, param_b)
        plt.plot(xs, ys, color=f"C{ii}", linewidth=4, alpha=0.5, label=f"sample {ii}")

    groundtruth_ys = model_func(xs, true_param_a, true_param_b)
    plt.plot(
        xs, groundtruth_ys, color="k", linestyle="--", alpha=0.5, label="true trend"
    )

    plt.xlim([data_range_x[0], data_range_x[1]])
    plt.xlabel("x value")

    plt.ylim([data_range_y[0], data_range_y[1]])
    plt.ylabel("y value")

    plt.title(f"N={len(y_obs)}")
    plt.legend(loc="upper left")
