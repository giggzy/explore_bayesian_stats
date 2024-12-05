import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        # Explore Grid Approximation

        One approach to Bayesian learning.
        """
    )
    return


@app.cell
def __(mo, np, plt):
    # Create synthetic data
    true_p = 0.7  # true probability of heads
    n_flips = 10
    simulated_flips = np.random.binomial(1, true_p, n_flips)  # simulate coin flips

    # Grid approximation
    grid_points = 200
    p_grid = np.linspace(0, 1, grid_points)  # possible values for p

    # Prior (uniform in this case)
    prior = np.ones(grid_points)  # uniform prior
    prior = prior / sum(prior)  # normalize


    # Likelihood
    likelihood = np.ones(grid_points)
    for data_point in simulated_flips:
        if data_point == 1:  # heads
            likelihood *= p_grid
        else:  # tails
            likelihood *= (1 - p_grid)

    # Posterior
    posterior = likelihood * prior
    posterior = posterior / sum(posterior)  # normalize

    # Plot
    plt.plot(p_grid, prior, label='Prior')
    plt.plot(p_grid, posterior, label='Posterior')
    plt.axvline(true_p, color='r', linestyle='--', label='True Value')
    plt.legend()
    plt.xlabel('Probability of Heads')
    plt.ylabel('Density')
    plt.title('Bayesian Learning with Grid Approximation')

    mo.vstack([
        grid_points,
        simulated_flips,
        #likelihood,
        len(likelihood),
        plt.show()] )
    #])
        #plt.show()] )

    #mo.vstack([plt.show()])
    return (
        data_point,
        grid_points,
        likelihood,
        n_flips,
        p_grid,
        posterior,
        prior,
        simulated_flips,
        true_p,
    )


@app.cell
def __(mo):
    mo.md(
        """
        ## Simulate 2 parameter bayesian learning

        Using a utility provided by the course materials to use grid approximation.
        """
    )
    return


@app.cell
def __(np, stats):
    # Model function required for simulation
    def linear_model(x: np.ndarray, intercept: float, slope: float) -> np.ndarray:
        return intercept + slope * x

    # Posterior function required for simulation
    def linear_regression_posterior(
        x_obs: np.ndarray,
        y_obs: np.ndarray,
        intercept_grid: np.ndarray,
        slope_grid: np.ndarray,
        likelihood_prior_std: float = 1.0
    ) -> np.ndarray:

        # Convert params to 1-d arrays
        if np.ndim(intercept_grid) > 1:
            intercept_grid = intercept_grid.ravel()

        if np.ndim(slope_grid):
            slope_grid = slope_grid.ravel()

        log_prior_intercept = stats.norm(0, 1).logpdf(intercept_grid)
        log_prior_slope = stats.norm(0, 1).logpdf(slope_grid)

        log_likelihood = np.array(
            [
                stats.norm(
                    intercept + slope * x_obs,
                    likelihood_prior_std
                ).logpdf(y_obs) for intercept, slope in zip(intercept_grid, slope_grid)
            ]
        ).sum(axis=1)

        # Posterior is equal to the product of likelihood and priors (here a sum in log scale)
        log_posterior = log_likelihood + log_prior_intercept + log_prior_slope

        # Convert back to natural scale
        return np.exp(log_posterior - log_posterior.max())
    return linear_model, linear_regression_posterior


@app.cell
def __(mo):
    mo.md(r"""### Simulating Posterior Updates""")
    return


@app.cell
def __(linear_model, linear_regression_posterior, np, plt, stats, utils):
    # Generate standardized regression data for demo
    np.random.seed(123)
    RESOLUTION = 100
    N_DATA_POINTS = 64

    # Ground truth parameters
    SLOPE = 0.5
    INTERCEPT = -1

    x = stats.norm().rvs(size=N_DATA_POINTS)
    y = INTERCEPT + SLOPE * x + stats.norm.rvs(size=N_DATA_POINTS) * 0.25

    slope_grid = np.linspace(-2, 2, RESOLUTION)
    intercept_grid = np.linspace(-2, 2, RESOLUTION)

    # Vary the sample size to show how the posterior adapts to more and more data
    # for n_samples in [0, 2, 4, 8, 16, 32, 64]:
    for n_samples in [0, 2, ]:
        # Run the simulation
        utils.simulate_2_parameter_bayesian_learning_grid_approximation(
            x_obs=x[:n_samples],
            y_obs=y[:n_samples],
            param_a_grid=intercept_grid,
            param_b_grid=slope_grid,
            true_param_a=INTERCEPT,
            true_param_b=SLOPE,
            model_func=linear_model,
            posterior_func=linear_regression_posterior,
            param_labels=['intercept', 'slope'],
            data_range_x=(-3, 3), data_range_y=(-3, 3)
        )

    plt.show()
    return (
        INTERCEPT,
        N_DATA_POINTS,
        RESOLUTION,
        SLOPE,
        intercept_grid,
        n_samples,
        slope_grid,
        x,
        y,
    )


@app.cell
def __(INTERCEPT, N_DATA_POINTS, SLOPE, mo, np, plt, stats, x):
    # play with a line

    line = np.linspace(-2, 2, 40)

    plt.plot(line, line)

    x1 = stats.norm().rvs(size=N_DATA_POINTS)
    y1 = INTERCEPT + SLOPE * x + stats.norm.rvs(size=N_DATA_POINTS) * 0.25

    mo.vstack([line, x1, max(x1), min(x1), y1, plt.show()])
    return line, x1, y1


@app.cell
def __(line, mo, x1):
    _df = mo.sql(
        f"""
        SELECT * FROM line;
        select * from x1
        """
    )
    return


@app.cell
def __():
    # Imports for the notebook

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats as stats
    import utils as utils
    return mo, np, plt, stats, utils


if __name__ == "__main__":
    app.run()
