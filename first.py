import marimo

__generated_with = "0.9.30"
app = marimo.App(width="medium")


@app.cell
def __(mo):
    mo.md(
        r"""
        # Play with Marimo

        Some testing of Marimo.
        """
    )
    return


@app.cell
def __():
    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sns
    import polars as pl
    import numpy as np

    return mo, np, pl, plt, sns


@app.cell
def __(np, plt, sns):
    data = np.random.normal(loc=50, scale=10, size=1000)
    sns.histplot(data, kde=True, bins=10)
    plt.gca()
    return (data,)


@app.cell
def __(data, mo):
    _df = mo.sql(
        f"""
        SELECT * FROM data
        """
    )
    return


@app.cell
def __(np, plt, sns):
    # Normal/Gaussian - symmetric, bell-shaped
    normal = np.random.normal(loc=0.5, scale=1, size=1000)

    # Uniform - equal probability across range
    uniform = np.random.uniform(low=0, high=50, size=1000)

    # Exponential - decay processes, waiting times
    exp = np.random.exponential(scale=10, size=1000)

    # Poisson - count data, rare events
    poisson = np.random.poisson(lam=10, size=1000)

    # Beta - probabilities, proportions (between 0 and 1)
    beta = np.random.beta(a=2, b=5, size=1000)

    # Gamma - waiting times, positive continuous data
    gamma = np.random.gamma(shape=2, scale=2, size=1000)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Common Probability Distributions", fontsize=16)

    # Plot the data
    norm_plt = axes[0, 0]
    norm_plt.set_title("Normal")
    sns.histplot(normal, kde=True, bins=10, ax=norm_plt)

    sns.histplot(uniform, kde=True, bins=10, ax=axes[0, 1])
    sns.histplot(exp, kde=True, bins=10, ax=axes[0, 2])
    sns.histplot(poisson, kde=True, bins=10, ax=axes[1, 0])
    sns.histplot(beta, kde=True, bins=10, ax=axes[1, 1])
    sns.histplot(gamma, kde=True, bins=10, ax=axes[1, 2])

    # get a new plot
    return axes, beta, exp, fig, gamma, norm_plt, normal, poisson, uniform


@app.cell
def __(np, plt, sns):
    # Create figure with two subplots
    my_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Binomial distribution
    n, p = 20, 0.5  # 20 trials, probability 0.5
    binomial_data = np.random.binomial(n, p, 1000)
    sns.histplot(binomial_data, discrete=True, ax=ax1)
    ax1.set_title("Binomial Distribution\n(e.g., number of heads in 20 flips)")

    # Normal distribution
    mu, sigma = 10, 2  # mean=10, standard deviation=2
    normal_data = np.random.normal(mu, sigma, 1000)
    sns.histplot(normal_data, ax=ax2)
    ax2.set_title("Normal Distribution\n(e.g., heights in a population)")

    plt.tight_layout()
    plt.show()
    return ax1, ax2, binomial_data, mu, my_fig, n, normal_data, p, sigma


@app.cell
def __(mo):
    mo.md(
        r"""
        ## Explore the Howell Dataset

        This is a dataset of weight and height of a population
        """
    )
    return


@app.cell
def __(pl, plt, sns):
    # Open the Howell dataset
    howell = pl.read_csv("data/Howell1.csv", separator=";")

    fig_h, axes_h = plt.subplots(1, 2, figsize=(15, 5))

    # print(howell.columns)
    adults = howell.filter(pl.col("age") > 18)
    children = howell.filter(pl.col("age") <= 18)

    # All age vs height
    sns.scatterplot(data=adults, y="weight", x="height", ax=axes_h[0], color="blue", label="Adults")
    sns.scatterplot(data=children, y="weight", x="height", ax=axes_h[0], color="red", label="Children")
    axes_h[0].set_title("Entire Dataset")

    # Just Adults
    sns.scatterplot(data=adults, y="weight", x="height", ax=axes_h[1], color="blue", label="Adults")

    axes_h[1].set_title("Adults")

    plt.show()
    howell
    return adults, axes_h, children, fig_h, howell


@app.cell(hide_code=True)
def __(mo, plt):
    mo.md(
        r"""
        ## Causal Diagram for Height and Weight with Unobserved Confounder

        This diagram illustrates how Height (H) influences Weight (W) and how an Unobserved variable influences Height.
        """
    )

    import daft

    # Create a new PGM
    pgm = daft.PGM()

    # Add nodes - coordinates are (x, y)
    # observed=False creates dashed nodes
    pgm.add_node("H", "H", 3, 3)
    pgm.add_node("U", "U", 2, 2, observed=True)
    pgm.add_node("W", "W", 3, 1)

    # Add edges
    pgm.add_edge("H", "W")
    pgm.add_edge("U", "W")

    # Render the PGM
    pgm.render()
    plt.show()
    return daft, pgm


@app.cell(hide_code=True)
def __(np, plt, sns):
    # W = bH + U

    def simulate_weight(H: np.ndarray, beta: float, sigma: float) -> np.ndarray:
        n_heights = len(H)

        # unobserved noise
        U = np.random.normal(loc=0, scale=sigma, size=n_heights)
        return beta * H + U

    n_heights = 100
    min_height = 130
    max_height = 170
    H = np.random.uniform(min_height, max_height, n_heights)

    # Simulate the weight
    W = simulate_weight(H, beta=0.5, sigma=5)

    # Plot
    gen_weight_fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    gen_weight_fig.suptitle("Height and Weight Analysis", y=1.05)

    # height
    plt.sca(axs[0])
    plt.hist(H, bins=25)
    plt.xlabel("height")
    plt.ylabel("frequency")
    plt.title("Simulated Height")

    # Height vs Weight
    plt.sca(axs[1])
    sns.scatterplot(x=H, y=W)
    plt.xlabel("height")
    plt.ylabel("weight")
    plt.title("Simulated W vs H")
    return (
        H,
        W,
        axs,
        gen_weight_fig,
        max_height,
        min_height,
        n_heights,
        simulate_weight,
    )


@app.cell(hide_code=True)
def __(mo):
    mo.md(r"""## Explore Grid Approximation""")
    return


@app.cell(hide_code=True)
def __(mo):
    mo.md(
        r"""
        ### Linear Regression

        Estimate how the average weight changes with a change in height:

        $$E[W_i | H_i] = \alpha + \beta H_i$$

        - $E[W_i | H_i]$: **average weight** conditioned on height
        - $\alpha$: **intercept** of line
        - $\beta$: **slope** of line


        ### Posterior Distribution

        $$
        p(\alpha, \beta, \sigma) = \frac{p(W_i | \alpha, \beta, \sigma) p(\alpha, \beta, \sigma)}{Z}
        $$

        - The only estimator in Bayesian data analysis

        - $p(\alpha, \beta, \sigma)$ -- **Posterior**: Probability of a specific line (model)
        - $p(W_i | \alpha, \beta, \sigma)$ -- **Likelihood**: The number of ways the generative proces (line) could have produced the data
            - aka the "Garden of Forking Data" from Lecture 2
        - $p(\alpha, \beta, \sigma)$ -- **Prior**: the previous Posterior (sometimes with no data)
        - $Z$ -- **normalizing constant**

        Common parameterization

        $$
        \begin{align}
        W_i &\sim \text{Normal}(\mu_i, \sigma) \\
        \mu_i &= \alpha + \beta H_i
        \end{align}
        $$

        _$W$ is distributed normally with mean $\mu$ that is a linear function of $H$_
        """
    )
    return

def test_cell() -> bool:
    return True

if __name__ == "__main__":
    app.run()
