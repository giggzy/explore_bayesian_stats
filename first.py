import marimo

__generated_with = "0.9.27"
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
    fig.suptitle('Common Probability Distributions', fontsize=16)

    # Plot the data
    norm_plt = axes[0,0]
    norm_plt.set_title('Normal')
    sns.histplot(normal, kde=True, bins=10, ax=norm_plt)

    sns.histplot(uniform, kde=True, bins=10, ax=axes[0, 1])
    sns.histplot(exp, kde=True, bins=10, ax=axes[0, 2])
    sns.histplot(poisson, kde=True, bins=10, ax=axes[1, 0])
    sns.histplot(beta, kde=True, bins=10, ax=axes[1, 1])
    sns.histplot(gamma, kde=True, bins=10, ax=axes[1, 2])

    # get a new plot
    return axes, beta, exp, fig, gamma, norm_plt, normal, poisson, uniform


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
    howell = pl.read_csv('data/Howell1.csv', separator=';')


    fig_h, axes_h = plt.subplots(1, 2, figsize=(15, 5))

    #print(howell.columns)
    adults = howell.filter(pl.col("age") > 18)
    children = howell.filter(pl.col("age") <= 18)


    # All age vs height
    sns.scatterplot(data=adults, y="weight", x="height", ax=axes_h[0], color='blue', label='Adults')
    sns.scatterplot(data=children, y="weight", x="height", ax=axes_h[0], color='red', label='Children')
    axes_h[0].set_title("Entire Dataset") 

    # Just Adults
    sns.scatterplot(data=adults, y="weight", x="height", ax=axes_h[1], color='blue', label='Adults')


    axes_h[1].set_title("Adults")

    plt.show()
    howell
    return adults, axes_h, children, fig_h, howell


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
