"""Visualization tools to plot training results"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats



def plot_correlation(x: np.ndarray, y: np.ndarray, path: str, show_corr: bool = True, show_rmse: bool = True ):
    """ Plot a correlation plot and show the regression line

    Optional show the pearson correlation with p-value
    Optional show the Mean Absolute Error

    Args:
        x: Predicted values
        y: True values
        path: Path of the results image
        show_corr: Indicator if pearson correlation is to be shown
        show_rmse: Indicator if RSME is to be shown

    Returns:
        None
    """
    plot = sns.jointplot(x=x, y=y, kind="reg")

    legend = ""

    if show_corr:
        r, p = stats.pearsonr(x=x, y=y)
        legend += 'r={:f}, p={:f}, '.format(r, p)
    if show_rmse:
        error = np.sqrt(np.mean((x-y)**2))
        legend += 'rmse={:f}, '.format(error)

    legend = legend[:-2]
    phantom, = plot.ax_joint.plot([], [], linestyle="", alpha=0)
    plot.ax_joint.legend([phantom], [legend])
    plt.savefig(path)
    plt.close()
