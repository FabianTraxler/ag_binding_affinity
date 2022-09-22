"""Visualization tools to plot training results"""
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.metrics import mean_absolute_error as mae


def plot_correlation(x: np.ndarray, y: np.ndarray, path: str, show_corr: bool = True, show_mae: bool = True ):
    """ Plot a correlation plot and show the regreeion line

    Optional show the pearson correlation with p-value
    Optional show the Mean Absolute Error

    Args:
        x: Predicted values
        y: True values
        path: Path of the results image
        show_corr: Indicator if pearson correlation is to be shown
        show_mae: Indicator if MAE is to be shown

    Returns:
        None
    """
    plot = sns.jointplot(x=x, y=y, kind="reg")

    legend = ""

    if show_corr:
        r, p = stats.pearsonr(x=x, y=y)
        legend += 'r={:f}, p={:f}, '.format(r, p)
    if show_mae:
        error = mae(y, x)
        legend += 'mae={:f}, '.format(error)

    legend = legend[:-2]
    phantom, = plot.ax_joint.plot([], [], linestyle="", alpha=0)
    plot.ax_joint.legend([phantom], [legend])
    plt.savefig(path)
    plt.close()