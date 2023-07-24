"""Visualization tools to plot training results"""
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats



def plot_correlation(x: np.ndarray, y: np.ndarray, path: Optional[str] = None, show_corr: bool = True, show_rmse: bool = True ):
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

    #min_affinity = np.min(np.concatenate([x, y]))
    #max_affinity = np.max(np.concatenate([x, y]))
    #plt.xlim([min_affinity, max_affinity])
    #plt.ylim([min_affinity, max_affinity])

    legend = legend[:-2]
    phantom, = plot.ax_joint.plot([], [], linestyle="", alpha=0)
    plot.ax_joint.legend([phantom], [legend])

    plot.ax_joint.set_xlabel("Label [-log(Kd)]")
    plot.ax_joint.set_ylabel("Prediction [-log(Kd)]")
    plt.tight_layout()
    if path:
        plt.savefig(path)
        plt.close()
    else:
        plt.show()
