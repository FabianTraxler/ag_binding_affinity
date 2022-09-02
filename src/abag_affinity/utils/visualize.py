import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np


def plot_correlation(x: np.ndarray, y: np.ndarray, path: str ):
    plot = sns.jointplot(x=x, y=y, kind="reg")

    r, p = stats.pearsonr(x=x, y=y)

    phantom, = plot.ax_joint.plot([], [], linestyle="", alpha=0)
    plot.ax_joint.legend([phantom], ['r={:f}, p={:f}'.format(r, p)])
    plt.savefig(path)
    plt.close()