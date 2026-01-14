import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


def plot_lines_logx(xs, series_dict, title, xlabel, ylabel, out_png, dpi=200):
    """
    series_dict: {label: ys}
    """
    plt.figure()
    for label, ys in series_dict.items():
        plt.plot(xs, ys, marker="o", label=label)

    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
