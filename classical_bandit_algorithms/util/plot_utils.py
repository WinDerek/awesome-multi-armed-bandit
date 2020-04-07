import numpy as np
import matplotlib.pyplot as plt


def plot_vertical_bar_chart(labels,
                            values,
                            title,
                            xlabel,
                            ylabel,
                            rotation,
                            horizontal_line_value,
                            bar_color='#9b59b6',
                            horizontal_line_color='#f22613',
                            save_figure_name=None):
    index = np.arange(len(labels))
    figure, ax = plt.subplots()
    ax.bar(index, values, color=bar_color)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(index)
    ax.set_xticklabels(labels, fontsize=10, rotation=rotation)
    ax.set_title(title)

    rects = ax.patches
    for rect, value in zip(rects, values):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            height,
            round(value, 4),
            ha='center',
            va='bottom')

    # Plot the horizontal line if necessary
    if horizontal_line_value is not None:
        plt.axhline(horizontal_line_value, color=horizontal_line_color)

    # Configure the figure
    figure = plt.gcf()
    figure.set_size_inches((16, 8))
    figure.set_dpi(200)

    if save_figure_name is not None:
        figure.savefig(save_figure_name)
    else:
        plt.show()

    return
