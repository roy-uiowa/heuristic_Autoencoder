# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 08:04 2021

Plotting functions for data.

@author: Cory Kromer-Edwards
"""
import matplotlib.pyplot as plt
import shutil

# Matplotlib figure number to create new figures for each new set of MNIST images
plt_fig_id = 1


def _plots_to_show():
    return plt_fig_id > 1


def show_avail_plots():
    if _plots_to_show():
        plt.show()


def plot_mnist(imgs, title):
    # print("Train X MNIST shape: " + str(imgs.shape))
    global plt_fig_id
    fig, axes = plt.subplots(3, 3)
    fig.suptitle(title)
    fig.canvas.set_window_title(title)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i], cmap=plt.get_cmap('gray'))

    # plt.draw()    # This may or may not make the figures shown quicker when code ends
    plt_fig_id += 1


def plot_loss(array, title, xlabel="Epoch number"):
    global plt_fig_id
    fig, axes = plt.subplots(1, 1)
    ax = axes
    fig.suptitle(title)
    fig.canvas.set_window_title(title)
    ax.plot(array)
    ax.set_ylabel("Loss")
    ax.set_xlabel(xlabel)
    plt_fig_id += 1


def print_progress_bar(iteration, total, decimals=3, fill='#', prefix='Progress:', suffix='Complete',
                       print_end='\r'):
    """
    Variables and progress bar code from:
    https://stackoverflow.com/a/34325723

    Updated to write line max to be width of console - 8% (EX: width = 235 then printline will be 206
    """
    terminal_width = shutil.get_terminal_size().columns
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    length = shutil.get_terminal_size().columns - (len(prefix) + len(suffix) + len(percent) + (terminal_width // 8))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    # Print New Line on Complete
    if iteration == total:
        print()
