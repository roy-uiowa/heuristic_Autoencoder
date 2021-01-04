# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 08:04 2021

Plotting functions for data.

@author: Cory Kromer-Edwards
"""
import matplotlib.pyplot as plt

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
