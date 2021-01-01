# -*- coding: utf-8 -*-
"""
Created on Tue Dec  29 15:53 2020

Build W encoder and Z decoder matrices.
Then find, and return, Least Squares loss.

NOTE: In python '@' means matrix multiplication.
    IE: x @ y <-> np.matmul(x, y)

@author: Cory Kromer-Edwards
"""

import numpy as np

# For testing purposes
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Matplotlib figure number to create new figures for each new set of MNIST images
plt_fig_id = 1


def _activation(z):
    """ReLu activation function"""
    tmp = np.zeros(z.shape)
    return np.maximum(tmp, z)


def phi(x, w):
    """
    Perform weight and input multiplication
    then run result through ReLu activation function.
    calculates: [varphi(x1;W), varphi(x2;W), ..., varphi(xN;W)]

    :param x:
    :param w:
    :return:
    """

    # Vectorized variable phi
    return _activation(w.transpose() @ x)


def psi(x, w):
    """
    Run 1 iteration to calculate new W and Z from input.

    W.shape = (p,m) = (# of layers [1], # of features)
        p will always be greater than n, so we can have p = n for simplicity to get below
        Should above be -> W.shape = (n,m) = (# of inputs/data point, # of features)
    Z.shape = (n,m) = (# of inputs/data point, # of features)
    X.shape = (n,N) = (# of inputs/data point, # of data points)
    """
    phi_w = phi(x, w)
    q_1, r_1 = np.linalg.qr(phi_w.transpose())

    # Z = XQ_1(R_1^T)^{-1}
    z = x @ q_1 @ np.linalg.inv(r_1.transpose())

    tmp_psi = (z @ phi_w) - x
    least_squares = (1 / x.shape[1]) * np.square(np.linalg.norm(tmp_psi, 'fro'))
    return z, least_squares


def calc_h(x, w):
    """Calculate the partial derivative with respect to W of variable phi function."""
    phi_w = phi(x, w)
    return np.gradient(phi_w, axis=0)   # Computes partial derivative over rows (W)


def calc_g(x, w):
    """Calculate the matrix gradient of psi(W)."""
    g = np.zeros(w.shape)
    n, m = w.shape
    n, cap_n = x.shape

    # Calculate A matrix
    z, least_squares = psi(x, w)
    phi_w = phi(x, w)
    tmp_psi = (z @ phi_w) - x
    a = z.transpose() @ tmp_psi

    #for i in range(m):
    #    a_i = a[i, :]
    #    w_i = w[:, i]
    #    delta = _activation(w_i.transpose() @ x)
    #    delta = (delta > 0).view('i1')              # ReLu derivative
    #    tmp = (a_i @ delta.transpose()) * x         # (a_{i}r'(w_{i}^{T}X)^{T})X -> (n x N) matrix
    #    g[:, i] = np.sum(tmp, axis=1)               # sum(over j, tmp) -> (n x 1) matrix

    for i in range(m):
        a_i = a[i, :]
        w_i = w[:, i]
        sum = np.zeros((n,))
        for j in range(cap_n):
            x_j = x[:, j]
            delta = _activation(w_i.transpose() @ x_j)
            delta = (delta > 0).view('i1')  # ReLu derivative
            result = (a_i[j] * delta) * x_j
            sum = np.add(sum, result)

        g[:, i] = sum

    #a_s = np.asarray([np.sum(a, axis=1)])               # (1 x m)
    #x_s = np.transpose([np.sum(x, axis=1)])             # (n x 1)
    #delta = _activation(w.transpose() @ x)
    #delta = (delta > 0).view('i1')  # ReLu derivative
    #delta_s = np.asarray([np.sum(delta, axis=1)])     # (1 x m)
    #tmp = x @ delta.transpose()
    #g_tmp = np.multiply(tmp, a_s)
    #g_tmp = x_s @ np.multiply(delta_s, a_s)
    #print(f"g_tmp shape: {g.shape}")
    #print(f"g and g_tmp equal: {np.array_equal(g, g_tmp)}")

    return z, least_squares, g


def show_mnist(imgs, title):
    # print("Train X MNIST shape: " + str(imgs.shape))
    global plt_fig_id
    fig, axes = plt.subplots(3, 3)
    fig.suptitle(title)
    fig.canvas.set_window_title(title)
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(imgs[i], cmap=plt.get_cmap('gray'))

    # plt.draw()    # This may or may not make the figures shown quicker when code ends
    plt_fig_id += 1


if __name__ == '__main__':
    # Sanity test to make sure that feature number positively impacts least squares error.
    np.random.seed(1234)
    num_points = 100
    num_data_per_point = 50
    learning_rate = 0.01
    for num_features in [1, 5, 10, 15, 20, 40]:
        x_in = np.random.uniform(size=(num_data_per_point, num_points))
        w_in = np.random.normal(size=(num_data_per_point, num_features))
        z_out, least_squares_test = psi(x_in, w_in)
        print(f"(# features : Least squares error = ({num_features} : {least_squares_test})")
        print("Starting gradient decent...")
        learning_rate = 0.5
        loss_values = []  # Keep track of loss values over epochs
        for epoch in range(150):
            z_grd, ls_grd, grd = calc_g(x_in, w_in)  # Calculate Z, Error, and Gradient Matrix
            w_in = w_in - (learning_rate * grd)  # Update W using Gradient Matrix
            loss_values.append(ls_grd)  # Log loss
            print(f"Epoch: {epoch}\t----------\tLoss: {ls_grd}")

        # print(loss_values)
        plt.plot(loss_values)
        plt.ylabel("Loss")
        plt.xlabel("Epoch number")
        plt.title(f"Gradient Loss Over Epochs (test) (num_features: {num_features})")
        plt.show()
        # print(z_out)

    quit()

    # Testing MNIST dataset
    (train_x, _), (_, _) = mnist.load_data()
    # show_mnist(train_x, "original")                                         # Show original mnist images
    num_img, img_dim, _ = train_x.shape                                     # Get number of images and # pixels per square img
    #for num_features in [1, 50, 100, 200, 300, 400, 500, 700]:
    #    mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))        # Reshape images to match autoencoder input
    #    w_in = np.random.normal(size=(img_dim * img_dim, num_features))     # Generate random W matrix to test
    #    z_img, least_squares_img = psi(mnist_in, w_in)                      # Run autoencoder to generate Z
    #    print(f"MNIST\t(# features : Least squares error = ({num_features} : {least_squares_img:.2E})")
    #    phi_w_img = phi(mnist_in, w_in)                                     # Calculate phi(W)
    #    new_mnist = z_img @ phi_w_img                                       # Recreate original images using Z and phi(W)
    #    new_imgs = np.reshape(new_mnist, train_x.shape)                     # Reshape new images have original shape
    #    # show_mnist(new_imgs, f"{num_features}_features")                    # Show new images

    # If there are any figures in the state machine, show them
    #if plt_fig_id != 1:
    #    plt.show()

    # Gradient check using MNIST
    learning_rate = 0.5
    num_features = 300
    w_in = np.random.normal(size=(img_dim * img_dim, num_features))         # Generate random W matrix to test
    loss_values = []                                                        # Keep track of loss values over epochs
    mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))            # Reshape images to match autoencoder input
    for epoch in range(50):
        z_grd, ls_grd, grd = calc_g(mnist_in, w_in)                         # Calculate Z, Error, and Gradient Matrix
        w_in = w_in - (learning_rate * grd)                                 # Update W using Gradient Matrix
        loss_values.append(ls_grd)                                          # Log loss
        print(f"Epoch: {epoch}\t----------\tLoss: {ls_grd}")

    # print(loss_values)
    plt.plot(loss_values)
    plt.ylabel("Loss")
    plt.xlabel("Epoch number")
    plt.title("Gradient Loss Over Epochs")
    plt.show()
