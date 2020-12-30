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
from matplotlib import pyplot


def _activation(z):
    """ReLu activation function"""
    tmp = np.zeros(z.shape)
    return np.maximum(tmp, z)


def theta(x, w):
    """
    Perform weight and input multiplication
    then run result through ReLu activation function.

    :param x:
    :param w:
    :return:
    """
    return _activation(w.transpose() @ x)


def sci(x, w):
    """
    Run 1 iteration to calculate new W and Z from input.

    (?) W.shape = (p,m) = (# of layers [1], # of features)
        Should above be -> W.shape = (n,m) = (# of inputs/data point, # of features)
    Z.shape = (n,m) = (# of inputs/data point, # of features)
    X.shape = (n,N) = (# of inputs/data point, # of data points)
    """
    theta_w = theta(x, w)
    q_1, r_1 = np.linalg.qr(theta_w.transpose())

    # Z = XQ_1(R_1^T)^{-1}
    z = x @ q_1 @ np.linalg.inv(r_1.transpose())

    tmp_sci = (z @ theta_w) - x
    least_squares = np.square(np.linalg.norm(tmp_sci, 'fro'))
    return z, least_squares


def show_mnist(imgs):
    # print("Train X MNIST shape: " + str(imgs.shape))
    fig, ax = pyplot.subplots()
    for i in range(9):
        pyplot.subplot(330 + 1 + i)
        pyplot.imshow(imgs[i], cmap=pyplot.get_cmap('gray'))

    pyplot.show()


if __name__ == '__main__':
    # Sanity test to make sure that feature number positively impacts least squares error.
    np.random.seed(1234)
    num_points = 50
    num_data_per_point = 20
    for num_features in [1, 5, 10, 15, 20]:
        x_in = np.random.normal(size=(num_data_per_point, num_points))
        w_in = np.random.normal(size=(num_data_per_point, num_features))
        z_out, least_squares = sci(x_in, w_in)
        print(f"(# features : Least squares error = ({num_features} : {least_squares})")
        # print(z_out)

    # Testing MNIST dataset
    (train_x, _), (_, _) = mnist.load_data()
    show_mnist(train_x)

    for num_features in [1, 50, 100, 200, 300, 400, 500, 700]:
        num_img, img_dim, _ = train_x.shape                                 # Get number of images and # pixels per square img
        mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))        # Reshape images to match autoencoder input
        w_in = np.random.normal(size=(img_dim * img_dim, num_features))     # Generate random W matrix to test
        z_img, least_squares_img = sci(mnist_in, w_in)                      # Run autoencoder to generate Z
        print(f"MNIST\t(# features : Least squares error = ({num_features} : {least_squares_img})")
        theta_w = theta(mnist_in, w_in)                                     # Calculate Theta(W)
        new_mnist = z_img @ theta_w                                         # Generate original images using Z and Theta(W)
        new_imgs = np.reshape(new_mnist, train_x.shape)                     # Reshape new images have original shape
        # show_mnist(new_imgs)                                                # Show new images
