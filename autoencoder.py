# -*- coding: utf-8 -*-
"""
Created on Tue Dec  29 15:53 2020

Build W encoder and Z decoder matrices.
Then find, and return, Least Squares loss.

NOTE: In python '@' means matrix multiplication.
    IE: x @ y <-> np.matmul(x, y)

NOTE: In python, when working with matrices, '*' is the
    Hadamard product of the 2 matrices.
    IE: x * y <-> np.multiply(x, y)
    | x_11y_11   x_12y_12 |
    | x_21y_21   x_22y_22 |
    | x_31y_31   x_32y_32 |

    Both x and y are (3x2) matrices.

@author: Cory Kromer-Edwards
"""
import plotter

import numpy as np
from numpy.linalg import LinAlgError

# For testing purposes
from keras.datasets import mnist


class AutoEncoder:

    def __init__(self, x, num_features):
        self.x = x


    def _activation(self, z):
        """ReLu activation function"""
        tmp = np.zeros(z.shape)
        return np.maximum(tmp, z)


    def phi(self, w):
        """
        Perform weight and input multiplication
        then run result through ReLu activation function.
        calculates: [varphi(x1;W), varphi(x2;W), ..., varphi(xN;W)]

        :param x:
        :param w:
        :return:
        """

        # Vectorized variable phi
        return self._activation(w.transpose() @ self.x)


    def _inner_psi(self, w):
        """Inner function to calculate inner psi function before norm, and Z matrix."""
        phi_w = self.phi(self.x, w)
        q_1, r_1 = np.linalg.qr(phi_w.transpose())

        if np.linalg.det(r_1) == 0:
            raise LinAlgError("Determinant of R_1 is 0, R_1 does not have an inverse.")

        # Z = XQ_1(R_1^T)^{-1}
        z = x @ q_1 @ np.linalg.inv(r_1.transpose())

        inner_psi = (z @ phi_w) - x + self.alpha * np.square(np.linalg.norm(z, 'fro'))
        return z, inner_psi


    def calc_least_square(x, inner_psi):
        """Calculate least squares error from inner psi function and x."""
        return (1 / x.shape[1]) * np.square(np.linalg.norm(inner_psi, 'fro'))


    def psi(x, w):
        """
        Run 1 iteration to calculate new W and Z from input.

        W.shape = (p,m) = (# of layers [1], # of features)
            p will always be greater than n, so we can have p = n for simplicity to get below
            -> W.shape = (n,m) = (# of inputs/data point, # of features)
        Z.shape = (n,m) = (# of inputs/data point, # of features)
        X.shape = (n,N) = (# of inputs/data point, # of data points)
        """
        z, inner_psi = _inner_psi(x, w)
        least_squares = calc_least_square(x, inner_psi)
        return z, least_squares


    def calc_g(x, w):
        """Calculate the matrix gradient of psi(W)."""
        # Calculate A matrix
        z, inner_psi = _inner_psi(x, w)
        least_squares = calc_least_square(x, inner_psi)
        a = z.transpose() @ inner_psi

        # ================Original, non vectorized, approach kept for reference========================
        #n, m = w.shape
        #n, cap_n = x.shape
        #g = np.zeros(w.shape)
        #for i in range(m):
        #    a_i = a[i, :]
        #    w_i = w[:, i]
        #    sum = np.zeros((n,))
        #    for j in range(cap_n):
        #        x_j = x[:, j]
        #        delta = _activation(w_i.transpose() @ x_j)
        #        delta = (delta > 0).view('i1')  # ReLu derivative
        #        result = (a_i[j] * delta) * x_j
        #        sum = np.add(sum, result)

        #    g[:, i] = sum
        # ==============================================================================================

        # Vectorized implementation of matrix gradient calculation
        # Partial derivative of Relu(W^TX) with respect to W.
        # -> ReLu'(W^TX)X [Using Chain Rule]
        # Formula: X(A*sigma'((W^T)))^T
        delta = phi(x, w)
        delta = (delta > 0).view('i1')  # ReLu derivative
        j = a * delta
        g = x @ j.transpose()

        # For testing old and new matrices. Need to round since there may be rounding errors for trailing
        # decimal places that would make the np.array_equal() method return false when it should be true.
        # print(f"g and g_tmp equal: {np.array_equal(np.around(g, decimals=9), np.around(g_tmp, decimals=9))}")

        return z, least_squares, g


def test_random():
    # Sanity test to make sure that feature number positively impacts least squares error.
    num_points = 100
    num_data_per_point = 55
    x_in = np.random.normal(size=(num_data_per_point, num_points))
    loss_values = []
    for num_features in [1, 5, 10, 15, 20, 40, 70]:
        w_in = np.random.normal(size=(num_data_per_point, num_features))
        try:
            z_out, least_squares_test = psi(x_in, w_in)
            loss_values.append(least_squares_test)
            print(f"(# features : Least squares error = ({num_features} : {least_squares_test})")
        except LinAlgError as e:
            print(f"Error occured with num_features = {num_features}")
            print(e)
            continue

    plotter.plot_loss(loss_values, "Random_Test_with_Features", "Num features from list [1, 5, 10, 15, 20, 40, 70]")


def test_mnist():
    (train_x, _), (_, _) = mnist.load_data()
    plotter.plot_mnist(train_x, "original")                                 # Show original mnist images
    num_img, img_dim, _ = train_x.shape                                     # Get number of images and # pixels per square img
    for num_features in [1, 50, 100, 200, 300, 400, 500, 700]:
        mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))        # Reshape images to match autoencoder input
        w_in = np.random.normal(size=(img_dim * img_dim, num_features))     # Generate random W matrix to test
        z_img, least_squares_img = psi(mnist_in, w_in)                      # Run autoencoder to generate Z
        print(f"MNIST\t(# features : Least squares error = ({num_features} : {least_squares_img:.2E})")
        phi_w_img = phi(mnist_in, w_in)                                     # Calculate phi(W)
        new_mnist = z_img @ phi_w_img                                       # Recreate original images using Z and phi(W)
        new_imgs = np.reshape(new_mnist, train_x.shape)                     # Reshape new images have original shape
        plotter.plot_mnist(new_imgs, f"{num_features}_features")            # Show new images


if __name__ == '__main__':
    np.random.seed(1234)
    test_random()
    test_mnist()

    # If there are any figures in the state machine, show them
    plotter.show_avail_plots()
