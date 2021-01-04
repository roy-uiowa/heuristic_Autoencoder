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

Shapes of matrices:
W.shape = (p,m) = (# of layers [1], # of features)
    p will always be greater than n, so we can have p = n for simplicity to get below
    -> W.shape = (n,m) = (# of inputs/data point, # of features)
Z.shape = (n,m) = (# of inputs/data point, # of features)
X.shape = (n,N) = (# of inputs/data point, # of data points)

@author: Cory Kromer-Edwards
"""
import plotter

import numpy as np
from scipy.linalg import svdvals

cupy_loaded = True
try:
    import cupy as cp
except ImportError or ModuleNotFoundError as e:
    print(f"Cupy failed to be imported with the following exception. Will fallback to numpy...")
    print(e)
    cupy_loaded = False

# For testing purposes
from keras.datasets import mnist


class AutoEncoder:

    def __init__(self, x, num_features, use_gpu=True, random_seed=None):
        """

        :param x: Numpy or Cupy ndarray of input in shape (n x N)
        :param num_features: Number of features to encode down to (m)
        :param use_gpu: (optional default=True) Whether to use GPU regardless of if Cupy is installed and X is large enough
        :param random_seed: (optional default=None) A random seed that will be set for testing if given
        """
        # Check if the size of X is over 10 million elements.
        # If it is, set the math library to Cupy to get speedup on GPU.
        # Otherwise, speedup would not be worth it, so set to Numpy.
        n, cap_n = x.shape
        if n * cap_n > 1e7 and cupy_loaded and use_gpu:
            self.ml = cp
            self.ml.synchronize = lambda: self.ml.cuda.Stream.null.synchronize()
        else:
            self.ml = np
            self.ml.synchronize = lambda: True

        if random_seed:
            self.ml.random.seed(random_seed)

        self.x = self.ml.asarray(x)
        tmp_w = self.ml.random.normal(size=(x.shape[0], num_features))

        # Full matrices make u = (M, M) and vh = (N, N) if true. This leads to memory limit error.
        # Setting to false has u = (M, K) and vh = (K, N). We do not care about u and vk
        # so we want this. We can also set compute_uv=False to save time on computing u and vh altogether.
        # This leaves to the output to be s instead of u, s, vh.
        phi_w = self.phi(tmp_w, output_numpy=False)
        try:
            phi_w = phi_w.get()
        except:
            pass

        s = np.linalg.svd(phi_w, full_matrices=False, compute_uv=False)
        self.alpha = 0.01 * self.ml.square(s.max())
        self.ml.synchronize()

    def _activation(self, z):
        """ReLu activation function"""
        tmp = self.ml.zeros(z.shape)
        return self.ml.maximum(tmp, z)

    def phi(self, w, output_numpy=True):
        """
        Perform weight and input multiplication
        then run result through ReLu activation function.
        calculates: [varphi(x1;W), varphi(x2;W), ..., varphi(xN;W)]

        :param w: Either Cupy or Numpy ndarray. It is the encoder matrix of size (p x m) [NOTE: p = n for simplicity]
        :param output_numpy: (optional default=True) Whether to output tmp as Numpy ndarray (True) or Cupy ndarray (False)
        :return: Phi(W) = sigma(W^T @ X)
        """

        # Vectorized variable phi
        tmp_w = self.ml.asarray(w)
        tmp = self._activation(self.ml.matmul(tmp_w.transpose(), self.x))
        self.ml.synchronize()
        if output_numpy:
            try:
                return tmp.get()
            except:
                return tmp
        else:
            return tmp

    def _inner_psi(self, w):
        """Inner function to calculate inner psi function before norm, and Z matrix."""
        phi_w = self.phi(w, output_numpy=False)
        fact = self.ml.concatenate((phi_w, self.ml.sqrt(self.alpha) * self.ml.identity(w.shape[1])), axis=1)
        q_1, r_1 = np.linalg.qr(fact.transpose())

        # Z = XQ_1(R_1^T)^{-1}
        q_1 = q_1[:-abs(self.x.shape[1] - q_1.shape[0]), :]
        z = self.x @ q_1 @ self.ml.linalg.inv(r_1.transpose())
        inner_psi = (z @ phi_w) - self.x

        return z, inner_psi

    def _calc_least_square(self, inner_psi, z):
        """Calculate least squares error from inner psi function and x."""
        tmp = (1 / self.x.shape[1]) * (self.ml.square(self.ml.linalg.norm(inner_psi, 'fro')) + self.alpha * self.ml.square(self.ml.linalg.norm(z, 'fro')))
        return tmp.item()

    def psi(self, w):
        """
        Run 1 iteration to calculate new W and Z from input.
        :param w: Either Cupy or Numpy ndarray. It is the encoder matrix of size (p x m) [NOTE: p = n for simplicity]
        :return: Psi(W) = ||Z^T @ W - X||_F^2 or least squares error between Z^T @ W and X.
        """
        z, inner_psi = self._inner_psi(w)
        least_squares = self._calc_least_square(inner_psi, z)
        self.ml.synchronize()
        try:
            return z.get(), least_squares
        except:
            return z, least_squares

    def calc_g(self, w):
        """
        Calculate the matrix gradient of psi(W).
        :param w: Either Cupy or Numpy ndarray. It is the encoder matrix of size (p x m) [NOTE: p = n for simplicity]
        :return: The gradient matrix G for Psi(W)
        """
        # Calculate A matrix
        z, inner_psi = self._inner_psi(w)
        least_squares = self._calc_least_square(inner_psi, z)
        a = self.ml.matmul(z.transpose(), inner_psi)

        # ================Original, non vectorized, approach kept for reference========================
        #n, m = w.shape
        #n, cap_n = self.x.shape
        #g = np.zeros(w.shape)
        #for i in range(m):
        #    a_i = a[i, :]
        #    w_i = w[:, i]
        #    sum = np.zeros((n,))
        #    for j in range(cap_n):
        #        x_j = self.x[:, j]
        #        delta = self._activation(w_i.transpose() @ x_j)
        #        delta = (delta > 0).view('i1')  # ReLu derivative
        #        result = (a_i[j] * delta) * x_j
        #        sum = np.add(sum, result)

        #    g[:, i] = sum
        # ==============================================================================================

        # Vectorized implementation of matrix gradient calculation
        # Partial derivative of Relu(W^TX) with respect to W.
        # -> ReLu'(W^TX)X [Using Chain Rule]
        # Formula: X(A*sigma'((W^T)))^T
        delta = self.phi(w, output_numpy=False)
        delta = (delta > 0).view('i1')  # ReLu derivative
        j = a * delta
        g = self.ml.matmul(self.x, j.transpose())
        self.ml.synchronize()

        # For testing old and new matrices. Need to round since there may be rounding errors for trailing
        # decimal places that would make the np.array_equal() method return false when it should be true.
        # print(f"g and g_tmp equal: {np.array_equal(np.around(g, decimals=9), np.around(g_tmp, decimals=9))}")

        try:
            return z.get(), least_squares, g.get()
        except:
            return z, least_squares, g


def test_random():
    # Sanity test to make sure that feature number positively impacts least squares error.
    num_points = 100
    num_data_per_point = 55
    x_in = np.random.normal(size=(num_data_per_point, num_points))
    loss_values = []
    for num_features in [1, 5, 10, 15, 20, 40, 70]:
        ae = AutoEncoder(x_in, num_features, random_seed=1234)
        w_in = np.random.normal(size=(num_data_per_point, num_features))
        z_out, least_squares_test = ae.psi(w_in)
        loss_values.append(least_squares_test)
        print(f"(# features : Least squares error = ({num_features} : {least_squares_test})")

    plotter.plot_loss(loss_values, "Random_Test_with_Features", "Num features from list [1, 5, 10, 15, 20, 40, 70]")


def test_mnist():
    (train_x, _), (_, _) = mnist.load_data()
    plotter.plot_mnist(train_x, "original")                                 # Show original mnist images
    num_img, img_dim, _ = train_x.shape                                     # Get number of images and # pixels per square img
    mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))            # Reshape images to match autoencoder input
    for num_features in [1, 50, 100, 200, 300, 400, 500, 700]:
        ae = AutoEncoder(mnist_in, num_features, random_seed=1234)
        w_in = np.random.normal(size=(img_dim * img_dim, num_features))     # Generate random W matrix to test
        z_img, least_squares_img = ae.psi(w_in)                             # Run autoencoder to generate Z
        print(f"MNIST\t(# features : Least squares error = ({num_features} : {least_squares_img:.2E})")
        phi_w_img = ae.phi(w_in)                                            # Calculate phi(W)
        new_mnist = z_img @ phi_w_img                                       # Recreate original images using Z and phi(W)
        new_imgs = np.reshape(new_mnist, train_x.shape)                     # Reshape new images have original shape
        plotter.plot_mnist(new_imgs, f"{num_features}_features")            # Show new images


if __name__ == '__main__':
    np.random.seed(1234)
    test_random()
    test_mnist()

    # If there are any figures in the state machine, show them
    plotter.show_avail_plots()
