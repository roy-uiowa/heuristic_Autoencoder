# -*- coding: utf-8 -*-
"""
Created on Fri Jan 1 13:34 2021

Gradient decent code.

@author: Cory Kromer-Edwards
"""
import autoencoder
import plotter

import numpy as np
from numpy.linalg import LinAlgError

# For testing purposes
from keras.datasets import mnist


def test_random():
    # Sanity test to make sure that feature number positively impacts least squares error.
    num_points = 100
    num_data_per_point = 55
    learning_rate = 0.5
    x_in = np.random.normal(size=(num_data_per_point, num_points))
    for num_features in [1, 5, 10, 15, 20, 40, 70]:
        w_in = np.random.normal(size=(num_data_per_point, num_features))
        try:
            z_out, least_squares_test = autoencoder.psi(x_in, w_in)
            print(f"(# features : Least squares error = ({num_features} : {least_squares_test})")
            print("Starting gradient decent...")
            loss_values = []  # Keep track of loss values over epochs
            for epoch in range(150):
                z_grd, ls_grd, grd = autoencoder.calc_g(x_in, w_in)  # Calculate Z, Error, and Gradient Matrix
                w_in = w_in - (learning_rate * grd)  # Update W using Gradient Matrix
                loss_values.append(ls_grd)  # Log loss
                print(f"Epoch: {epoch}\t----------\tLoss: {ls_grd}")

            # print(loss_values)
            plotter.plot_loss(loss_values, f"Gradient Loss Over Epochs (test) (num_features: {num_features})")
        except LinAlgError as e:
            print(f"Error occured with num_features = {num_features}")
            print(e)
            continue
            
            
def test_mnist():
    # Gradient check using MNIST
    (train_x, _), (_, _) = mnist.load_data()
    # autoencoder.show_mnist(train_x, "original")                           # Show original mnist images
    num_img, img_dim, _ = train_x.shape  # Get number of images and # pixels per square img
    learning_rate = 0.5
    num_features = 700
    w_in = np.random.normal(size=(img_dim * img_dim, num_features))  # Generate random W matrix to test
    loss_values = []  # Keep track of loss values over epochs
    mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))  # Reshape images to match autoencoder input
    for epoch in range(150):
        z_grd, ls_grd, grd = autoencoder.calc_g(mnist_in, w_in)  # Calculate Z, Error, and Gradient Matrix
        w_in = w_in - (learning_rate * grd)  # Update W using Gradient Matrix
        loss_values.append(ls_grd)  # Log loss
        print(f"Epoch: {epoch}\t----------\tLoss: {ls_grd}")

    phi_w_img = autoencoder.phi(mnist_in, w_in)  # Calculate phi(W)
    new_mnist = z_grd @ phi_w_img  # Recreate original images using Z and phi(W)
    new_imgs = np.reshape(new_mnist, train_x.shape)  # Reshape new images have original shape
    autoencoder.show_mnist(new_imgs, f"{num_features}_features_gradient")  # Show new images

    # print(loss_values)
    plotter.plot_loss(loss_values, "MNIST_Gradient_Loss_Over_Epochs")


def test_gradient():
    num_points = 30                                                    # N
    num_data_per_point = 20                                             # n
    num_features = 12                                                   # m
    const = np.random.normal(size=(num_data_per_point, num_points))     # X
    x = np.random.normal(size=(num_data_per_point, num_features))       # W

    def f(input):
        return autoencoder.psi(const, input)[1]

    def df(input):
        return autoencoder.calc_g(const, input)[2]  # G

    # I have also tried "np.gradient(x) * 1e-3" and "np.random.normal(size=(num_data_per_point, num_features)) * 1e-3"
    dx = x * 1e-3

    # Test 1: Check norm(dx)
    check1 = f(x + dx)
    check2 = f(x - dx)
    check3 = 2*np.tensordot(df(x), dx, axes=2)
    differror = np.linalg.norm(check1 - check2 - check3) / np.linalg.norm(dx)
    print("Test 1 of gradient check: differror should be smaller than, or close to, norm(dx):")
    print(f"differror: {differror}")
    print(f"norm(dx): {np.linalg.norm(dx)}")
    print()

    # Test 2: drop dx by factor of 10 and see if differror drops by 10-100
    new_dx = dx * 0.1
    check1 = f(x + new_dx)
    check2 = f(x - new_dx)
    check3 = 2 * np.tensordot(df(x), new_dx, axes=2)
    new_differror = np.linalg.norm(check1 - check2 - check3) / np.linalg.norm(new_dx)
    print("Test 2 of gradient check: new_differror should be 10-100 factors smaller than differror:")
    print(f"new_differror: {new_differror}")
    print(f"differror: {differror}")

if __name__ == '__main__':
    np.random.seed(1234)
    #test_random()
    test_gradient()
    #test_mnist()
    plotter.show_avail_plots()
