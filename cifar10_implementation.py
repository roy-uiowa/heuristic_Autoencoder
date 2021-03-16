# -*- coding: utf-8 -*-
"""
Created on Fri Jan 1 13:34 2021

Gradient decent code.

@author: Cory Kromer-Edwards
@modification: Tarun Roy
"""
from autoencoder import AutoEncoder
import plotter

import numpy as np

# baseline model with dropout and data augmentation on the cifar10 dataset
import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import BatchNormalization

# For testing purposes
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.models import load_model
import time
import matplotlib.pyplot as plt

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = cifar10.load_data()
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
  # convert from integers to floats
  train_norm = train.astype('float32')
  test_norm = test.astype('float32')
  # normalize to range 0-1
  train_norm = train_norm / 255.0
  test_norm = test_norm / 255.0
  # convert into gray_scale
  train_gray = np.dot(train_norm[...,:3], [0.299, 0.587, 0.114])
  test_gray = np.dot(test_norm[...,:3], [0.299, 0.587, 0.114])
  # return normalized images
  return train_gray, test_gray


def test_random():
    # Sanity test to make sure that feature number positively impacts least squares error.
    num_points = 100
    num_data_per_point = 55
    learning_rate = 0.5
    x_in = np.random.normal(size=(num_data_per_point, num_points))
    for num_features in [1, 5, 10, 15, 20, 40, 70]:
        ae = AutoEncoder(x_in, num_features, random_seed=1234)
        w_in = np.random.normal(size=(num_data_per_point, num_features))
        z_out, least_squares_test = ae.psi(w_in)
        print(f"(# features : Least squares error = ({num_features} : {least_squares_test})")
        print("Starting gradient decent...")
        loss_values = []  # Keep track of loss values over epochs
        for epoch in range(1000):
            z_grd, ls_grd, grd = ae.calc_g(w_in)  # Calculate Z, Error, and Gradient Matrix
            w_in = w_in - (learning_rate * grd)  # Update W using Gradient Matrix
            loss_values.append(ls_grd)  # Log loss
            print(f"Epoch: {epoch}\t----------\tLoss: {ls_grd}")

        # print(loss_values)
        plotter.plot_loss(loss_values, f"Gradient Loss Over Epochs (test) (num_features: {num_features})")


def do_epoch(ae, w, learning_rate, loss_values, times, loss_values_less, loss_diffs, epoch, start_time):
    epoch_start_time = time.time()
    z_grd, ls_grd, grd = ae.calc_g(w)  # Calculate Z, Error, and Gradient Matrix
    w_in = w - ((1 / learning_rate) * grd)  # Update W using Gradient Matrix
    if len(loss_values) >= 1:
        loss_diffs.append(abs(loss_values[-1] - ls_grd))
    else:
        loss_diffs.append(1000000000)

    loss_values.append(ls_grd)  # Log loss
    epoch_end_time = time.time()
    times.append(epoch_end_time - epoch_start_time)
    print(f"Epoch: {epoch}\t----------\tElapsed time(sec): {(epoch_end_time - epoch_start_time):.2f}\
\t----------\tLoss: {ls_grd:.2f}\
\t----------\tLoss diff: {loss_diffs[-1]:.2f}\
\t----------\tAverage time/epoch(sec): {sum(times) / len(times):.2f}\
\t----------\tRun time(sec): {(epoch_start_time - start_time):.2f}")
    # plotter.plot_loss(loss_values, f"Epoch: {epoch}")
    if epoch > 0:
        loss_values_less.append(ls_grd)

    return w_in, z_grd


def test_gradient():
    num_points = 30  # N
    num_data_per_point = 20  # n
    num_features = 12  # m
    const = np.random.normal(size=(num_data_per_point, num_points))  # X
    x = np.random.normal(size=(num_data_per_point, num_features))  # W
    dx = x * 1e-3
    ae = AutoEncoder(const, num_features, random_seed=1234)

    def f(input):
        return ae.psi(input)[1]

    def df(input):
        return ae.calc_g(input)[2]  # G

    # Test 1: Check norm(dx)
    check1 = f(x + dx)
    check2 = f(x - dx)
    check3 = 2 * np.tensordot(df(x), dx, axes=2)
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


def rgb2gray(rgb):
    """Convert from color image (RGB) to grayscale.
       Source: opencv.org
       grayscale = 0.299*red + 0.587*green + 0.114*blue
    Argument:
        rgb (tensor): rgb image
    Return:
        (tensor): grayscale image
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def test_cifar10(num_epochs=None):
    # (train_x, _), (_, _) = cifar10.load_data()
    (_, _), (train_x, _) = cifar10.load_data()
    print(train_x.shape)
    plotter.plot_mnist(train_x, "original")
    train_x = rgb2gray(train_x)
    train_x = train_x / 255
    plotter.plot_mnist(train_x, "grayscale")
    num_img, img_h, img_w = train_x.shape
    print(train_x.shape)
    learning_rate = 0.5
    num_features = 768;
    loss_values = []
    loss_values_less = []
    loss_diffs = []

    w_in = np.random.normal(size=(img_h * img_w, num_features))
    cifar_in = np.reshape(train_x, (img_h * img_w, num_img))
    # cifar_in = np.reshape(train_x, (img_h, img_w, num_img*img_ch))
    print(cifar_in.shape)

    ae = AutoEncoder(cifar_in, num_features, random_seed=1234, use_gpu=True)
    start_time = time.time()
    times = []
    if num_epochs:
        for epoch in range(num_epochs):
            w_in, z_grd = do_epoch(ae, w_in, learning_rate, loss_values, times, loss_values_less, loss_diffs, epoch,
                                   start_time)
    else:
        epoch_history_check = 5
        epoch = 0
        loss_avg = 1000
        tol = 0.03
        while loss_avg > tol:
            w_in, z_grd = do_epoch(ae, w_in, learning_rate, loss_values, times, loss_values_less, loss_diffs, epoch,
                                   start_time)
            loss_check = loss_diffs[-epoch_history_check:]
            loss_avg = sum(loss_check) / len(loss_check)
            epoch += 1

    print(f"Total time to run gradient decent (sec): {time.time() - start_time}")
    phi_w_img = ae.phi(w_in)  # Calculate phi(W)
    new_cifar = z_grd @ phi_w_img  # Recreate original images using Z and phi(W)
    print(new_cifar.shape)
    new_imgs = np.reshape(new_cifar, train_x.shape)  # Reshape new images have original shape
    plotter.plot_mnist(new_imgs, f"{num_features}_features_gradient")  # Show new images

    # print(loss_values)
    plotter.plot_loss(loss_values, "CIFAR10_Gradient_Loss_Over_Epochs")
    plotter.plot_loss(loss_values_less, "CIFAR10_Gradient_Loss_Over_Epochs_all_epochs_except_zero")
    # return train_x
    return new_imgs


if __name__ == '__main__':
    np.random.seed(1234)
    #test_random()
    #test_gradient()
    # test_mnist(1000)
    newCIFAR10_data = test_cifar10(300)  #currently it uses testset for compression
    print(newCIFAR10_data.shape)
    plotter.show_avail_plots()

    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    trainX, testX = prep_pixels(trainX, testX)
    testX = testX.reshape(10000,32,32,1)
    print(testX.shape)

    #download h5 file from this link: https://drive.google.com/file/d/1KnWbp2YDdZGeDEaV-4fWWGtkPYwEqB4I/view?usp=sharing
    new_model = load_model('cifar10vgg.h5')

    # evaluate model
    _, acc = new_model.evaluate(newCIFAR10_data, testY, verbose=1) # calculate model accuracy on test set
    print('> %.3f' % (acc * 100.0))