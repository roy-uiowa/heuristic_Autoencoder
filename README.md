# Heuristic Autoencoder
Using Least Squares calculations to perform heuristic search for an encoder matrix and decoder matrix.

# Library Setup
The following libraries, with given pip commands, must be installed before running
* ``pip install numpy==1.19.3``
* ``pip install keras``
* ``pip install tensorflow``
* ``pip install matplotlib``

A note about the libraries above. Installing them in order matters. Currently, on Windows at least,
the latest numpy version cannot work. The version given is the newest version that works. This must be 
installed first before keras or tensorflow as they may install newer version.

# Test Output
There is an output folder called ``mnist_tests``. That folder contains test results for the autoencoder 
that ran through with the given number of features to generate each picture. The code to generate the pictures
is currently commented out in the main method, but they can be generated if line 94 is 
uncommented (which has: ``show_mnist(new_imgs)``).

# Running Autoencoder
The following is how to run the Autoencoder
```python
x = # generate or load with shape : (num_data_per_point, num_points)
w = # Randomly generate or pass in from previous run with shape: (num_data_per_point, num_features))
z, least_squares = sci(x, w)
```

If you then want to test how well the autoencoder can recreate the input, run the lines below:
```python
theta_w = theta(x, w)
recreated_x = z @ theta_w
```