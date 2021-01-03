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

## Cupy and GPU programming
If you wish to have all computations run on Nvidia GPU then you will need to import [Cupy](https://cupy.dev/) which is a Cuda 
wrapper for Numpy. Having this library to run the Autoencoder is **NOT** required. If Cupy is not installed
then the library will fallback on Numpy automatically. 

### Setting up environment for Cupy
The tested version of Cupy that is verified to work is with Cuda version 10.2.
Visual Studio 2017 should be installed before setting up Cuda 10.2. After Visual Studio 2017 is installed and
Cuda 10.2 is installed, you may need to install the cuDNN that corresponds with Cuda 10.2.
If you want to compile Cupy from source or add a new kernel from source then you will need cuDNN
In all, to set up the environment, you will need to follow the following steps:
1. Install Visual Studio 2017 from [here](https://visualstudio.microsoft.com/vs/older-downloads/)
2. Install Cuda 10.2 from [here](https://developer.nvidia.com/cuda-10.2-download-archive)
3. Install corresponding cuDNN version for Cuda 10.2 from [here](https://developer.nvidia.com/CUDNN) (only if you want to do further development)

Once this is all done, you will be able to install Cupy and run it.

### Running Autoencoder with Cupy
You can still make X through Numpy and do all computation outside the Autoencoder through Numpy if desired. When X is passed into
the initialization of the AutoEncoder() class, it is converted to a Cupy ndarray and put on the GPU regardless of if it was already on 
there or not. The same happens for W when it is passed into each method call. Then, all variables returned are automatically changed to 
Numpy ndarray's and brought to the CPU before being returned. In other words, the example given below is the same in all following
scenarios:
* Cupy not installed
* Cupy installed and Cupy.ndarray passed in for X and W
* Cupy installed and Numpy.ndarray passed in for X and W

In all cases, a Numpy.ndarray is returned.

### Installing Cupy
Installing a precompiled wheel for Python is the easiest option for Cupy. Fortunately, Cupy has these wheels available on
Pypi to install with pip. You need to install the right Cupy version for the given Cuda version installed. In this case,
the Autoencoder runs on version 10.2. The pip install command for Cupy is then:

``pip install cupy-cuda102``

If you have a different version of Cuda installed, say for example Cuda version X.Y, then you would use

``pip install cupy-cudaXY``

For more information about a version, you can visit [https://pypi.org/project/cupy-cuda102/](https://pypi.org/project/cupy-cuda102/),
or you can change the URL for your version by pypi.org/project/cupy-cudaXY/ as you did above with pip.

# Test Output
There is an output folder called ``mnist_tests``. That folder contains test results for the autoencoder 
that ran through with the given number of features to generate each picture.

# Running Autoencoder
The following is how to run the Autoencoder. When you start the program, you may get errors or warnings
from Tensorflow. This is ok as we are only using Tensorflow to load MNIST dataset. All 
computation is done through numpy.
```python
from autoencoder import AutoEncoder
x = # generate or load with shape : (num_data_per_point, num_points)
w = # Randomly generate with shape: (num_data_per_point, num_features))
ae = AutoEncoder(x, num_features)
z, least_squares = ae.psi(w)
# Update W using least squares loss as heuristic
z, least_squares = ae.psi(w)    # Pass updated W in and run again...
```

If you then want to test how well the autoencoder can recreate the input, run the lines below:
```python
phi_w = ae.phi(w)
recreated_x = z @ phi_w
```