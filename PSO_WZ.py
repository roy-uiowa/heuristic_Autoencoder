"""
PSO :
 Input: 1 particle = w_in
        w_in = np.random.random() * np.random.normal(size=(num_data_per_point, num_features), scale=5)

 Function to optimize: PSI
        z_img, least_squares_img = ae.psi(w_in)
        z_img is the decoder
        w_in is the encoder matrix
        least_squars_img is the cost to optimize (want low)

        z*phi(w) = output_img => save

"""
import numpy as np
import plotter
from autoencoder import AutoEncoder
from keras.datasets import mnist
import random


class Particle:
    def __init__(self, name, initial_weight, initial_z, cost_function, max_iter):
        self.name = name
        self.velocity_w = []  # particle velocity
        self.pos_best_w = []  # best position individual
        self.err_best_w = -1  # best error individual
        self.cost_w = -1  # error individual
        self.velocity_w = np.multiply(initial_weight, 0)
        self.position_w = initial_weight

        self.velocity_z = []  # particle velocity
        self.pos_best_z = []  # best position individual
        self.err_best_z = -1  # best error individual
        self.cost_z = -1  # error individual
        self.velocity_z = np.multiply(initial_weight, 0)
        self.position_z = initial_z

        self.cost_function = cost_function
        self.w = sorted([.3 + random.random() for i in range(max_iter)])[::-1]
        self.c1 = sorted([random.randrange(1, 5) for i in range(max_iter)])[::-1]
        self.c2 = sorted([random.randrange(1, 5) for i in range(max_iter)])

    # evaluate current fitness
    def evaluate(self):
        self.cost_w = self.cost_function(self.position_w, self.position_z)
        self.cost_z = self.cost_w
        # check to see if the current position is an individual best
        if self.err_best_w == -1 or self.cost_w < self.err_best_w:
            self.pos_best_w = self.position_w
            self.err_best_w = self.cost_w

        if self.err_best_z == -1 or self.cost_z < self.err_best_z:
            self.pos_best_z = self.position_z
            self.err_best_z = self.cost_z

    # update new particle velocity
    def update_velocity(self, pos_best_g_w,pos_best_g_z, iteration):
        vel_cog_w = self.c1[iteration] * np.random.random()
        vel_cog_w = np.multiply(vel_cog_w, np.subtract(self.pos_best_w, self.position_w))
        vel_soc_w = self.c2[iteration] * np.random.random()
        vel_soc_w = np.multiply(vel_soc_w, np.subtract(pos_best_g_w, self.position_w))

        vel_pre_w = self.w[iteration] * self.velocity_w
        self.velocity_w = vel_pre_w + vel_soc_w + vel_cog_w

        vel_cog_z = self.c1[iteration] * np.random.random()
        vel_cog_z = np.multiply(vel_cog_z, np.subtract(self.pos_best_z, self.position_z))
        vel_soc_z = self.c2[iteration] * np.random.random()
        vel_soc_z = np.multiply(vel_soc_z, np.subtract(pos_best_g_z, self.position_z))

        vel_pre_z = self.w[iteration] * self.velocity_z
        self.velocity_z = vel_pre_z + vel_soc_z + vel_cog_z

    def update_position(self):
        self.position_w = np.add(self.position_w, self.velocity_w)
        self.position_z = np.add(self.position_z, self.velocity_z)

class Algorithm():
    def __init__(self, x_in, history, maxiter, num_particles, num_features, num_data_per_point, shape):
        self.err_best_g_w = None  # best error for group
        self.pos_best_g_w = []  # best position for group
        self.err_best_g_z = None  # best error for group
        self.pos_best_g_z = []  # best position for group
        self.maxiter = maxiter
        self.shape = shape
        # establish the swarm
        self.ae = AutoEncoder(x_in, num_features, random_seed=1234, use_gpu=False)
        self.history = history
        self.updates = []
        self.num_features = num_features
        self.num_particles = num_particles
        self.swarm = []
        for i in range(0, num_particles):
            w_in = np.random.normal(size=(num_data_per_point, num_features), scale=5)
            z_in = np.random.normal(size=(num_data_per_point, num_features), scale=5)
            self.swarm.append(Particle(i, w_in, z_in,  self.ae.psi, maxiter))

    def run(self):
        # begin optimization loop
        loss_values = []
        for i in range(self.maxiter):
            print("----------\n\t" + str(i))
            # cycle through particles in swarm and evaluate fitness
            min_loss = 9e12
            smallest_particle_cost = None
            for particle in self.swarm:
                particle.evaluate()
                print("p_z{} == cost: {}".format(particle.name, int(particle.cost_z)))
                # determine if current particle is the best (globally)
                if min_loss > particle.cost_w:
                    min_loss = particle.cost_w

                if self.err_best_g_w is None or particle.cost_w < self.err_best_g_w:
                    self.pos_best_g_w = particle.position_w
                    self.err_best_g_w = particle.cost_w
                    print("new best weight")

                if self.err_best_g_z is None or particle.cost_z < self.err_best_g_z:
                    self.pos_best_g_z = particle.position_z
                    self.err_best_g_z = particle.cost_z
                    print("new best z")

            for particle in self.swarm:
                particle.update_velocity(self.pos_best_g_w, self.pos_best_g_z, i)
                particle.update_position()

        return self.ae, self.pos_best_g_w, self.pos_best_g_z


def test_mnist():
    print("===== RUNNING MNIST =====")
    (train_x, _), (_, _) = mnist.load_data()
    train_x = train_x / 255
    plotter.plot_mnist(train_x, "original")  # Show original mnist images
    num_img, img_dim, _ = train_x.shape  # Get number of images and # pixels per square img
    mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))  # Reshape images to match autoencoder input

    history = 20
    maxiter = 4
    num_particles = 5
    for num_features in [10]:
        PSO = Algorithm(mnist_in, history, maxiter, num_particles, num_features, img_dim * img_dim, train_x.shape)
        ae, w_in, z_in = PSO.run()
        z_img, least_squares_img = AutoEncoder.psi(w_in)  # Run autoencoder to generate Z
        phi_w_img = AutoEncoder.phi(w_in)  # Calculate phi(W)
        new_mnist = z_img @ phi_w_img  # Recreate original images using Z and phi(W)
        new_imgs = np.reshape(new_mnist, train_x.shape)  # Reshape new images have original shape
        plotter.plot_mnist(new_imgs, f"{num_features}_features")  # Show new images


if __name__ == '__main__':
    np.random.seed(1234)
    test_mnist()
