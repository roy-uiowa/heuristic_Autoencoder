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
    def __init__(self, name, initial_weight, cost_function, max_iter):
        self.name = name
        self.velocity = []  # particle velocity
        self.pos_best = []  # best position individual
        self.err_best = -1  # best error individual
        self.cost = -1  # error individual
        self.velocity = np.multiply(initial_weight, 0)
        self.position = initial_weight
        self.cost_function = cost_function
        self.w = sorted([.3 + random.random() for i in range(max_iter)])[::-1]
        self.c1 = sorted([1 for i in range(max_iter)])[::-1]
        self.c2 = sorted([1 for i in range(max_iter)])

    # evaluate current fitness
    def evaluate(self):
        _, self.cost = self.cost_function(self.position)
        # check to see if the current position is an individual best
        if self.err_best == -1 or self.cost < self.err_best:
            self.pos_best = self.position
            self.err_best = self.cost

    # update new particle velocity
    def update_velocity(self, pos_best_g, iteration):
        vel_cog = self.c1[iteration] * np.random.random()
        vel_cog = np.multiply(vel_cog, np.subtract(self.pos_best, self.position))

        vel_soc = self.c2[iteration] * np.random.random()
        vel_soc = np.multiply(vel_soc, np.subtract(pos_best_g, self.position))

        vel_pre = self.w[iteration] * self.velocity
        self.velocity = vel_pre + vel_soc + vel_cog
        # update the particle position based off new velocity updates

    def update_position(self):
        self.position = np.add(self.position, self.velocity)


class Algorithm():
    def __init__(self, x_in, history, maxiter, num_particles, num_features, num_data_per_point, shape):
        self.err_best_g = None  # best error for group
        self.pos_best_g = []  # best position for group
        self.num_particles = 3
        self.maxiter = maxiter
        self.shape = shape
        # establish the swarm
        self.ae = AutoEncoder(x_in, num_features, random_seed=1234, use_gpu=True)
        self.history = history
        self.updates = []
        self.num_features = num_features
        self.num_particles = num_particles
        self.swarm = []
        for i in range(0, num_particles):
            w_in = np.random.normal(size=(num_data_per_point, num_features), scale=5)
            self.swarm.append(Particle(i, w_in, self.ae.psi_w, maxiter))

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
                print("p{} == cost: {}".format(particle.name, int(particle.cost)))
                # determine if current particle is the best (globally)
                if min_loss > particle.cost:
                    min_loss = particle.cost

                if self.err_best_g is None or particle.cost < self.err_best_g:
                    self.pos_best_g = particle.position
                    self.err_best_g = particle.cost
                    print("new best")

                if smallest_particle_cost is None or particle.cost > smallest_particle_cost:
                    smallest_particle_cost = particle.cost

            loss_values.append(min_loss)
            dis = 0
            for p1 in self.swarm:
                for p2 in self.swarm:
                    dis += np.linalg.norm(p1.position - p2.position)
                break

            print("ave. distance between matrix: {}".format(dis))
            for particle in self.swarm:
                particle.update_velocity(self.pos_best_g, i)
                particle.update_position()

            if i % self.history == 0:
                z_grd, ls_grd, grd = self.ae.calc_g(self.pos_best_g)  # Calculate Z, Error, and Gradient Matrix
                phi_w_img = self.ae.phi(self.pos_best_g)  # Calculate phi(W)
                new_mnist = z_grd @ phi_w_img  # Recreate original images using Z and phi(W)
                new_imgs = np.reshape(new_mnist, self.shape)  # Reshape new images have original shape
                plotter.plot_mnist(new_imgs, f"d{self.num_features}_features_{i}_iteration")  # Show new images

            if int(self.num_particles / self.maxiter) % (i + 1) == 0 and len(self.swarm) > 3:
                index = 0
                for particle_loc in range(len(self.swarm)):
                    if self.swarm[particle_loc].cost == smallest_particle_cost:
                        index = particle_loc
                        break

               # del self.swarm[index]
                print("removed worse")
        return self.ae, self.pos_best_g, self.err_best_g, self.updates, loss_values


def test_mnist():
    print("===== RUNNING MNIST =====")
    (train_x, _), (_, _) = mnist.load_data()
    train_x = train_x / 255
    plotter.plot_mnist(train_x, "original")  # Show original mnist images
    num_img, img_dim, _ = train_x.shape  # Get number of images and # pixels per square img
    mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))  # Reshape images to match autoencoder input

    history = 2
    maxiter = 100
    num_particles = 50
    for num_features in [200]:
        PSO = Algorithm(mnist_in, history, maxiter, num_particles, num_features, img_dim * img_dim, train_x.shape)
        ae, w_in, least_squares_test, updated_history, loss_values = PSO.run()
        plotter.plot_loss(loss_values, f"{num_features}_features_pso_w")
        print(f"(# features : Least squares error = ({num_features} : {least_squares_test})")


if __name__ == '__main__':
    np.random.seed(1234)
    test_mnist()
