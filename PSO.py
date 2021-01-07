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
        self.history = [initial_weight]

    # evaluate current fitness
    def evaluate(self):
        _, self.cost = self.cost_function(self.position)
        # check to see if the current position is an individual best
        if self.err_best == -1 or self.cost < self.err_best:
            self.pos_best = self.position
            self.err_best = self.cost

    # update new particle velocity
    def update_velocity(self, pos_best_g, iteration):
        w =.5
        c1=1
        c2=2

        vel_cog = c1 * np.random.random()
        vel_cog = np.multiply(vel_cog, np.subtract(self.pos_best, self.position))

        vel_soc = c2 * np.random.random()
        vel_soc = np.multiply(vel_soc, np.subtract(pos_best_g, self.position))

        vel_pre = w * self.velocity
        self.velocity = vel_pre + vel_soc + vel_cog 
    # update the particle position based off new velocity updates
    def update_position(self):
        self.position = np.add(self.position, self.velocity)
        self.history.append(self.position)


class Algorithm():
    def __init__(self, x_in,history,  maxiter, num_particles, num_features, num_data_per_point):
        self.err_best_g = None  # best error for group
        self.pos_best_g = []  # best position for group
        self.num_particles = 3
        self.maxiter = maxiter
        # establish the swarm
        self.ae = AutoEncoder(x_in, num_features, random_seed=1234)
        self.history = history
        self.updates = []

        self.swarm = []
        for i in range(0, num_particles):
            w_in = np.random.normal(size=(num_data_per_point, num_features), scale=5)
            self.swarm.append(Particle(i, w_in, self.ae.psi, maxiter))

    def run(self):
        # begin optimization loop
        for i in range(self.maxiter):
            print("----------\n\t"+str(i))
            # cycle through particles in swarm and evaluate fitness
            for particle in self.swarm:
                particle.evaluate()
                print("p{} == cost: {}".format(particle.name, particle.cost))
                # determine if current particle is the best (globally)
                if self.err_best_g is None or particle.cost < self.err_best_g:
                    self.pos_best_g = particle.position
                    self.err_best_g = particle.cost
                    print("new best")
            dis = 0
            for p1 in self.swarm:
                for p2 in self.swarm:
                    dis += np.linalg.norm(p1.position-p2.position)
                break

            print("ave. distance between matrix: {}".format(dis))
            for particle in self.swarm:
                particle.update_velocity(self.pos_best_g, i)
                particle.update_position()
            if i % self.history == 0:
                self.updates.append(self.pos_best_g)

        return self.ae, self.pos_best_g , self.err_best_g, self.updates



def test_mnist():
    print("===== RUNNING MNIST =====")
    (train_x, _), (_, _) = mnist.load_data()
    train_x = train_x / 255
    plotter.plot_mnist(train_x, "original")  # Show original mnist images
    num_img, img_dim, _ = train_x.shape  # Get number of images and # pixels per square img
    mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))  # Reshape images to match autoencoder input

    history = 1
    maxiter = 5
    num_particles = 5
    for num_features in [700]:
        PSO = Algorithm(mnist_in, history, maxiter, num_particles, num_features, img_dim * img_dim)
        ae, w_in, least_squares_test,updated_history = PSO.run()
        print(f"(# features : Least squares error = ({num_features} : {least_squares_test})")
        for pos in range(len(updated_history)):
            z_grd, ls_grd, grd = ae.calc_g(updated_history[pos])  # Calculate Z, Error, and Gradient Matrix
            phi_w_img = ae.phi(updated_history[pos])  # Calculate phi(W)
            new_mnist = z_grd @ phi_w_img  # Recreate original images using Z and phi(W)
            new_imgs = np.reshape(new_mnist, train_x.shape)  # Reshape new images have original shape
            plotter.plot_mnist(new_imgs, f"{num_features}_features_{pos*history}_iteration")  # Show new images
        plotter.show_avail_plots()


if __name__ == '__main__':
    np.random.seed(1234)
    test_mnist()
