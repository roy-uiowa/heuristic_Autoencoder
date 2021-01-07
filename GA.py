# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 15:55:47 2020

@author: Cory Kromer-Edwards
"""

import random
import numpy as np

from deap import base
from deap import creator
from deap import tools

from autoencoder import AutoEncoder
import plotter

# To turn input dictionary into namespace for easier access
from argparse import Namespace

# To pair-zip hyperparameter options
from itertools import product

# For testing purposes
from keras.datasets import mnist
import time

# =====================================================================================
# tools: A set of functions to be called during genetic algorithm operations (mutate, mate, select, etc)
# Documentation: https://deap.readthedocs.io/en/master/api/tools.html
# =====================================================================================

# =====================================================================================
# Creator: Creates a class based off of a given class (known as containers)
#   creator.create(class name, base class to inherit from, args**)
#     class name: what the name of the class should be
#     base class: What the created class should inherit from
#     args**: key-value argument pairs that the class should have as fields
#
#   EX: creator.create("FitnessMax", base.Fitness, weights=(1.0,))
#     Creates a class named "FitnessMax" that inherits from base fitness class
#     that library has (maximizes fitness value). It then has a tuple of weights
#     that are given as a field for the class to use later.
# =====================================================================================

# The base.Fitness function will try to maximize fitness*weight, so we want
# negative weight here so the closer we get to 0 (with domain (-inf, 0]) the
# larger the fitness will become.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin, generation=0)


# NOTE ON FITNESS WEIGHTS:
#   Weights will be used when finding the maximum fitness within the Deap library,
#   but you will see the fitness value that is return from evaluation function
#   IE: During fitness max function -> fitness * weights
#       When calling "individual.fitness.values" -> fitness / weights


# =====================================================================================
# Toolbox: Used to add aliases and fixed arguements for functions that we will use later.
#   toolbox.[un]register(alias name, function, args*)
#     alias name: name to give the function being added
#     function: the function that is being aliased in the toolbox
#     args*: arguments to fix for the function when calling it later
#
#   EX: toolbox.register("attr_bool", random.randint, 0, 1)
#     Creates an alias for the random.randint function with the name "attr_bool"
#       with the default min and max int values being passed in being 0 and 1.
#
#   EX: toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100)
#     Creates an alias for the tools.initRepeat function with the name "individual".
#     This function takes in the class that we want to repeadidly intialize from, the function to
#     initialize values with, and how many values to create from that function. This will create
#     an individual with 100 random boolean values.
# =====================================================================================

# toolbox = base.Toolbox()

def get_params_gs():
    """Get hyperparameter pairs to run through grid search"""
    mu = [1, 0.5]
    sigma = [0.5, 0.1]
    alpha = [0.5, 0.9]
    indpb = [0.1, 0.5]
    tournsize = [2, 3]
    cxpb = [0.5, 0.9]
    mutpb = [0.1, 0.5]
    options = product(mu, sigma, alpha, indpb, tournsize, cxpb, mutpb)
    return options


def to_string():
    """Get the name of the algorithm to be put on output strings"""
    return "ga"


class Algorithm:
    def __init__(self, **args):

        args = Namespace(**args)

        self.toolbox = base.Toolbox()

        self.stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        # if not pool:
        #     self.map_func = map
        # else:
        #     self.map_func = pool.map
        self.map_func = map

        if not hasattr(args, 'x'):
            raise ValueError("variable 'x' must be given as numpy array of shape (n x N)")
        else:
            x = args.x

        if not hasattr(args, 'num_features'):
            raise ValueError("variable 'num_features' must be given")
        else:
            num_features = args.num_features

        if not hasattr(args, 'mu'):
            args.mu = 0.5

        if not hasattr(args, 'sigma'):
            args.sigma = 0.5

        if not hasattr(args, 'alpha'):
            args.alpha = 0.9

        if not hasattr(args, 'indpb'):
            args.indpb = 0.1

        if not hasattr(args, 'tournsize'):
            args.tournsize = 2

        if not hasattr(args, 'debug'):
            self.debug = 0
        else:
            self.debug = args.debug

        if not hasattr(args, 'pop_size'):
            self.pop_size = 300
        else:
            self.pop_size = args.pop_size

        if not hasattr(args, 'number_generations'):
            self.num_gen = 100
        else:
            self.num_gen = args.number_generations

        if not hasattr(args, 'cxpb'):
            self.cxpb = 0.9
        else:
            self.cxpb = args.cxpb

        if not hasattr(args, 'mutpb'):
            self.mutpb = 0.1
        else:
            self.mutpb = args.mutpb

        self.ae = AutoEncoder(x, num_features, random_seed=1234, use_gpu=True)
        self.w_shape = (x.shape[0], num_features)

        # Set up ways to define individuals in the population
        self.toolbox.register("attr_x", np.random.normal, 0, 1)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_x, num_features * x.shape[0])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Set up ways to change population
        self.toolbox.register("mate", tools.cxBlend, alpha=args.alpha)
        self.toolbox.register("mutate", tools.mutGaussian, mu=args.mu, sigma=args.sigma, indpb=args.indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=args.tournsize)

    # Fitness evaluation methods (must return iterable)
    # Remember, we want to minimize these functions, so to hurt them we need to return
    # large positive numbers.
    # =====================================================================================
    def _evaluate(self, individual):
        w = np.reshape(individual, self.w_shape)
        # w = np.asarray(individual).transpose()
        _, cost = self.ae.psi(w)
        return (cost,)

    # =====================================================================================

    def run(self):
        """
        Run a genetic algorithm with the given evaluation function and input parameters.
        Main portion of code for this method found from Deap example at URL:
        https://deap.readthedocs.io/en/master/overview.html

        Parameters
        ----------
        None

        Returns
        -------
        best_individual: List
          The best individual found out of all iterations
        fitness: Float
          The best_individual's fitness value
        logbook : Dictionary
          A dictionary of arrays for iterations, min, max, average, and std. dev. for each iteration.

        """
        pop = self.toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(25, similar=np.allclose)
        logbook = tools.Logbook()

        # Evaluate the entire population
        fitnesses = list(self.map_func(self._evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
            ind.generation = 0

        record = self.stats.compile(pop) if self.stats else {}
        logbook.record(gen=0, **record)
        times = []

        for g in range(self.num_gen):
            start_time = time.time()
            # Select the next generation individuals (with replacement)
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals (since selection only took references rather than values)
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(self.map_func(self._evaluate, invalid_ind))

            if self.debug >= 2:
                print("Generation %i has (min, max) fitness values: (%.3f, %.3f)" % (
                    g, min(fitnesses)[0], max(fitnesses)[0]))
            elif self.debug == 1:
                plotter.print_progress_bar(g + 1, self.num_gen, suffix=f"Complete--(Gen: fitness): ({g + 1}, {min(fitnesses)[0]:.3f})")

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                ind.generation = g + 1

            # The population is entirely replaced by the offspring
            pop[:] = offspring
            hof.update(pop)
            record = self.stats.compile(pop) if self.stats else {}
            logbook.record(gen=g + 1, **record)
            times.append(time.time() - start_time)

        if self.debug >= 0:
            print("Problem results:")
            print(f"\tBest individual seen fitness value:\t\t{hof[0].fitness.values[0]:3f}")
            print(f"\tBest individual seen generation appeared in:\t{hof[0].generation}")

        gen, min_results, max_results, avg, std = logbook.select("gen", "min", "max", "avg", "std")
        return hof[0], hof[0].fitness.values[0], {"iterations": gen, "min": min_results, "max": max_results, "avg": avg,
                                                  "std": std, "times": times}


def test_grid_search():
    num_points = 65000
    num_data_per_point = 154
    x_in = np.random.normal(size=(num_data_per_point, num_points))
    num_features = 100
    total = len(list(get_params_gs()))
    iteration = 0

    best_values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    best_fitness = 90000000
    for params in get_params_gs():
        plotter.print_progress_bar(iteration, total, suffix=f"Complete--Best fitness: {best_fitness:.3f}")
        alg = Algorithm(x=x_in, mu=params[0], sigma=params[1], alpha=params[2], indpb=params[3],
                        tournsize=params[4], cxpb=params[5], mutpb=params[6], debug=-1,
                        num_features=num_features, pop_size=5, number_generations=20)
        _, fitness, _ = alg.run()

        if fitness <= best_fitness:
            best_fitness = fitness
            best_values = params

        iteration += 1

    print(
        "Best values from grid search evaluation "
        "is:\n\tMu:%.3f\n\tSigma:%.3f\n\tAlpha:%.3f\n\tIndpb:%.3f\n\tTournsize:%i\n\tCxpb:%.3f\n\tMutpb:%.3f "
        % best_values)
    print(f"Best parameters had fitness: {best_fitness:.3f}")


def test_random():
    # Sanity test to make sure that feature number positively impacts least squares error.
    num_points = 100
    num_data_per_point = 55
    x_in = np.random.normal(size=(num_data_per_point, num_points))
    loss_values = []
    for num_features in [1, 5, 10, 15, 20, 40, 70]:
        ga = Algorithm(x=x_in, num_features=num_features, debug=1)
        # w_in = np.random.normal(size=(num_data_per_point, num_features))
        w_out, best_cost, logs = ga.run()
        loss_values.append(best_cost)

    plotter.plot_loss(loss_values, "Random_Test_with_Features", "Num features from list [1, 5, 10, 15, 20, 40, 70]")


def test_mnist():
    # Gradient check using MNIST
    (train_x, _), (_, _) = mnist.load_data()
    train_x = train_x / 255                                             # Normalizing images
    # plotter.plot_mnist(train_x, "original")                           # Show original mnist images

    num_img, img_dim, _ = train_x.shape                                 # Get number of images and # pixels per square img
    num_features = 500
    mnist_in = np.reshape(train_x, (img_dim * img_dim, num_img))        # Reshape images to match autoencoder input
    ga = Algorithm(x=mnist_in, num_features=num_features, debug=1, pop_size=20)
    w_out, best_cost, logs = ga.run()

    print(f"Average time/generation (sec): {sum(logs['times']) / len(logs['times'])}")
    print(f"Total time to run GA (sec): {logs['times']}")

    ae = AutoEncoder(mnist_in, num_features, random_seed=1234, use_gpu=True)
    z, _ = ae.psi(w_out)
    phi_w_img = ae.phi(w_out)                                            # Calculate phi(W)
    new_mnist = z @ phi_w_img                                       # Recreate original images using Z and phi(W)
    new_imgs = np.reshape(new_mnist, train_x.shape)                     # Reshape new images have original shape
    plotter.plot_mnist(new_imgs, f"{num_features}_features_ga")   # Show new images

    # print(loss_values)
    plotter.plot_loss(logs['min'], "MNIST_Gradient_Loss_Over_Generations")


if __name__ == '__main__':
    np.random.seed(1234)
    # test_grid_search()
    # test_random()
    test_mnist()
