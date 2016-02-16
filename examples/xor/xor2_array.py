"""
Testing NEAT algorithm on an BNP competition and see how it will perform.
"""

from __future__ import print_function

import math
import os
import time
import numpy as np

from neat import nn, parallel, population, visualize

xor_inputs = np.asarray(((0, 0), (0, 1), (1, 0), (1, 1)))
xor_outputs = np.asarray([0, 1, 1, 0])
xor_outputs = np.reshape(xor_outputs,(-1,1))
xor_sample_size = xor_outputs.shape[0]

def fitness(genome):
    """
    This function will be run in parallel by ParallelEvaluator.  It takes one
    argument (a single genome) and should return one float (that genome's fitness).

    Note that this function needs to be in module scope for multiprocessing.Pool
    (which is what ParallelEvaluator uses) to find it.  Because of this, make
    sure you check for __main__ before executing any code (as we do here in the
    last two lines in the file), otherwise you'll have made a fork bomb
    instead of a neuroevolution demo. :)
    """
    net = nn.create_feed_forward_phenotype(genome)

    error = 0.0
    outputs = net.array_activate(xor_inputs)
    diffs = (xor_outputs - outputs) ** 2
    error_sum = np.sum(diffs)
    return 1.0 - np.sqrt(error_sum / xor_sample_size)


def run():
    t0 = time.time()

    # Get the path to the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'xor2_config')

    # Use a pool of four workers to evaluate fitness in parallel.
    pe = parallel.ParallelEvaluator(fitness,3)

    pop = population.Population(config_path)
    pop.epoch(pe.evaluate, 400)

    print("total evolution time {0:.3f} sec".format((time.time() - t0)))
    print("time per generation {0:.3f} sec".format(((time.time() - t0) / pop.generation)))

    print('Number of evaluations: {0:d}'.format(pop.total_evaluations))

    # Verify network output against training data.
    print('\nBest network output:')
    winner = pop.most_fit_genomes[-1]
    net = nn.create_feed_forward_phenotype(winner)
    outputs = net.array_activate(xor_inputs)
    
    print("Expected XOR output : ", xor_outputs)
    print("Generated output : ", outputs)
    
    # Visualize the winner network and plot statistics.
    visualize.plot_stats(pop)
    visualize.plot_species(pop)
    visualize.draw_net(winner, view=True)


if __name__ == '__main__':
    run()
