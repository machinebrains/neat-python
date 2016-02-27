"""
A parallel version of XOR using neatsociety.parallel with numpy arrays.

Since XOR is a simple experiment, a parallel version probably won't run any
faster than the single-process version, due to the overhead of
inter-process communication.

If your evaluation function is what's taking up most of your processing time
(and you should probably check by using a profiler while running
single-process), you should see a significant performance improvement by
evaluating in parallel.

This example is only intended to show how to do a parallel experiment
in neatsociety-python.  You can of course roll your own parallelism mechanism
or inherit from ParallelEvaluator if you need to do something more complicated.
"""

from __future__ import print_function

import math
import os
import time
import numpy as np

from neatsociety import nn, parallel, population, visualize

# Network inputs and expected outputs.
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
    sum_square_errors = (xor_outputs - outputs) ** 2
    error_sum = np.sum(sum_square_errors)
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
    pop.run(pe.evaluate, 400)

    print("total evolution time {0:.3f} sec".format((time.time() - t0)))
    print("time per generation {0:.3f} sec".format(((time.time() - t0) / pop.generation)))

    print('Number of evaluations: {0:d}'.format(pop.total_evaluations))

    # Verify network output against training data.
    print('\nBest network output:')
    winner = pop.statistics.best_genome()
    net = nn.create_feed_forward_phenotype(winner)
    outputs = net.array_activate(xor_inputs)
    
    print("Expected XOR output : ", xor_outputs)
    print("Generated output : ", outputs)
    
    # Visualize the winner network and plot statistics.
    visualize.plot_stats(pop.statistics)
    visualize.plot_species(pop.statistics)
    visualize.draw_net(winner, view=True)


if __name__ == '__main__':
    run()
