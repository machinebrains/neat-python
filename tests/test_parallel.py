import os
from neatsociety.population import Population
from neatsociety import parallel


# dummy fitness function
def eval_fitness(individual):
    return 1.0


def test_minimal():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'test_configuration')

    pop = Population(config_path)
    pe = parallel.ParallelEvaluator(eval_fitness,4)
    pop.run(pe.evaluate, 400)
