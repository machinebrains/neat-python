import os

from neatsociety.genes import NodeGene, ConnectionGene
from neatsociety.genome import Genome, FFGenome
from neatsociety import activation_functions
from neatsociety.reproduction import DefaultReproduction
from neatsociety.stagnation import DefaultStagnation

try:
    from configparser import ConfigParser
except ImportError:
    from ConfigParser import SafeConfigParser as ConfigParser


class Config(object):
    '''
    A simple container for all of the user-configurable parameters of NEAT.
    '''

    # TODO: Add the ability to write a Config to text file.

    # TODO: Split out the configuration into implementation-specific sections. For example,
    # a node gene class FooNode would expect to find a [FooNode] section within the configuration
    # file, and the NEAT framework doesn't need to know about this section in any way. This
    # allows all the configuration to stay in one text file, without unnecessary complication.
    # It also makes the config file and associated setup code somewhat self-documenting, as the
    # classes you need to give to NEAT are shown in the config file.

    allowed_connectivity = ['fs_neat', 'fully_connected', 'partial']

    def __init__(self, filename):
        self.registry = {'DefaultStagnation': DefaultStagnation,
                         'DefaultReproduction': DefaultReproduction}
        self.type_config = {}
        self.load(filename)

    def load(self, filename):
        if not os.path.isfile(filename):
            raise Exception('No such config file: ' + os.path.abspath(filename))

        with open(filename) as f:
            parameters = ConfigParser()
            if hasattr(parameters, 'read_file'):
                parameters.read_file(f)
            else:
                parameters.readfp(f)

        if not parameters.has_section('Types'):
            raise RuntimeError("'Types' section not found in NEAT configuration file.")

        # Phenotype configuration
        self.input_nodes = int(parameters.get('phenotype', 'input_nodes'))
        self.output_nodes = int(parameters.get('phenotype', 'output_nodes'))
        self.hidden_nodes = int(parameters.get('phenotype', 'hidden_nodes'))
        self.initial_connection = parameters.get('phenotype', 'initial_connection')
        self.connection_fraction = None
        self.max_weight = float(parameters.get('phenotype', 'max_weight'))
        self.min_weight = float(parameters.get('phenotype', 'min_weight'))
        self.feedforward = bool(int(parameters.get('phenotype', 'feedforward')))
        self.weight_stdev = float(parameters.get('phenotype', 'weight_stdev'))
        self.activation_functions = parameters.get('phenotype', 'activation_functions').strip().split()

        # Verify that initial connection type is valid.
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise Exception("'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify that specified activation functions are valid.
        for fn in self.activation_functions:
            if not activation_functions.is_valid(fn):
                raise Exception("Invalid activation function name: {0!r}".format(fn))

        # Select a genotype class.
        if self.feedforward:
            self.genotype = FFGenome
        else:
            self.genotype = Genome

        # Genetic algorithm configuration
        self.pop_size = int(parameters.get('genetic', 'pop_size'))
        self.init_pop_size = int(parameters.get('genetic', 'init_pop_size'))
        self.max_fitness_threshold = float(parameters.get('genetic', 'max_fitness_threshold'))
        self.prob_add_conn = float(parameters.get('genetic', 'prob_add_conn'))
        self.prob_add_node = float(parameters.get('genetic', 'prob_add_node'))
        self.prob_delete_conn = float(parameters.get('genetic', 'prob_delete_conn'))
        self.prob_delete_node = float(parameters.get('genetic', 'prob_delete_node'))
        self.prob_mutate_bias = float(parameters.get('genetic', 'prob_mutate_bias'))
        self.bias_mutation_power = float(parameters.get('genetic', 'bias_mutation_power'))
        self.prob_mutate_response = float(parameters.get('genetic', 'prob_mutate_response'))
        self.response_mutation_power = float(parameters.get('genetic', 'response_mutation_power'))
        self.prob_mutate_weight = float(parameters.get('genetic', 'prob_mutate_weight'))
        self.prob_replace_weight = float(parameters.get('genetic', 'prob_replace_weight'))
        self.weight_mutation_power = float(parameters.get('genetic', 'weight_mutation_power'))
        self.prob_mutate_activation = float(parameters.get('genetic', 'prob_mutate_activation'))
        self.prob_toggle_link = float(parameters.get('genetic', 'prob_toggle_link'))
        self.reset_on_extinction = bool(int(parameters.get('genetic', 'reset_on_extinction')))

        # genotype compatibility
        self.compatibility_threshold = float(parameters.get('genotype compatibility', 'compatibility_threshold'))
        self.excess_coefficient = float(parameters.get('genotype compatibility', 'excess_coefficient'))
        self.disjoint_coefficient = float(parameters.get('genotype compatibility', 'disjoint_coefficient'))
        self.weight_coefficient = float(parameters.get('genotype compatibility', 'weight_coefficient'))

        # Gene types
        self.node_gene_type = NodeGene
        self.conn_gene_type = ConnectionGene

        stagnation_type_name = parameters.get('Types', 'stagnation_type')
        reproduction_type_name = parameters.get('Types', 'reproduction_type')

        if stagnation_type_name not in self.registry:
            raise Exception('Unknown stagnation type: {!r}'.format(stagnation_type_name))
        self.stagnation_type = self.registry[stagnation_type_name]
        self.type_config[stagnation_type_name] = parameters.items(stagnation_type_name)

        if reproduction_type_name not in self.registry:
            raise Exception('Unknown reproduction type: {!r}'.format(reproduction_type_name))
        self.reproduction_type = self.registry[reproduction_type_name]
        self.type_config[reproduction_type_name] = parameters.items(reproduction_type_name)

        # Gather statistics for each generation.
        try:
            self.collect_statistics = bool(int(parameters.get('RunControl', 'collect_statistics')))
        except Exception as e:
            self.collect_statistics = True
            #print("Warning : " + str(e))
            #print("Note : Setting collect_statistics to True.")

        # Show stats after each generation.
        try:
            self.report = bool(int(parameters.get('RunControl', 'report_stats_per_gen')))
        except Exception as e:
            self.report = True
            # print("Warning : " + str(e))
            # print("Note : Setting report_stats_per_gen to True.")

        # Save the best genome from each generation.
        try:
            self.save_best = bool(int(parameters.get('RunControl', 'save_best')))
        except Exception as e:
            self.save_best = False
            # print("Warning : " + str(e))
            # print("Note : Setting save_best to False.")

        # Time in minutes between saving checkpoints, None for no timed checkpoints.
        try:
            self.checkpoint_time_interval = int(parameters.get('RunControl', 'checkpoint_per_minute'))
        except Exception as e:
            self.checkpoint_time_interval = None
            # print("Warning : " + str(e))
            # print("Note : no checkpoints will be generated per time.")

        # Generations between saving checkpoints, None for no generational checkpoints.
        try:
            self.checkpoint_gen_interval = int(parameters.get('RunControl', 'checkpoint_per_generations'))
        except Exception as e:
            self.checkpoint_gen_interval = None
            # print("Warning : " + str(e))
            # print("Note : no checkpoints will be generated per generations.")

        # Society directory to save the society check points and best individual.
        try:
            self.society_directory = parameters.get('RunControl', 'control_society_file').strip()
            if not os.path.isdir(self.society_directory):
                print("Error: Models directory path provided does not exist: " + str(self.society_directory))
                exit()
        except Exception as e:
            self.society_directory = None
            # print("Warning : " + str(e))
            # print("Note : Will save the models in the current working directory.")

        # Society directory to save the society check points and best individual.
        try:
            self.society_directory = parameters.get('RunControl', 'problem_society_file').strip()
            if not os.path.isdir(self.society_directory):
                print("Error: Models directory path provided does not exist: " + str(self.society_directory))
                exit()
        except Exception as e:
            self.society_directory = None
            # print("Warning : " + str(e))
            # print("Note : Will save the models in the current working directory.")

        # # Clean Socicty directory per generations
        # try:
        #     self.clean_society_dir_per_generations = bool(int(parameters.get('RunControl', 'clean_society_dir_per_generations')))
        # except Exception as e:
        #     self.clean_society_dir_per_generations = False
        #     # 0 will not clean the society directory from previous models
        #     # print("Warning : " + str(e))
        #     # print("Note : Will save the models in the current working directory.")

    def register(self, typeName, typeDef):
        """
        User-defined classes mentioned in the config file must be provided to the
        configuration object before the load() method is called.
        """

        self.registry[typeName] = typeDef

    def get_type_config(self, typeInstance):
        return dict(self.type_config[typeInstance.__class__.__name__])
