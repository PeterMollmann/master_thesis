import ray

from alpine.gp import util
from deap import gp
import warnings
import re
import numpy as np
import pygmo as pg


def check_trig_fn(ind):
    return len(re.findall("cos", str(ind))) + len(re.findall("sin", str(ind))) + len(re.findall("tan", str(ind)))


def check_nested_trig_fn(ind):
    return util.detect_nested_trigonometric_functions(str(ind))


def eval_model(individual, D, consts=[]):
    warnings.filterwarnings("ignore")
    y_pred = individual(*D.X, consts)
    return y_pred


def compute_MSE(individual, D, consts=[]):
    y_pred = eval_model(individual, D, consts)
    MSE = np.mean((D.y - y_pred) ** 2)
    if np.isnan(MSE) or np.isinf(MSE):
        MSE = 1e8
    return MSE


# Compiles and finds the number of constants to update in each tree
def compile_individual_with_consts(tree, toolbox, special_term_name="a"):
    const_idx = 0
    tree_clone = toolbox.clone(tree)
    for i, node in enumerate(tree_clone):
        if isinstance(node, gp.Terminal) and node.name[0:3] != "ARG":
            if node.name == special_term_name:
                new_node_name = special_term_name + "[" + str(const_idx) + "]"
                tree_clone[i] = gp.Terminal(new_node_name, True, float)
                const_idx += 1

    individual = toolbox.compile(
        expr=tree_clone, extra_args=[special_term_name])
    return individual, const_idx


# evaluate trees using MSE and tune constats in tree using Pygmo (genetic algoithm)
def eval_MSE_and_tune_constants(tree, toolbox, D):
    individual, num_consts = compile_individual_with_consts(tree, toolbox)

    if num_consts > 0:

        # config()

        def eval_MSE(consts):
            warnings.filterwarnings("ignore")
            y_pred = individual(*D.X, consts)
            total_err = np.mean((D.y - y_pred) ** 2)
            return total_err

        objective = eval_MSE

        x0 = np.ones(num_consts)

        class fitting_problem:
            def fitness(self, x):
                total_err = objective(x)
                return [total_err]

            def get_bounds(self):
                return (-5.0 * np.ones(num_consts), 5.0 * np.ones(num_consts))

        # PYGMO SOLVER
        prb = pg.problem(fitting_problem())
        # algo = pg.algorithm(pg.nlopt(solver="lbfgs"))
        # algo.extract(pg.nlopt).maxeval = 10
        # algo = pg.algorithm(pg.cmaes(gen=70))
        algo = pg.algorithm(pg.pso(gen=10))
        # algo = pg.algorithm(pg.sea(gen=70))
        pop = pg.population(prb, size=10)
        # pop = pg.population(prb, size=1)
        pop.push_back(x0)
        pop = algo.evolve(pop)
        MSE = pop.champion_f[0]
        consts = pop.champion_x
        # print(pop.problem.get_fevals())
        if np.isinf(MSE) or np.isnan(MSE):
            MSE = 1e8
    else:
        MSE = compute_MSE(individual, D)
        consts = []
    return MSE, consts


def get_features_batch(
        individuals_str_batch,
        individ_feature_extractors=[len, check_nested_trig_fn, check_trig_fn]):

    features_batch = [[fe(i) for i in individuals_str_batch]
                      for fe in individ_feature_extractors]

    individ_length = features_batch[0]
    nested_trigs = features_batch[1]
    num_trigs = features_batch[2]
    return individ_length, nested_trigs, num_trigs


@ray.remote
def eval_expr(individuals_str_batch, toolbox, dataset, penalty, fitness_scale):

    predictions = [None] * len(individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        predictions[i] = eval_model(callable, dataset, consts=tree.consts)

    return predictions


@ray.remote
def compute_MSEs(individuals_str_batch, toolbox, dataset, penalty, fitness_scale):

    total_errs = [None] * len(individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):
        callable, _ = compile_individual_with_consts(tree, toolbox)
        total_errs[i] = compute_MSE(callable, dataset, consts=tree.consts)

    return total_errs


@ray.remote
def compute_attributes(individuals_str_batch, toolbox, dataset, penalty, fitness_scale):
    attributes = [None] * len(individuals_str_batch)

    individ_length, nested_trigs, num_trigs = get_features_batch(
        individuals_str_batch)

    for i, tree in enumerate(individuals_str_batch):

        # Tarpeian selection
        if individ_length[i] >= 50:
            consts = None
            fitness = (1e8,)
        else:
            MSE, consts = eval_MSE_and_tune_constants(tree, toolbox, dataset)
            fitness = (
                fitness_scale
                * (
                    MSE
                    + 100000 * nested_trigs[i]
                    + 0.0 * num_trigs[i]
                    + penalty["reg_param"] * individ_length[i]
                ),
            )
        attributes[i] = {"consts": consts, "fitness": fitness}
    return attributes


def assign_attributes(individuals, attributes):
    for ind, attr in zip(individuals, attributes):
        ind.consts = attr["consts"]
        ind.fitness.values = attr["fitness"]
