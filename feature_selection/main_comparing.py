import argparse
import logging
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os
import pandas as pd
import random
import shutil

from deap import base
from deap import creator
from deap import tools
from scipy.sparse import csr_matrix
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

__author__ = "Jackson Antonio do Prado Lima"
__email__ = "jacksonpradolima@gmail.com"
__license__ = "GPL"
__version__ = "1.0"

# Split my data in 3 set with same size: Train, Test, and Validation.
# Train and Validation are used in the fitness function, after all, the Test dataset is used to evaluate the best individual
features, labels = load_svmlight_file("data/data")

x, X_test, y, y_test = train_test_split(features, labels, test_size=0.3333, random_state=42, stratify=labels)
X_train, X_cv, y_train, y_cv = train_test_split(x, y, test_size=0.5, train_size=0.5, random_state=42, stratify=y)

# Classifier instance
clf = DecisionTreeClassifier()

# Store fitness evolution
graph = []
# Store features
num_features = []

### Functions

def make_hist(ax, x, bins=None, binlabels=None, width=0.85, extra_x=1, extra_y=4, 
              text_offset=0.3, title=r"Frequency diagram", 
              xlabel="Features", ylabel="Frequency"):
    if bins is None:
        xmax = max(x)+extra_x
        bins = range(xmax+1)
    if binlabels is None:
        if np.issubdtype(np.asarray(x).dtype, np.integer):
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else 
                         '{}-{}'.format(bins[i], bins[i+1]-1)
                         for i in range(len(bins)-1)]
        else:
            binlabels = [str(bins[i]) if bins[i+1]-bins[i] == 1 else 
                         '{}-{}'.format(*bins[i:i+2])
                         for i in range(len(bins)-1)]
        if bins[-1] == np.inf:
            binlabels[-1] = '{}+'.format(bins[-2])
    n, bins = np.histogram(x, bins=bins)
    patches = ax.bar(range(len(n)), n, align='center', width=width)
    ymax = max(n)+extra_y

    # ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis='y')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')

def run_classifier(_X_train, _X_test, _y_test):
    """
    Execute the classifier
    """
    model = clf.fit(_X_train, y_train)
    predictions = clf.predict(_X_test) 
    score = clf.score(_X_test, _y_test)
    probas = clf.predict_proba(_X_test)

    return predictions, probas, score

def select_idx(individual):
    """
    Get the indexes with 1 (ones) from a individual - this means what features were selected
    """
    return np.array(individual).nonzero()[0].tolist()

def select_features(columns, dataset):
    """
    Filter the dataset only the features selected
    """
    df = pd.DataFrame(dataset.toarray())
    return csr_matrix(pd.DataFrame(df, columns=columns))

def evaluate(individual, _x_test, _y_test):
    """
    Evaluate the dataset with the feature set selected
    """
    columns = select_idx(individual)
    return run_classifier(select_features(columns, X_train), select_features(columns, _x_test), _y_test)

def evalOneMax(individual):
    """
    Fitness function
    """
    predictions, probas, score = evaluate(individual, X_cv, y_cv)
    return score,

def store_fitness(pop):
    # logging.debug("--- Fitness ---")
    length = len(pop)
    # Gather all the fitnesses in one list and logging.debug the stats
    fits = [ind.fitness.values[0] for ind in pop]
            
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
            
    # logging.debug("  Min %s" % min(fits))
    # logging.debug("  Max %s" % max(fits))
    # logging.debug("  Avg %s" % mean)
    # logging.debug("  Std %s" % std)
    
    graph.append(max(fits))

def store_features(pop):
    # logging.debug("--- Features ---")
    length = len(pop)
    nf = [np.count_nonzero(ind) for ind in pop]
    
    mean = sum(nf) / length
    sum2 = sum(x*x for x in nf)
    std = abs(sum2 / length - mean**2)**0.5
            
    # logging.debug("  Min %s" % min(nf))
    # logging.debug("  Max %s" % max(nf))
    # logging.debug("  Avg %s" % mean)
    # logging.debug("  Std %s" % std)
    num_features.append(max(nf))

### DEAP CONFIGUTATION ###
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 132)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operator registering
toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    if os.path.isdir("results"):
        shutil.rmtree(os.getcwd() + "/results")

    if os.path.isdir("plots"):
        shutil.rmtree(os.getcwd() + "/plots")

    os.makedirs("plots")
    os.makedirs("results")

    random.seed(64)

    # Population size
    populations = [50, 150]

    # Probabilities for Crossover, Mutation
    crossovers  = [0.5, 0.9, 0.1]
    mutations   = [0.2, 0.01, 0.9]

    # Independent runs and number of generations (iterations)
    runs, NGEN = 10, 300

    # Process Pool of 4 workers
    pool = multiprocessing.Pool(processes=4)
    toolbox.register("map", pool.map)

    with open("results/kruskal.csv", 'w') as f:
        f.write("Group;Value\n")
        with open("results/configuration.csv", 'w') as fc:
            for POP in populations:
                for CXPB in crossovers:
                    for MUTPB in mutations:
                        plt.figure()
                        individuals_idx = []
                        frequency = np.zeros(132)
                        labels = []
                        scores = []

                        configuration = "POP={}_CXPB={}_MUTPB={}".format(POP, CXPB, MUTPB)
                        logging.debug("\n\n\nConfiguration({})".format(configuration))

                        # loop runs
                        for i in range(runs):
                             # Clean global variable to next execution
                            del graph[:]

                            labels.append("Execução = " +str(i+1))
                            pop = toolbox.population(n=POP)

                            logging.debug("\n\nStart of evolution - Run {}".format(i+1))

                            # Evaluate the entire population
                            fitnesses = list(map(toolbox.evaluate, pop))
                            for ind, fit in zip(pop, fitnesses):
                                ind.fitness.values = fit
                            
                            # Begin the evolution
                            for g in range(NGEN):
                                logging.debug("-- Generation %i --" % g)

                                # Select the next generation individuals
                                offspring = toolbox.select(pop, len(pop))

                                # Clone the selected individuals
                                offspring = list(map(toolbox.clone, offspring))

                                # Apply crossover and mutation on the offspring
                                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                                    if random.random() < CXPB:
                                        toolbox.mate(child1, child2)
                                        del child1.fitness.values
                                        del child2.fitness.values

                                for mutant in offspring:
                                    if random.random() < MUTPB:
                                        toolbox.mutate(mutant)
                                        del mutant.fitness.values

                                # Evaluate the individuals with an invalid fitness
                                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

                                #invalid_ind = [ind for ind in offspring]
                                fitnesses = map(toolbox.evaluate, invalid_ind)
                                for ind, fit in zip(invalid_ind, fitnesses):
                                    ind.fitness.values = fit

                                # The population is entirely replaced by the offspring
                                pop[:] = offspring

                                store_fitness(pop)
                                #store_features(pop)

                            logging.debug("-- End of (successful) evolution --")

                            best_ind = tools.selBest(pop, 1)[0]
                            predictions, probas, score = evaluate(best_ind, X_test, y_test)

                            logging.debug("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
                            logging.debug("Test score: {}\nNumber of features selected: {}".format(score, np.count_nonzero(best_ind)))

                            plt.plot(range(1,NGEN+1), graph)

                            f.write("{};{}\n".format(configuration, score*100))
                            scores.append(score*100)

                            frequency += best_ind
                            individuals_idx += select_idx(best_ind)

                        logging.debug("-- End of (successful) Independent runs --")
                        logging.debug("\n\nFrequency: {}\n--------\nFeatures selected >= 90%: {}\n--------\nFeatures selected <= 30%: {}\n--------\nFeatures not used: {}".format(frequency, np.where(frequency >= runs*0.9), np.where(frequency <= runs*0.3), np.where(frequency <= 0)))

                        fc.write("{};".format(configuration))
                        for s in scores:
                            fc.write("{};".format(s))
                        fc.write("\n")

                        plt.legend(labels, ncol=4, loc='upper center', bbox_to_anchor=[0.5, 1.1], 
                                   columnspacing=1.0, labelspacing=0.0,
                                   handletextpad=0.0, handlelength=1.5,
                                   fancybox=True, shadow=True)

                        plt.ylabel('Escore')
                        plt.xlabel('Geração')
                        plt.savefig("plots/fitness_evolution_{}.pdf".format(configuration))
                        plt.close()

                        plt.figure()
                        fig, ax = plt.subplots()
                        make_hist(ax, individuals_idx, bins=np.arange(0, 132), extra_y=1, xlabel="Características", ylabel="Frequência")
                        plt.tight_layout()
                        plt.savefig("plots/frequency_diagram_{}.pdf".format(configuration))
    pool.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    args = ap.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    main()
