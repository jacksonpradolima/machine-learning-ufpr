#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from deap import base
from deap import creator
from deap import tools
from scipy.sparse import csr_matrix
from scoop import futures
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

features, labels = load_svmlight_file("data/data")

x, X_test, y, y_test = train_test_split(features, labels, test_size=0.3333, random_state=42, stratify=labels)
X_train, X_cv, y_train, y_cv = train_test_split(x, y, test_size=0.5, train_size=0.5, random_state=42, stratify=y)

clf = DecisionTreeClassifier(random_state=0)

def run_classifier(_X_train, _X_test, _y_test):
	model = clf.fit(_X_train, y_train)
	predictions = clf.predict(_X_test) 
	score = clf.score(_X_test, _y_test)
	probas = clf.predict_proba(_X_test)

	return predictions, probas, score

def select_idx(individual):
	return np.array(individual).nonzero()[0].tolist()

def select_features(columns, dataset):		
	df = pd.DataFrame(dataset.toarray())
	return csr_matrix(pd.DataFrame(df, columns=columns))	

def evaluate(individual, _x_test, _y_test):
	columns = select_idx(individual)
	return run_classifier(select_features(columns, X_train), select_features(columns, _x_test), _y_test)

def evalOneMax(individual):
	predictions, probas, score = evaluate(individual, X_cv, y_cv)
	return score,

graph = []
num_features = []

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

# SCOOP
toolbox.register("map", futures.map)

def main():
    random.seed(64)
	
    ## population size
    pop = toolbox.population(n=50)
    
    ## Probabilities for Crossover, Mutation and number of generations (iterations)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 300
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    
    # Begin the evolution
    for g in range(NGEN):
        print("\n-- Generation %i --" % g)
        
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
            #print("Ind: {} - Fit: {}".format(ind, fit))
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        length = len(pop)

        print("--- Fitness ---")
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
                
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
                
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        
        graph.append(max(fits))
        
        print("--- Features ---")
        nf = [np.count_nonzero(ind) for ind in pop]
        
        mean = sum(nf) / length
        sum2 = sum(x*x for x in nf)
        std = abs(sum2 / length - mean**2)**0.5
                
        print("  Min %s" % min(nf))
        print("  Max %s" % max(nf))
        print("  Avg %s" % mean)
        print("  Std %s" % std)        
        num_features.append(max(nf))
            
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))

    predictions, probas, score = evaluate(best_ind, X_test, y_test)

    print("Test score: {} - N. features selected: {}".format(score, np.count_nonzero(best_ind)))
    
    line =plt.plot(graph)
    plt.show()

    line =plt.plot(num_features)
    plt.show()

if __name__ == "__main__":
    main()
