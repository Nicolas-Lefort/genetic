'''

Tune the hyperparameters to see the impact
Simple genetic algorithm implementation for educational purpose
The target in this code is to find a solution to 1*U + 1*V + 1*W + 1*X + 1*Y + 1*Z = 30
The function to minimize is f = 1*U + 1*V + 1*W + 1*X + 1*Y + 1*Z - 30 see also evaluation attribute in class Individual
The fitness represent how good a chromosome/solution is and is borned to 1
The selection process is done according to roulette wheel

Interesting links:

Some doc to understand the main concepts:
https://www.tutorialspoint.com/genetic_algorithms/genetic_algorithms_introduction.htm

An introduction to genetic algorithm with code implementation in C++ and Python by Atul Kumar :
https://www.geeksforgeeks.org/genetic-algorithms/

For more complex problems with multiple-objective or constrained problems see this paper by Julian Blank:
https://arxiv.org/pdf/2002.04504.pdf

I am not linked to the previously mentionned work and authors.

The code below can be copied and re-used
'''

import random
import numpy as np
import time

class Individual:

    def __init__(self, value=None, mutate_prob=0.01, available_genes=None , target=None):
        self.mutate_prob = mutate_prob
        self.available_genes = available_genes
        self.target = target

        if type(available_genes)!=list:
            raise Exception("Parameter available_genes must be a list")

        if type(target)!=list:
            raise Exception("Parameter target must be a list")

        if value is None:
            self.value = self.create()
        else:
            self.value = value

    @property
    def evaluation(self) -> np.array:
        '''
        :return: appy target function to candidate
        '''
        evaluation = np.dot(np.array(self.value),np.array(self.target)) - 30
        return evaluation

    @property
    def fitness(self) -> float:
        '''
        :return: normalize results between 0 and 1. A fitness = 1 means we reached our goal
        '''
        return 1/(1+abs(self.evaluation))

    def create(self) -> list:
        '''
        :return: initialize a random solution
        '''
        return [random.choice(self.available_genes) for _ in range(len(self.target))]

    def mutate(self) -> object:
        '''
        :return: pick an element at random index and mutate it a random new value
        '''
        if self.mutate_prob > np.random.rand():
            mutate_index = random.randint(0, len(self.target) - 1)
            self.value[mutate_index] = random.choice(self.available_genes)
        return self

class Population:

    def __init__(self,  population_size=100, mutate_prob=0.01, retain=0.2, available_genes=None , target=None, generations=1000):
        self.done = False
        self.population_size = population_size
        self.mutate_prob = mutate_prob
        self.retain = retain
        self.available_genes = available_genes
        self.target = target
        self.generations = generations
        self.individuals = self.create()
        self.fitness_list = []
        self.fitness_pop = 0

    def create(self) -> list:
        '''
        :return: initialize a random list of Individual using hyperparameters
        '''
        return [Individual(mutate_prob=self.mutate_prob, available_genes=self.available_genes, target=self.target)
                for _ in range(self.population_size)]

    def get_fitness(self):
        '''
        step before calculating cumulative probability
        '''
        self.fitness_list = [individual.fitness for individual in self.individuals]
        self.fitness_pop = np.sum(np.array(self.fitness_list))

    def evaluate(self) -> np.array:
        '''
        :return: calculate cumulative probabilty on sorted population
        '''
        self.individuals = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        if self.individuals[0].evaluation == 0:
            self.done = True
        self.get_fitness()
        return np.cumsum([fitness/self.fitness_pop for fitness in self.fitness_list])

    def turn_wheel(self):
        '''
        perfom selection and survival
        '''
        cumulative_proba = self.evaluate()
        retain_length = int(self.retain * len(self.individuals))
        self.parents = list(np.delete(self.individuals, np.where(np.array(cumulative_proba)<random.random())))
        self.parents = self.parents[:retain_length]

    def crossover(self):
        '''
        perform crossover - refill self.individuals with children coming from crossover of the retained parents
        '''
        target_children_size = self.population_size - len(self.parents)
        children = [] #initialize children
        if len(self.parents) > 0:
            while len(children) < target_children_size:
                father = random.choice(self.parents) #pick a random individual from self.parents
                mother = random.choice(self.parents) #pick a random individual from self.parents
                if father != mother:
                    child_value = [random.choice(pair) for pair in zip(father.value, mother.value)]
                    child = Individual(value=child_value,
                                       mutate_prob=self.mutate_prob,
                                       available_genes=self.available_genes,
                                       target=self.target)
                    children.append(child)
            self.individuals = self.parents + children

    def mutate(self):
        '''
        perform mutation. Mutation is an important step too
        Diversity must be introduced to suggest new path to the algo
        Without mutation, your algo have good chance to stay stuck on a local minimum
        '''
        self.individuals=[individual.mutate() for individual in self.individuals]

    def evolve(self):
        '''
        produce a new population according to selection/crossover/Mutation principles
        '''
        self.turn_wheel()
        self.crossover()
        self.mutate()
        self.parents =  [] #initialize parents for next generation

    def run(self):
        '''
        iterate the evolution process till target is reached or till max generation is reached
        '''
        a = time.time()
        for _ in range(self.generations):
            self.evaluate()
            if pop.done:
                print("Finished at generation: ", _ ,
                      "Solution was found at individual: ", self.individuals[0].value,
                      "Target: ", self.individuals[0].evaluation)
                break
            else:
                print("Generation: ", _,
                      "Best individual: ", self.individuals[0].value,
                      "Target: ", self.individuals[0].evaluation)
                self.evolve()
        b = time.time()
        print("Processing time :", round(b - a,4), " seconds")

if __name__ == "__main__":
    #Hyperparameters
    population_size = 1000
    mutate_prob = 0.8
    retain = 0.2
    available_genes = list(range(-10000, 10000))
    max_generations = 5000
    #Target
    target = [1, 1, 1, 1, 1, 1]
    pop = Population(population_size=population_size,
                     mutate_prob=mutate_prob,
                     retain=retain,
                     available_genes=available_genes,
                     target=target,
                     generations=max_generations)
    pop.run()
