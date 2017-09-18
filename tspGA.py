# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:22:14 2017

@author: Emanuele

Genetic algorithm attack to TSP: this version is a bit harder version of classical TSP
A solution is a series of vertices that the salesman visits in order and deliveries the package
The main difference wrt the classical TSP is that if you specify that the salesman travels the vertices
{1,5}, he takes the shortest path between them but he does not delivery anything except for the vertices {1,5}
So a solution that randomly passes through a good path is not accetable, you have to know what to do not by chance
"""

import numpy as np
import graph as gr
import shortestpath as sp
import pylab as pl

# generate a random TSP instance with 
#   the number of vertices V
#   a given edge density
# return 
#   the graph G and the shortest path matrix SP_cost
def generateRandomTSPInstance(V, e):
    M = gr.generateRandMatrix(V, e); # generate a random adjacency matrix with n vertices, and a probability to be connected to each other vertex set to p
    G = gr.generateRandomGraph(M, np.shape(M)[0], 0, 0, 0); # generate the graph with the adjacency matrix M
    n = np.size(G.getAdjacencyMatrix()[0]);
    SP, SP_cost = np.array(sp.shortest_path(G.getAdjacencyMatrix(),n,n));
    connected_adj = gr.fromSpToAdjacency(sp.connectMatrix(SP_cost));    
    #print("Adjacency graph of the connected matrix is\n", connected_adj);
    for i in range(np.shape(connected_adj)[0]):
        G.getVertex(i).setAdjacents(connected_adj[i], G);    
    diameter = np.matrix.max((sp.shortest_path(G.getAdjacencyMatrix(),n,n))[1]).astype(int); # calculate the diameter in order to set the min/max deadlines on G
    G.setAllDeadlines(diameter, (2*diameter)+1);  
    return G, SP_cost;

# fitness function on a single element of the population
# return the lenght of the path that this cromosome encodes
def fitness(cromosome, SP, V):
    cost = 0;
    for c in range(V-1):
        cost += SP[cromosome[c],cromosome[c+1]];        
    return cost;

# fitness function on a single element of the population
# return the lenght of the path that this cromosome encodes
def fitnessPopulation(population, SP, V):
    cost = np.zeros(len(population));
    index = 0;
    for p in population:
        for c in range(V-1):
            cost[index] += SP[p[c],p[c+1]];        
        index += 1;
    return cost;

# return 'number_of_pools' pools, of size 'pool_size' each, ready to be "crossovered",
# according to a selection policy, in this case we use the Tournament Policy:
#   we take k elements at random from the population, and we put them in a pool
#   we order them by fitness descending, and assign to the first one a probability
#   to be selected of p, to the second one p(1-p), to the third one p(1-p)^2 and so on..
#   In this way we obtain a certain number of the element of the population, we iterate through all
#   the pools and we get the new population!
# Implementation detail: generate more than 'population_size' elements, then take the best ones!
def tournamentDispositions(pool_size, number_of_pools, population_size, V):
    tournaments = np.array([]);
    for n in range(number_of_pools):
        tournaments = np.append(tournaments, [population[k] for k in np.random.permutation(population_size)[:pool_size]]);
    return tournaments.reshape(number_of_pools, int(population_size/number_of_pools), V+1);

# torunament selection function:
# take the population and split them among number_of_pools 'pools' each of size 'pool_size' (pool_size*number_of_pools) = population_size
# for each pool select two elements according to a geometric probability function (the first in the population is assigned a probability p, the second one p(1-p), the thirs one p(1-p)^2 etc.)
def tournamentSelection(tournament_p, p_crossover, pool_size, number_of_pools, population_size, V):
    new_population = np.array([]);
    probability_vector = np.array([tournament_p*np.power(1-tournament_p, i) for i in range(pool_size)])
    probability_vector[-1] += (1-sum(probability_vector)); # add a probability to the last element in such a way that the probability_vector is a probability vector (sum up to 1)
    for p in range(1,len(probability_vector)):
        probability_vector[p] += probability_vector[p-1];
    tournaments = tournamentDispositions(pool_size, number_of_pools, population_size, V);
    for t in tournaments:
        t = t[t[:,-1].argsort()];
        index_generated = 0;
        while(index_generated < pool_size):
            temp = probability_vector - np.random.rand();
            n = np.where(temp>=0)[0][0];
            c1 = t[n];            
            temp = probability_vector - np.random.rand();
            c2 = t[np.where(temp>=0)][0];
            new_population = np.append(new_population, orderedCrossover(c1.astype(int), c2.astype(int), p_crossover, V));
            index_generated += 1;
    new_population = new_population.reshape(int(population_size)*2, V+1);    
    new_population = new_population[new_population[:,-1].argsort()][:population_size];
    #print(new_population);    
    return new_population;

# take two cromosomes c1, c2 according to a selection criterionand with a given probability p_crossover:
#   decides a split point (at least one) u:
#   take all the points of the first cromosome c1 till u and put them into the new cromosome c1'
#       and fill the rest of the cromosome with the other elements of c2, in order of appearance in c2
#       do the same for c2'
def orderedCrossover(c1, c2, p_crossover, V):
    if np.random.rand() <= p_crossover:
        crossover_point = np.random.randint(1, V);
        c1_new = np.append(c1[:crossover_point], np.setdiff1d(c2[:-1], c1[:crossover_point]));
        c2 = np.append(c1[crossover_point:-1], np.setdiff1d(c2[:-1], c1[crossover_point:-1]));
        c1_new = np.append(c1_new, fitness(c1_new, SP, V));
        c2 = np.append(c2, fitness(c2, SP, V));
        return c1_new, c2;
    else:
        return c1, c2;
    
# take a cromosome and with probability p_mutation for each allele, randomly permutate it with another point    
def orderedMutation(p_mutation):
    for p in population:
        for el in range(len(p[:-1])):
            if np.random.rand() <= p_mutation:
                temp = p[el];
                n = np.random.randint(0,len(p[:-1]));
                p[el] = p[n];
                p[n] = temp;
                break;
                
# plot the data with different colors    
def plot(fitnesses, epochs):
    pl.plot(np.array([e for e in range(epochs+1)]), fitnesses);
    
""" Test """

### Parameters assignment for the problem ###
V = 150; # dimension of the instance of the graph
e = 0.2; # edge density
population_size = 30; # size of the population
probability_crossover = 0.9; # probability that between two elements crossover happens
probability_mutation = 0.05; # probability that for each allele, mutation happens
pool_size = 10; # size of each torunament's pool
number_of_pools = 3; # number of pools
tournament_p = 0.75;
epochs = 1000; # number of epochs we perform the algorithm
elitism_number = 5;
elite = np.array([]); # array with the elite population (i.e. elitism)
### end of parameter assignment ###

### Parameters to plot the results ###
fitnesses = np.array([]);
### end of parameters to plot ###

graph_instance, SP = generateRandomTSPInstance(V, e);
population = np.array([np.append(np.random.permutation(V), V*gr.getDiameter(graph_instance)) for i in range(population_size)]);
for p in range(population_size):
    population[p][-1] = fitness(population[p], SP, V); # attach to each element the respective fitness    
population = population[population[:,-1].argsort()][:population_size]; # order the population by decreasing fitness

print("Statring the algoritm with this population: ");
print(population); 

fitnesses = np.array(population[0][-1]);

for e in range(epochs): 
    # save the best elements of the population to be preserved for the next round
    elite = population[:elitism_number];
    population = tournamentSelection(tournament_p, probability_crossover, pool_size, number_of_pools, population_size, V)
    orderedMutation(probability_mutation); # mutation
    population = population[population[:,-1].argsort()][:population_size]; # order pop by decreasing fitness
    population[-elitism_number:] = elite[:elitism_number];
    population = population[population[:,-1].argsort()]; # order pop by decreasing fitness
    fitnesses = np.append(fitnesses, population[0][-1]);
print("\n New best solution is \n", (population[0].astype(int)));
plot(fitnesses, epochs); # plot the fitnesses against the epochs of training