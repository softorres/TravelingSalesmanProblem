#TSP project
#Purpose: From the starting point of the path, we wanna find the shorterst path of the distance.
#Last day edited: 6/20/2023
#Autors: Mia Marte, Sofia Torres, Stephanie Saenz

import random

import numpy as np
import pandas as pd
import streamlit as st
#import permutation
from collections import namedtuple
from numpy.random import permutation
from copy import copy
import matplotlib.pyplot as plt



from streamlit_solver import genetic_tsp
from utils import read_input

st.set_page_config(layout="wide")

"""
# Genetic TSP

Demo of genetic algorithm solver for the Traveling Salesman Problem.

Feel free to play with the parameters in the sidebar and see how they impact the
solution.

"""

with st.sidebar:
    select_dataset = st.selectbox(
        label="Select a dataset",
        options=("p01.in", "dj15.in", "dj38.in", "att48.in", "qa194.in"),
    )

    num_generations = st.number_input(
        "Number of generations", min_value=10, max_value=5000, step=10
    )

    population_size = st.number_input(
        "Population size", min_value=10, max_value=5000, step=10
    )

    mutation_prob = st.number_input(
        "Mutation probability", min_value=0.0, max_value=1.0, value=0.1
    )

    random_seed_checkbox = st.checkbox("Set a random seed?")

    if random_seed_checkbox:
        random_seed = st.number_input("Random seed", min_value=0, step=1, value=42)
        random.seed(random_seed)
        np.random.seed(random_seed)

###########indivual.py
class Individual:
    def __init__(self, genes: list[int]):
        self.genes = genes

def random_individual(num_genes: int) -> Individual:
    return Individual(genes=permutation(range(num_genes)))
###########
def test_can_create_random_individual():
    num_genes = 5
    
    individual = random_individual(num_genes=num_genes)

    assert sorted(individual.genes) == list(range(num_genes))      
     ###################################################   
        
City = namedtuple("City", ["x", "y"])


def pairwise(iterable):
    iterable = list(iterable)
    return zip(iterable, iterable[1:] + iterable[:1])


def random_interval(individual):
    i, j = random.sample(range(len(individual)), k=2)
    i, j = min(i, j), max(i, j)
    return i, j


def random_population(population_size, num_genes):
    return [list(permutation(range(num_genes))) for _ in range(population_size)]


def read_input(path):
    cities = []
    with open(path, "r") as input_file:
        for line in input_file:
            x, y = line.split(" ")
            city = City(float(x), float(y))
            cities.append(city)
    return cities      
        
        
        
#########################################################  
random.seed(42)
np.random.seed(42)


cities = None


def crossover(parent1, parent2):
    size = len(parent1)

    i, j = random_interval(parent1)

    c1 = parent1[i:j]
    c2 = parent2[i:j]

    for k in range(size):
        child_pos = (j + k) % size

        if parent2[child_pos] not in c1:
            c1.append(parent2[child_pos])

        if parent1[child_pos] not in c2:
            c2.append(parent1[child_pos])

    c1 = c1[-i:] + c1[:-i]
    c2 = c2[-i:] + c2[:-i]

    return c1, c2


def evaluate_fitness(individual):
    dist = 0
    for x, y in pairwise(individual):
        dist += math.dist(cities[y], cities[x])
    return 1 / dist


def mutate(individual, prob):
    for i in range(len(individual)):
        if random.random() < prob:
            i, j = random_interval(individual)
            individual[i], individual[j] = individual[j], individual[i]
    return individual


def breed(population, mutation_prob):
    offspring = []

    for i in range(len(population)):
        for j in range(i + 1, len(population)):
            child1, child2 = crossover(population[i], population[j])
            child1 = mutate(child1, mutation_prob)
            child2 = mutate(child2, mutation_prob)
            offspring.extend([child1, child2])

    return population + offspring


def rank_selection(population, num_selected):
    pop_by_fitness = sorted(
        population, key=lambda ind: evaluate_fitness(ind), reverse=True
    )
    return pop_by_fitness[:num_selected]


def genetic_tsp(
    dataset_name,
    num_generations,
    population_size,
    mutation_prob,
    chart,
    plot,
    progress_bar,
    current_distance,
):
global cities
    cities = read_input(f"data/{dataset_name}")

    population = random_population(population_size, len(cities))

    pop_fitness = [evaluate_fitness(individual) for individual in population]
    best_solution = population[np.argmax(pop_fitness)]
    best_distance = 1 / evaluate_fitness(best_solution)

    progress_bar.progress(0)

    current_distance.text("")

    solution = copy(best_solution)
    solution.append(solution[0])

    fig, ax = plt.subplots()

    ax.plot(
        [cities[i].x for i in solution],
        [cities[i].y for i in solution],
        "-o",
    )

    plot.pyplot(fig)

    chart.line_chart()

    for gen in range(num_generations):
        population_with_offspring = breed(population, mutation_prob)
        population = rank_selection(population_with_offspring, population_size)

        pop_fitness = [evaluate_fitness(individual) for individual in population]
        best_solution = population[np.argmax(pop_fitness)]
        best_distance = 1 / evaluate_fitness(best_solution)

        progress_bar.progress(int(gen / num_generations * 100))
        current_distance.write(f"Current distance: {best_distance}")
        chart.add_rows({"Distance": [best_distance]})

        solution = copy(best_solution)
        solution.append(solution[0])
        ax.clear()
        ax.plot(
            [cities[i].x for i in solution],
            [cities[i].y for i in solution],
            "-o",
        )

        plot.pyplot(fig)
    progress_bar.empty()

    return best_solution, best_distance


##########################################
col1, col2 = st.columns(2)

col1.header("Best solution")
progress_bar = st.empty()
current_distance = st.empty()
plot = col1.empty()
done = st.empty()
final_distance = st.empty()

optimal_distances = {
    "p01.in": 284,
    "dj15.in": 3172,
    "dj38.in": 6656,
    "att48.in": 33523,
    "qa194.in": 9352,
}
optimal_distance = st.write(
    f"**Optimal Distance:** {optimal_distances[select_dataset]}"
)

col2.header("Distance over time")
df = pd.DataFrame({"Distance": []})
chart = col2.empty()


## Run the Genetic Algorithm
best_solution, best_distance = genetic_tsp(
    select_dataset,
    num_generations,
    population_size,
    mutation_prob,
    chart,
    plot,
    progress_bar,
    current_distance)

progress_bar.empty()
current_distance.empty()

cities = read_input(f"data/{select_dataset}")


done.write("**Done**!")
final_distance.write(f"**Final Distance:** {best_distance}")


