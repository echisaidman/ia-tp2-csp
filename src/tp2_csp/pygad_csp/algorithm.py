import random
from collections import Counter
from typing import Any, Callable, Literal

import numpy as np
import numpy.typing as npt
import pygad

from tp2_csp.pygad_csp.generalized_order_crossover import generalized_order_crossover

STOCK_LENGTH = 5600
REQUIRED_CUTS = {
    1380: 22,
    1520: 25,
    1560: 12,
    1710: 14,
    1820: 18,
    1880: 18,
    1930: 20,
    2000: 10,
    2050: 12,
    2100: 14,
    2140: 16,
    2150: 18,
    2200: 20,
}

gene_space: list[int] = []
for length, count in REQUIRED_CUTS.items():
    gene_space.extend([length] * count)

sol_per_pop = 500  # Number of solutions in the population.
num_genes = len(gene_space)  # Number of genes in each solution.


# Create the initial population. Each chromosome is a random permutation
# of the required cuts.
def create_initial_population() -> list[list[int]]:
    initial_population: list[list[int]] = []
    for _ in range(sol_per_pop):
        chromosome = gene_space[:]
        random.shuffle(chromosome)
        initial_population.append(chromosome)
    return initial_population


initial_population = create_initial_population()


def stock_elements_used_by_solution(solution: npt.NDArray[np.int64]) -> int:
    stock_elements_used = 0
    current_stock_length = 0
    for cut_length in solution:
        if cut_length <= current_stock_length:
            # The cut fits in the current stock element
            current_stock_length -= cut_length
        else:
            # The cut does not fit, start a new stock element
            stock_elements_used += 1
            current_stock_length = STOCK_LENGTH - cut_length
    return stock_elements_used


def assert_solution_is_valid(solution: npt.NDArray[np.int64]) -> None:
    genes_to_use = Counter(gene_space)
    for gene in solution:
        genes_to_use[gene] -= 1
    assert all(count == 0 for count in genes_to_use.values())


def fitness_func(ga_instance: pygad.GA, solution: npt.NDArray[np.int64], solution_idx: int):
    """
    Calculates the fitness of a solution. The fitness is inversely proportional
    to the number of stock elements used.
    A higher fitness value means a better solution.
    """
    assert_solution_is_valid(solution)
    stock_elements_used = stock_elements_used_by_solution(solution)

    # We want to minimize the number of stock elements, but PyGAD maximizes
    # fitness. So, we use the reciprocal.
    fitness = 1.0 / stock_elements_used
    return fitness


def ordered_crossover(
    parents: npt.NDArray[np.int64],
    offspring_size,
    ga_instance: pygad.GA,
    # ) -> npt.NDArray[npt.NDArray[np.int64]]:
):
    """
    Performs Ordered Crossover (OX1) for permutation-based chromosomes.
    It correctly handles chromosomes with duplicate gene values.

    Args:
        parents (numpy.ndarray): The selected parents.
        offspring_size (tuple): The size of the offspring to be produced.

    Returns:
        numpy.ndarray: An array of the generated offspring.
    """
    offspring: list[list[int]] = []
    idx = 0
    while len(offspring) < offspring_size[0]:
        parent1: list[int] = parents[idx % parents.shape[0], :].tolist()
        parent2: list[int] = parents[(idx + 1) % parents.shape[0], :].tolist()

        size = len(parent1)
        child = [-1] * size

        start, end = sorted(random.sample(range(size), 2))

        p1_slice = parent1[start : end + 1]
        child[start : end + 1] = p1_slice

        # Get genes from parent2 that are not in the slice from parent1.
        # This handles duplicates correctly by "ticking off" each item from the slice.
        # Use a Counter to track the frequency of each gene in p1_slice
        p1_slice_counts = Counter(p1_slice)
        p2_genes_to_add: list[int] = []
        for gene in parent2:
            if p1_slice_counts[gene] > 0:
                p1_slice_counts[gene] -= 1
            else:
                p2_genes_to_add.append(gene)

        # 4. Fill the remaining empty slots in the child
        p2_idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = p2_genes_to_add[p2_idx]
                p2_idx += 1

        offspring.append(child)
        idx += 1

    return np.array(offspring)


def inverse_mutation(offspring: npt.NDArray[np.int64], ga_instance: pygad.GA):
    for chromosome_idx in range(offspring.shape[0]):
        start, end = sorted(random.sample(range(offspring.shape[1]), 2))
        sub_sequence = offspring[chromosome_idx, start : end + 1]
        sub_sequence = list(reversed(sub_sequence))
        offspring[chromosome_idx, start : end + 1] = sub_sequence
    return offspring


def on_generation(ga_instance: pygad.GA):
    if ga_instance.generations_completed % 20 == 0:
        print(f"Generation {ga_instance.generations_completed}")
        print("Stock elements used by best solution:", stock_elements_used_by_solution(ga_instance.best_solution()[0]))


selection_type: Literal["sss", "rws", "rank", "tournament"] = "tournament"
crossover_type: Callable[..., Any] = (
    # ordered_crossover
    generalized_order_crossover
)
mutation_type: Literal["swap", "inversion"] | Callable[[npt.NDArray[np.int64], pygad.GA], npt.NDArray[np.int64]] = (
    # inverse_mutation
    # "swap"
    "inversion"
)

ga_instance = pygad.GA(
    num_generations=3_000,
    sol_per_pop=sol_per_pop,
    initial_population=initial_population,
    fitness_func=fitness_func,
    num_genes=num_genes,
    gene_type=int,  # The cuts are integers
    # num_parents_mating=2,
    num_parents_mating=10,
    parent_selection_type=selection_type,
    K_tournament=20,
    # keep_parents=0,
    crossover_type=crossover_type,
    mutation_type=mutation_type,
    # mutation_percent_genes=10,  # 10% of genes will be swapped
    # mutation_percent_genes=20,
    # Stop if fitness doesn't improve for 20 generations
    # stop_criteria="saturate_20",
    on_generation=on_generation,
)

# Run the GA
ga_instance.run()

# --- 5. Displaying the Results ---
best_solution: list[int] = ga_instance.best_solution()[0]  # type: ignore
best_solution_fitness: float = ga_instance.best_solution()[1]  # type: ignore
best_stocks_used: float = 1.0 / best_solution_fitness

print(f"Best solution found: \n{best_solution}\n")
print(f"Fitness of the best solution: {best_solution_fitness}")
print(f"Number of stock elements used by the best solution: {int(best_stocks_used)}\n")

# To verify the best solution, let's print how the cuts are laid out.
print("--- Layout of Cuts for the Best Solution ---")
stock_elements_used = 0
current_stock_length = 0
for cut_length in best_solution:
    if current_stock_length >= cut_length:
        current_stock_length -= cut_length
        print(f"  - Cut of length {cut_length}. Remaining length: {current_stock_length}")
    else:
        stock_elements_used += 1
        current_stock_length = STOCK_LENGTH - cut_length
        print(f"\nStock Element #{stock_elements_used} (Length: {STOCK_LENGTH}):")
        print(f"  - Cut of length {cut_length}. Remaining length: {current_stock_length}")
