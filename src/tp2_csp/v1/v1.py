import random
import time
from collections import Counter
from typing import Any, cast

# The standard length of a steel bar available from the supplier (e.g., in mm)
BAR_LENGTH = 5600

# The required cuts: {length: quantity}
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

POPULATION_SIZE = 300  # Number of solutions in each generation
NUM_GENERATIONS = 300  # Number of evolutionary cycles
TOURNAMENT_SIZE = 5  # Number of individuals to compete in selection
MUTATION_RATE = 0.5  # Probability of a mutation occurring
CROSSOVER_RATE = 0.8  # Probability of crossover occurring

type Individual = list[int]


def calculate_fitness(individual: Individual, stock_length: int) -> int:
    """
    Calculates the fitness of an individual (a permutation of cuts).
    The fitness is the number of stock bars required for that cutting order.
    A lower number of bars means a better fitness.
    """
    bars_used = 0
    current_bar_length = 0

    for cut in individual:
        if cut > current_bar_length:
            bars_used += 1
            current_bar_length = stock_length
        current_bar_length -= cut

    return bars_used


def selection(population: list[Individual], fitnesses: list[int], k: int = TOURNAMENT_SIZE) -> Individual:
    """
    Selects a single individual from the population using tournament selection.
    """
    # Select k random individuals from the population
    tournament_contenders_indices = random.sample(range(len(population)), k)

    # Find the contender with the best fitness (lowest number of bars)
    best_contender_index = min(tournament_contenders_indices, key=lambda idx: fitnesses[idx])

    return population[best_contender_index]


def ordered_crossover(parent1: Individual, parent2: Individual) -> Individual:
    """
    Performs ordered crossover (OX1) to create a child from two parents.
    This version correctly handles lists with duplicate values.
    """
    size = len(parent1)
    child = [-1] * size

    # 1. Select a random slice
    start, end = sorted(random.sample(range(size), 2))

    # 2. Copy the slice from parent1 to the child
    slice_from_p1 = parent1[start : end + 1]
    child[start : end + 1] = slice_from_p1

    # 3. Get genes from parent2, correctly handling duplicates.
    #    We create a temporary list of the slice elements. When we find a
    #    matching element in parent2, we remove it from our temp list to "account for it".
    slice_elements_to_find = list(slice_from_p1)
    genes_from_p2: list[int] = []
    for item in parent2:
        if item in slice_elements_to_find:
            slice_elements_to_find.remove(item)  # Account for one instance of the duplicate
        else:
            genes_from_p2.append(item)

    # 4. Fill the remaining slots in the child with the collected genes from parent2
    current_p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            child[i] = genes_from_p2[current_p2_idx]
            current_p2_idx += 1

    return child


def swap_mutation(individual: Individual) -> Individual:
    """
    Performs swap mutation on an individual.
    Two random genes (cuts) are swapped.
    """
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


# ==============================================================================
# --- MAIN GENETIC ALGORITHM EXECUTION ---
# ==============================================================================


def run_genetic_algorithm(stock_length: int, required_cuts: dict[int, int]) -> Individual:
    """
    Runs the genetic algorithm to find an optimal cutting solution.
    """
    # Create a flat list of all individual cuts needed
    all_cuts: list[int] = []
    for length, quantity in required_cuts.items():
        all_cuts.extend([length] * quantity)

    # --- Initialization ---
    print("🧬 Initializing population...")
    population = [random.sample(all_cuts, len(all_cuts)) for _ in range(POPULATION_SIZE)]

    best_solution_so_far = None
    best_fitness_so_far = float("inf")

    # --- Evolution Loop ---
    start_time = time.time()
    for generation in range(NUM_GENERATIONS):
        # Calculate fitness for the entire population
        fitnesses = [calculate_fitness(ind, stock_length) for ind in population]

        # Find the best individual in the current generation
        min_fitness = min(fitnesses)
        if min_fitness < best_fitness_so_far:
            best_fitness_so_far = min_fitness
            best_solution_so_far = population[fitnesses.index(min_fitness)]
            print(
                f"Generation {generation + 1}/{NUM_GENERATIONS} | New best solution found: {best_fitness_so_far} bars"
            )

        # --- Create the next generation ---
        next_generation: list[Individual] = []

        # Elitism: Keep the best individual from the current population
        elite = sorted(population, key=lambda ind: calculate_fitness(ind, stock_length))[0]
        next_generation.append(elite)

        # Generate the rest of the new population through crossover and mutation
        while len(next_generation) < POPULATION_SIZE:
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)

            if random.random() < CROSSOVER_RATE:
                child = ordered_crossover(parent1, parent2)
            else:
                child = parent1  # No crossover, just copy a parent

            if random.random() < MUTATION_RATE:
                child = swap_mutation(child)

            next_generation.append(child)

        population = next_generation

    end_time = time.time()
    print(f"\n✅ Evolution complete in {end_time - start_time:.2f} seconds.")
    return cast(Individual, best_solution_so_far)


# ==============================================================================
# --- DISPLAY RESULTS ---
# ==============================================================================


def display_solution(solution: Individual, stock_length: int):
    """
    Prints the cutting plan for the best solution found by the GA.
    """
    if not solution:
        print("No solution was found.")
        return

    bars: list[dict[str, Any]] = []
    current_bar_cuts: list[int] = []
    current_bar_length = stock_length

    # Re-run the packing process for the best solution to get the details
    for cut in solution:
        if cut <= current_bar_length:
            current_bar_cuts.append(cut)
            current_bar_length -= cut
        else:
            bars.append({"cuts": current_bar_cuts, "waste": current_bar_length})
            current_bar_cuts = [cut]
            current_bar_length = stock_length - cut

    # Add the last bar to the list
    bars.append({"cuts": current_bar_cuts, "waste": current_bar_length})

    print("\n" + "=" * 50)
    print("          OPTIMAL CUTTING PLAN")
    print("=" * 50)

    total_waste = 0
    for i, bar in enumerate(bars):
        total_waste += bar["waste"]
        cuts_summary: Any = dict(Counter(bar["cuts"]))
        print(f"Bar {i + 1:02d}: Cuts = {cuts_summary} | Waste = {bar['waste']}mm")

    total_cut_length = sum(solution)
    total_stock_length = len(bars) * stock_length
    efficiency = (total_cut_length / total_stock_length) * 100

    print("\n" + "-" * 50)
    print("                SUMMARY")
    print("-" * 50)
    print(f"Total steel bars required: {len(bars)}")
    print(f"Total length of stock used: {total_stock_length}mm")
    print(f"Total length of pieces cut: {total_cut_length}mm")
    print(f"Total waste (offcuts): {total_waste}mm")
    print(f"Material Efficiency: {efficiency:.2f}%")
    print("=" * 50)


# ==============================================================================
# --- MAIN EXECUTION BLOCK ---
# ==============================================================================


def run() -> None:
    best_cutting_order = run_genetic_algorithm(BAR_LENGTH, REQUIRED_CUTS)
    display_solution(best_cutting_order, BAR_LENGTH)
