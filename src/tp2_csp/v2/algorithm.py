import pprint
import random
from collections import Counter

from .individual import Individual

type Population = list[Individual]


class CSPGeneticAlgorithm:
    def __init__(
        self,
        required_cuts: dict[int, int],
        population_size: int,
        generations: int,
        bar_length: int,
        elitism_size: int,
        tournament_size: int,
        crossover_rate: float,
        mutation_rate: float,
    ) -> None:
        self.all_cuts = self.__flatten_cuts(required_cuts)
        self.population_size = population_size
        self.generations = generations
        self.bar_length = bar_length
        self.elitism_size = elitism_size
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate

    def run(self) -> Individual:
        """
        Runs the genetic algorithm to find the optimal cutting pattern.
        """
        print("Genetic Algorithm for Cutting Stock Problem")
        print("------------------------------------------")

        population = self.__initialize_population()
        best_solution_ever = max(population, key=lambda x: x.fitness_score)

        for gen in range(self.generations):
            # Sort by fitness (descending) to easily find the best
            sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)

            # Update the best solution ever found
            current_best = sorted_population[0]
            if current_best.fitness_score > best_solution_ever.fitness_score:
                best_solution_ever = current_best

            if gen == 0 or (gen + 1) % 20 == 0 or gen == self.generations - 1:
                print(
                    f"Generation {gen + 1:4d}: Best Fitness = {best_solution_ever.fitness_score:.2f} | "
                    f"Bars Used = {best_solution_ever.calculate_required_bars()} | "
                    f"Total Waste = {best_solution_ever.calculate_total_waste()} | "
                    f"% Wasted = {best_solution_ever.calculate_percentage_wasted():.2f}%"
                )

            # 2. Create Next Generation
            next_population: list[Individual] = []

            # Elitism: carry over the best individuals
            elite = [ind for ind in sorted_population[: self.elitism_size]]
            next_population.extend(elite)

            # Prepare population for selection (fitness, chromosome only)
            selection_pool = [ind for ind in sorted_population]

            # Fill the rest of the new population
            while len(next_population) < self.population_size:
                parent1 = self.__selection(selection_pool)
                parent2 = self.__selection(selection_pool)

                child1, child2 = self.__crossover(parent1, parent2)

                next_population.append(self.__mutate(child1))
                if len(next_population) < self.population_size:
                    next_population.append(self.__mutate(child2))

            population = next_population

        print("\nEvolution finished.")
        return best_solution_ever

    def __flatten_cuts(self, cuts: dict[int, int]) -> list[int]:
        """Converts the dictionary of cuts into a single list of all cuts."""
        flat_list: list[int] = []
        for length, quantity in cuts.items():
            flat_list.extend([length] * quantity)
        return flat_list

    def __initialize_population(self) -> Population:
        """Creates the initial population with random permutations of cuts."""
        population: Population = []
        for _ in range(self.population_size):
            chromosome = self.all_cuts[:]
            random.shuffle(chromosome)
            population.append(Individual(self.bar_length, chromosome))
        return population

    def __selection(self, population: Population) -> Individual:
        """
        Selects a parent using Tournament Selection.
        """
        tournament = random.sample(population, self.tournament_size)
        # The rated_population is sorted descending by fitness, so the first element is the best
        tournament = sorted(tournament, key=lambda x: x.fitness_score, reverse=True)
        return tournament[0]

    def __crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs Ordered Crossover (OX1) to create two children.
        This preserves the permutation nature of the chromosomes.
        """
        if random.random() > self.crossover_rate:
            child1 = Individual(self.bar_length, parent1.chromosome[:])
            child2 = Individual(self.bar_length, parent2.chromosome[:])
            return child1, child2

        size = len(parent1.chromosome)
        child1_chromosome, child2_chromosome = [-1] * size, [-1] * size

        start, end = sorted(random.sample(range(size), 2))

        # Copy the slice from parents to children
        slice_from_p1 = parent1.chromosome[start : end + 1]
        slice_from_p2 = parent2.chromosome[start : end + 1]
        child1_chromosome[start : end + 1] = slice_from_p1
        child2_chromosome[start : end + 1] = slice_from_p2

        # Fill remaining genes for Child 1 (from Parent 2)
        p1_slice_counter = Counter(slice_from_p1)
        genes_from_p2_for_child1: list[int] = []
        for item in parent2.chromosome:
            if p1_slice_counter[item] > 0:
                p1_slice_counter[item] -= 1
            else:
                genes_from_p2_for_child1.append(item)

        p2_idx = 0
        for i in range(size):
            if child1_chromosome[i] == -1:
                child1_chromosome[i] = genes_from_p2_for_child1[p2_idx]
                p2_idx += 1

        # Fill remaining genes for Child 2 (from Parent 1)
        p2_slice_counter = Counter(slice_from_p2)
        genes_from_p1_for_child1: list[int] = []
        for item in parent1.chromosome:
            if p2_slice_counter[item] > 0:
                p2_slice_counter[item] -= 1
            else:
                genes_from_p1_for_child1.append(item)

        p1_idx = 0
        for i in range(size):
            if child2_chromosome[i] == -1:
                child2_chromosome[i] = genes_from_p1_for_child1[p1_idx]
                p1_idx += 1

        return Individual(self.bar_length, child1_chromosome), Individual(self.bar_length, child2_chromosome)

    def __mutate(self, individual: Individual) -> Individual:
        """
        Performs Inversion Mutation on a chromosome.
        A random sub-sequence is selected and its order is reversed.
        """
        if random.random() > self.mutation_rate:
            return individual

        start, end = sorted(random.sample(range(len(individual.chromosome)), 2))
        chromosome = individual.chromosome[:]

        # Reverse the sub-sequence
        sub_sequence = chromosome[start : end + 1]
        sub_sequence = list(reversed(sub_sequence))
        chromosome[start : end + 1] = sub_sequence

        return Individual(self.bar_length, chromosome)


def print_best_solution(best_solution: Individual, bar_length: int) -> None:
    """
    Prints the details of the best solution found.
    """
    print("\n=================================")
    print("      Best Solution Found      ")
    print("=================================")
    print(f"Bars Required: {best_solution.calculate_required_bars()}")
    print(f"Total Waste: {best_solution.calculate_total_waste()} units")
    print(f"% Wasted: {best_solution.calculate_percentage_wasted():.2f}%")
    print("---------------------------------\n")

    distinct_cuts = sorted(set(best_solution.chromosome))
    cuts_quantity = {cut: best_solution.chromosome.count(cut) for cut in distinct_cuts}
    print("Cuts in the solution (to check against the required cuts)")
    pprint.pprint(cuts_quantity)
    print("---------------------------------\n")

    for i, bar in enumerate(best_solution.cuts_layout):
        bar_sum = sum(bar)
        waste = bar_length - bar_sum
        print(f"Bar {i + 1:2d} (Length: {bar_length}):")
        print(f"  Cuts:  {bar}")
        print(f"  Sum:   {bar_sum}")
        print(f"  Waste: {waste}\n")


def run() -> None:
    BAR_LENGTH = 5600

    # Define the required cuts: {length: quantity}
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

    # --- GA Parameters ---
    POPULATION_SIZE = 300
    GENERATIONS = 300
    TOURNAMENT_SIZE = 5
    MUTATION_RATE = 0.50
    CROSSOVER_RATE = 0.80
    ELITISM_SIZE = 5

    # --- Run the Algorithm ---
    solver = CSPGeneticAlgorithm(
        REQUIRED_CUTS,
        POPULATION_SIZE,
        GENERATIONS,
        BAR_LENGTH,
        ELITISM_SIZE,
        TOURNAMENT_SIZE,
        CROSSOVER_RATE,
        MUTATION_RATE,
    )
    solution = solver.run()
    print_best_solution(solution, BAR_LENGTH)
