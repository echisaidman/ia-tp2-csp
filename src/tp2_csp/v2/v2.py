import random
from typing import Dict, List, Tuple

# Type aliases for clarity
type Chromosome = list[int]
type Population = list[Chromosome]
type Result = tuple[float, List[List[int]], int, int]  # fitness, layout, bars, waste


class GeneticAlgorithmCSP:
    """
    Solves the Cutting Stock Problem using a Genetic Algorithm.
    """

    def __init__(
        self,
        bar_length: int,
        cuts: Dict[int, int],
        population_size: int = 100,
        generations: int = 200,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.2,
        elitism_size: int = 2,
        tournament_size: int = 5,
    ):
        """
        Initializes the Genetic Algorithm solver.

        Args:
            bar_length (int): The length of each stock bar.
            cuts (Dict[int, int]): A dictionary of required cuts {length: quantity}.
            population_size (int): The number of individuals in each generation.
            generations (int): The number of generations to run the algorithm for.
            crossover_rate (float): The probability of crossover occurring.
            mutation_rate (float): The probability of an individual mutating.
            elitism_size (int): The number of best individuals to carry over to the next generation.
            tournament_size (int): The number of individuals to select for a tournament.
        """
        self.bar_length = bar_length
        self.required_cuts = cuts
        self.all_cuts = self._flatten_cuts(cuts)
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        self.tournament_size = tournament_size

        self.population: Population = []
        self.best_solution_ever: Result = (
            0.0,
            [],
            0,
            0,
        )

    def _flatten_cuts(self, cuts: dict[int, int]) -> list[int]:
        """Converts the dictionary of cuts into a single list of all cuts."""
        flat_list: list[int] = []
        for length, quantity in cuts.items():
            flat_list.extend([length] * quantity)
        return flat_list

    def _initialize_population(self) -> Population:
        """Creates the initial population with random permutations of cuts."""
        population: Population = []
        for _ in range(self.population_size):
            chromosome = self.all_cuts[:]
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def _calculate_fitness(self, chromosome: Chromosome) -> tuple[float, List[List[int]], int, int]:
        """
        Calculates the fitness of a chromosome using a First Fit heuristic.

        Fitness is inversely proportional to the total waste.

        Returns:
            A tuple containing: (fitness_score, layout_of_bars, num_bars_used, total_waste).
        """
        bars: list[list[int]] = []
        bar_remainders: list[int] = []

        for cut in chromosome:
            placed = False
            # Try to place the cut in an existing bar (First Fit)
            for i, remainder in enumerate(bar_remainders):
                if cut <= remainder:
                    bars[i].append(cut)
                    bar_remainders[i] -= cut
                    placed = True
                    break

            # If it doesn't fit anywhere, start a new bar
            if not placed:
                bars.append([cut])
                bar_remainders.append(self.bar_length - cut)

        total_waste = sum(bar_remainders)
        num_bars = len(bars)

        # Fitness: Higher is better. We want to minimize waste.
        # Adding 1.0 to the denominator to avoid division by zero.
        fitness = 1.0 / (total_waste + 1.0)

        return fitness, bars, num_bars, total_waste

    def _selection(self, rated_population: List[Tuple[float, Chromosome]]) -> Chromosome:
        """
        Selects a parent using Tournament Selection.
        """
        tournament = random.sample(rated_population, self.tournament_size)
        # The rated_population is sorted descending by fitness, so the first element is the best
        tournament.sort(key=lambda x: x[0], reverse=True)
        return tournament[0][1]

    def _crossover(self, parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """
        Performs Ordered Crossover (OX1) to create two children.
        This preserves the permutation nature of the chromosomes.
        """
        if random.random() > self.crossover_rate:
            return parent1[:], parent2[:]

        size = len(parent1)
        child1, child2 = [-1] * size, [-1] * size

        start, end = sorted(random.sample(range(size), 2))

        # Copy the slice from parents to children
        slice_from_p1 = parent1[start : end + 1]
        slice_from_p2 = parent2[start : end + 1]
        child1[start : end + 1] = slice_from_p1
        child2[start : end + 1] = slice_from_p2

        # Get genes from parent2, correctly handling duplicates.
        # We create a temporary list of the slice elements. When we find a
        # matching element in parent2, we remove it from our temp list to "account for it".
        slice_elements_to_find_from_p1 = list(slice_from_p1)
        genes_from_p2: list[int] = []
        for item in parent2:
            if item in slice_elements_to_find_from_p1:
                slice_elements_to_find_from_p1.remove(item)  # Account for one instance of the duplicate
            else:
                genes_from_p2.append(item)

        # Fill the rest of child1 from parent2
        p2_idx = 0
        for i in range(size):
            if child1[i] == -1:
                child1[i] = genes_from_p2[p2_idx]
                p2_idx += 1

        # Get genes from parent1, correctly handling duplicates.
        # We create a temporary list of the slice elements. When we find a
        # matching element in parent1, we remove it from our temp list to "account for it".
        slice_elements_to_find_from_p2 = list(slice_from_p2)
        genes_from_p1: list[int] = []
        for item in parent1:
            if item in slice_elements_to_find_from_p2:
                slice_elements_to_find_from_p2.remove(item)  # Account for one instance of the duplicate
            else:
                genes_from_p1.append(item)

        # Fill the rest of child2 from parent1
        p1_idx = 0
        for i in range(size):
            if child2[i] == -1:
                child2[i] = genes_from_p1[p1_idx]
                p1_idx += 1

        return child1, child2

    def _mutate(self, chromosome: Chromosome) -> Chromosome:
        """
        Performs Swap Mutation on a chromosome.
        """
        if random.random() < self.mutation_rate:
            idx1, idx2 = random.sample(range(len(chromosome)), 2)
            chromosome[idx1], chromosome[idx2] = chromosome[idx2], chromosome[idx1]
        return chromosome

    def run(self) -> None:
        """
        Runs the genetic algorithm to find the optimal cutting pattern.
        """
        self.population = self._initialize_population()
        print("Genetic Algorithm for Cutting Stock Problem")
        print("------------------------------------------")
        print(f"Starting evolution for {self.generations} generations...")

        for gen in range(self.generations):
            # 1. Evaluate Fitness
            rated_population: list[tuple[float, Chromosome, list[list[int]], int, int]] = []
            for chromo in self.population:
                fitness, layout, num_bars, waste = self._calculate_fitness(chromo)
                rated_population.append((fitness, chromo, layout, num_bars, waste))

            # Sort by fitness (descending) to easily find the best
            rated_population = sorted(rated_population, key=lambda x: x[0], reverse=True)

            # Update the best solution ever found
            current_best = rated_population[0]
            if current_best[0] > self.best_solution_ever[0]:
                self.best_solution_ever = (current_best[0], current_best[2], current_best[3], current_best[4])

            if (gen + 1) % 20 == 0:
                print(
                    f"Generation {gen + 1:4d}: Best Fitness = {self.best_solution_ever[0]:.4f} | "
                    f"Bars Used = {self.best_solution_ever[2]} | "
                    f"Total Waste = {self.best_solution_ever[3]}"
                )

            # 2. Create Next Generation
            next_population: list[Chromosome] = []

            # Elitism: carry over the best individuals
            elite = [chromo for _, chromo, _, _, _ in rated_population[: self.elitism_size]]
            next_population.extend(elite)

            # Prepare population for selection (fitness, chromosome only)
            selection_pool = [(fit, chromo) for fit, chromo, _, _, _ in rated_population]

            # Fill the rest of the new population
            while len(next_population) < self.population_size:
                parent1 = self._selection(selection_pool)
                parent2 = self._selection(selection_pool)

                child1, child2 = self._crossover(parent1, parent2)

                next_population.append(self._mutate(child1))
                if len(next_population) < self.population_size:
                    next_population.append(self._mutate(child2))

            self.population = next_population

        print("\nEvolution finished.")

    def print_best_solution(self) -> None:
        """
        Prints the details of the best solution found.
        """
        _, best_layout, num_bars, total_waste = self.best_solution_ever
        print("\n=================================")
        print("      Best Solution Found      ")
        print("=================================")
        print(f"Bars Required: {num_bars}")
        print(f"Total Waste: {total_waste} units")
        print("---------------------------------\n")

        for i, bar in enumerate(best_layout):
            bar_sum = sum(bar)
            waste = self.bar_length - bar_sum
            print(f"Bar {i + 1:2d} (Length: {self.bar_length}):")
            print(f"  Cuts:  {bar}")
            print(f"  Sum:   {bar_sum}")
            print(f"  Waste: {waste}\n")


def run() -> None:
    STOCK_BAR_LENGTH = 5600

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
    POPULATION_SIZE = 150
    GENERATIONS = 300
    MUTATION_RATE = 0.25
    CROSSOVER_RATE = 0.85
    ELITISM_SIZE = 5

    # --- Run the Algorithm ---
    solver = GeneticAlgorithmCSP(
        bar_length=STOCK_BAR_LENGTH,
        cuts=REQUIRED_CUTS,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE,
        elitism_size=ELITISM_SIZE,
    )

    solver.run()
    solver.print_best_solution()
