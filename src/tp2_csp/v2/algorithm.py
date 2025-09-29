import pprint
import random
from collections import Counter

from .common_types import Chromosome, CutsLayoutStrategy, MutationStrategy
from .individual import Individual, Population
from .parameters import Parameters
from .result import SimulationResult


class CSPGeneticAlgorithm:
    def __init__(self, parameters: Parameters) -> None:
        self.parameters = parameters
        self.bar_length = parameters.bar_length
        self.required_cuts = parameters.required_cuts
        self.all_cuts = self.__flatten_cuts(self.required_cuts)
        self.population_size = parameters.population_size
        self.generations = parameters.generations
        self.elitism_size = parameters.elitism_size
        self.tournament_size = parameters.tournament_size
        self.crossover_rate = parameters.crossover_rate
        self.mutation_rate = parameters.mutation_rate
        self.mutation_strategy: MutationStrategy = parameters.mutation_strategy
        self.cuts_layout_strategy: CutsLayoutStrategy = parameters.cuts_layout_strategy
        self.solution_target = parameters.solution_target

    def run(self) -> SimulationResult:
        """
        Runs the genetic algorithm to find the optimal cutting pattern.
        """
        print("Genetic Algorithm for Cutting Stock Problem")
        print("------------------------------------------")

        population = self.__initialize_population()
        best_solutions_by_generation: list[Individual] = []
        best_solution_ever: Individual | None = None

        gen = 0
        while not self.__should_stop(gen, best_solution_ever):
            # Sort by fitness (descending) to easily find the best
            sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)

            # Update the best solution for this generation
            current_best = sorted_population[0]
            best_solutions_by_generation.append(current_best)
            if best_solution_ever is None or current_best.fitness_score > best_solution_ever.fitness_score:
                best_solution_ever = current_best

            print(
                f"Generation {gen + 1:4d}: Best Fitness = {current_best.fitness_score:.2f} | "
                f"Bars Used = {current_best.calculate_required_bars()} | "
                f"Total Waste = {current_best.calculate_total_waste()} | "
                f"% Wasted = {current_best.calculate_percentage_wasted():.2f}%"
            )

            # 2. Create Next Generation
            next_population: list[Individual] = []

            # Elitism: carry over the best individuals
            elite = [ind for ind in sorted_population[: self.elitism_size]]
            next_population.extend(elite)

            # Prepare population for selection
            selection_pool = sorted_population

            # Fill the rest of the new population
            while len(next_population) < self.population_size:
                parent1 = self.__selection(selection_pool)
                parent2 = self.__selection(selection_pool)

                child1, child2 = self.__crossover(parent1, parent2)

                next_population.append(self.__mutate(child1))
                if len(next_population) < self.population_size:
                    next_population.append(self.__mutate(child2))

            population = next_population
            gen += 1

        print("\nEvolution finished.")
        result = SimulationResult(self.parameters, gen, best_solutions_by_generation)
        return result

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
            population.append(Individual(self.bar_length, self.cuts_layout_strategy, chromosome))
        return population

    def __should_stop(self, generation: int, best_solution_ever: Individual | None) -> bool:
        # Stop if # of generations has exceeded the limit or if we have achieved the target
        exceeded_generations = generation >= self.generations
        met_target = (
            best_solution_ever.calculate_required_bars() <= self.solution_target
            if best_solution_ever is not None and self.solution_target is not None
            else False
        )
        return exceeded_generations or met_target

    def __selection(self, population: Population) -> Individual:
        """
        Selects a parent using Tournament Selection.
        """
        tournament = random.sample(population, self.tournament_size)
        best = max(tournament, key=lambda x: x.fitness_score)
        return best

    def __crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs Ordered Crossover (OX1) to create two children.
        This preserves the permutation nature of the chromosomes.
        """

        def get_unused_cuts_for_child(
            used_cuts_from_parent_a: list[int], chromosome_from_parent_b: Chromosome
        ) -> list[int]:
            used_cuts_counter = Counter(used_cuts_from_parent_a)
            genes_from_parent_b_for_child: list[int] = []
            for item in chromosome_from_parent_b:
                if used_cuts_counter[item] > 0:
                    used_cuts_counter[item] -= 1
                else:
                    genes_from_parent_b_for_child.append(item)
            assert len(genes_from_parent_b_for_child) == len(chromosome_from_parent_b) - len(used_cuts_from_parent_a)
            return genes_from_parent_b_for_child

        if random.random() > self.crossover_rate:
            child1 = Individual(self.bar_length, self.cuts_layout_strategy, parent1.chromosome[:])
            child2 = Individual(self.bar_length, self.cuts_layout_strategy, parent2.chromosome[:])
            return child1, child2

        size = len(parent1.chromosome)
        child1_chromosome = [-1] * size
        child2_chromosome = [-1] * size

        start, end = sorted(random.sample(range(size), 2))

        # Copy the slice from parents to children
        slice_from_p1 = parent1.chromosome[start : end + 1]
        slice_from_p2 = parent2.chromosome[start : end + 1]
        child1_chromosome[start : end + 1] = slice_from_p1
        child2_chromosome[start : end + 1] = slice_from_p2

        # Fill remaining genes for Child 1 (from Parent 2)
        genes_from_p2_for_child1 = get_unused_cuts_for_child(
            used_cuts_from_parent_a=slice_from_p1, chromosome_from_parent_b=parent2.chromosome
        )
        p2_idx = 0
        for i in range(size):
            if child1_chromosome[i] == -1:
                child1_chromosome[i] = genes_from_p2_for_child1[p2_idx]
                p2_idx += 1
        assert len([cut for cut in child1_chromosome if cut != -1]) == size

        # Fill remaining genes for Child 2 (from Parent 1)
        genes_from_p1_for_child2 = get_unused_cuts_for_child(
            used_cuts_from_parent_a=slice_from_p2, chromosome_from_parent_b=parent1.chromosome
        )
        p1_idx = 0
        for i in range(size):
            if child2_chromosome[i] == -1:
                child2_chromosome[i] = genes_from_p1_for_child2[p1_idx]
                p1_idx += 1
        assert len([cut for cut in child2_chromosome if cut != -1]) == size

        return (
            Individual(self.bar_length, self.cuts_layout_strategy, child1_chromosome),
            Individual(self.bar_length, self.cuts_layout_strategy, child2_chromosome),
        )

    def __mutate(self, individual: Individual) -> Individual:
        def mutate_reverse_subsequence(individual: Individual) -> Individual:
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

            return Individual(self.bar_length, self.cuts_layout_strategy, chromosome)

        def mutate_swap_cuts(individual: Individual) -> Individual:
            if random.random() > self.mutation_rate:
                return individual

            chromosome = individual.chromosome[:]

            # Mutate the Individual (randomly swap genes)
            number_of_mutations = random.randint(len(chromosome) // 10, len(chromosome) // 4)
            for _ in range(number_of_mutations):
                i, j = sorted(random.sample(range(len(chromosome)), 2))
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]

            return Individual(self.bar_length, self.cuts_layout_strategy, chromosome)

        return (
            mutate_swap_cuts(individual)
            if self.mutation_strategy == "SwapCuts"
            else mutate_reverse_subsequence(individual)
        )


def print_best_solution(best_solution: Individual) -> None:
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
        waste = best_solution.bar_length - bar_sum
        print(f"Bar {i + 1:2d} (Length: {best_solution.bar_length}):")
        print(f"  Cuts:  {bar}")
        print(f"  Sum:   {bar_sum}")
        print(f"  Waste: {waste}\n")


def run(parameters: Parameters) -> SimulationResult:
    solver = CSPGeneticAlgorithm(parameters)
    result = solver.run()
    return result


def run_standalone() -> None:
    parameters = Parameters(
        population_size=500,
        generations=3_000,
        # solution_target=76,
        tournament_size=20,
        mutation_rate=0.60,
        crossover_rate=0.80,
        mutation_strategy="ReverseSubsequence",
    )
    result = run(parameters)
    best_solution = max(result.best_solutions_by_generation, key=lambda x: x.fitness_score)
    print_best_solution(best_solution)
