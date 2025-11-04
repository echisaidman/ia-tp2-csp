import inspect
import random
from time import sleep

from utils import get_project_root

from ..logger import Logger
from .individual import Individual, Population
from .parameters import Parameters
from .plot import create_plot_for_run
from .result import SimulationResult


class CSPGeneticAlgorithm:
    parameters: Parameters
    all_cuts: list[int]
    tournament_size: int
    logger: Logger

    def __init__(self, parameters: Parameters) -> None:
        self.parameters = parameters
        self.all_cuts = self.__flatten_cuts(parameters.required_cuts)
        self.tournament_size = min(parameters.tournament_size, parameters.population_size)
        self.logger = Logger(ga_provider="ours")

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
            selection_pool = self.__get_population_after_selection(population)
            crossover_pool = self.__get_population_after_crossover(selection_pool)
            mutation_pool = self.__get_population_after_mutation(crossover_pool)
            population = mutation_pool
            gen += 1

            current_best = max(population, key=lambda x: x.fitness_score)
            best_solutions_by_generation.append(current_best)
            if best_solution_ever is None or current_best.fitness_score > best_solution_ever.fitness_score:
                best_solution_ever = current_best
            message = (
                f"Gen {gen:4d} | "
                f"Barras usadas = {current_best.calculate_required_bars()} | "
                f"Desperdicio = {current_best.calculate_total_waste()} | "
                f"% Desperdiciado = {current_best.calculate_percentage_wasted():.2f}%"
            )
            self.logger.log(message)
            print(message)

        print("\nEvolution finished.")

        result = SimulationResult(
            parameters=self.parameters,
            generations=gen,
            best_solutions_by_generation=best_solutions_by_generation,
        )
        self.__save_plot_for_run(result)
        best_solution = max(result.best_solutions_by_generation, key=lambda x: x.fitness_score)
        self.__print_best_solution(best_solution)
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
        for _ in range(self.parameters.population_size):
            chromosome = self.all_cuts[:]
            random.shuffle(chromosome)
            population.append(Individual(self.parameters.bar_length, self.parameters.cuts_layout_strategy, chromosome))
        return population

    def __should_stop(self, generation: int, best_solution_ever: Individual | None) -> bool:
        # Stop if # of generations has exceeded the limit or if we have achieved the target
        exceeded_generations = generation >= self.parameters.generations
        met_target = (
            best_solution_ever.calculate_required_bars() <= self.parameters.solution_target
            if best_solution_ever is not None and self.parameters.solution_target is not None
            else False
        )
        return exceeded_generations or met_target

    def __get_population_after_selection(self, population: Population) -> Population:
        """
        Selects parents using Tournament Selection or Ranking Selection.
        """

        def tournament_selection() -> Population:
            selection_population: Population = []
            for _ in range(self.parameters.selection_size):
                tournament = random.sample(population, self.tournament_size)
                best = max(tournament, key=lambda x: x.fitness_score)
                selection_population.append(best)
            return selection_population

        def ranking_selection() -> Population:
            ordered_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)
            return ordered_population[: self.parameters.selection_size]

        return tournament_selection() if self.parameters.selection_strategy == "Tournament" else ranking_selection()

    def __get_population_after_crossover(self, selection_pool: Population) -> Population:
        children: Population = []
        while len(children) < self.parameters.population_size:
            parent1_idx, parent2_idx = random.sample(range(len(selection_pool)), 2)
            parent1, parent2 = selection_pool[parent1_idx], selection_pool[parent2_idx]
            child1, child2 = self.__crossover(parent1, parent2)
            children.append(child1)
            if len(children) < self.parameters.population_size:
                children.append(child2)
        assert len(children) == self.parameters.population_size
        return children

    def __crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """
        Performs Ordered Crossover (OX1) to create two children.
        This preserves the permutation nature of the chromosomes.
        """
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
        genes_from_p2_for_child1 = parent2.get_missing_cuts(slice_from_p1)
        p2_idx = 0
        for i in range(size):
            if child1_chromosome[i] == -1:
                child1_chromosome[i] = genes_from_p2_for_child1[p2_idx]
                p2_idx += 1
        assert len([cut for cut in child1_chromosome if cut != -1]) == size

        # Fill remaining genes for Child 2 (from Parent 1)
        genes_from_p1_for_child2 = parent1.get_missing_cuts(slice_from_p2)
        p1_idx = 0
        for i in range(size):
            if child2_chromosome[i] == -1:
                child2_chromosome[i] = genes_from_p1_for_child2[p1_idx]
                p1_idx += 1
        assert len([cut for cut in child2_chromosome if cut != -1]) == size

        return (
            Individual(self.parameters.bar_length, self.parameters.cuts_layout_strategy, child1_chromosome),
            Individual(self.parameters.bar_length, self.parameters.cuts_layout_strategy, child2_chromosome),
        )

    def __get_population_after_mutation(self, population: Population) -> Population:
        if random.random() > self.parameters.mutation_rate:
            return population

        number_of_individuals_to_mutate = (
            1
            if self.parameters.percentage_of_individuals_to_mutate is None
            else int(len(population) * self.parameters.percentage_of_individuals_to_mutate)
        )
        for _ in range(number_of_individuals_to_mutate):
            idx_to_mutate = random.randint(0, len(population) - 1)
            individual_to_mutate = population[idx_to_mutate]
            mutated_individual = self.__mutate(individual_to_mutate)
            population[idx_to_mutate] = mutated_individual
        return population

    def __mutate(self, individual: Individual) -> Individual:
        def mutate_reverse_subsequence(individual: Individual) -> Individual:
            start, end = sorted(random.sample(range(len(individual.chromosome)), 2))
            chromosome = individual.chromosome[:]
            sub_sequence = chromosome[start : end + 1]
            sub_sequence = sub_sequence[::-1]
            chromosome[start : end + 1] = sub_sequence
            return Individual(self.parameters.bar_length, self.parameters.cuts_layout_strategy, chromosome)

        def mutate_swap_cuts(individual: Individual) -> Individual:
            chromosome = individual.chromosome[:]
            i, j = sorted(random.sample(range(len(chromosome)), 2))
            chromosome[i], chromosome[j] = chromosome[j], chromosome[i]
            return Individual(self.parameters.bar_length, self.parameters.cuts_layout_strategy, chromosome)

        return (
            mutate_swap_cuts(individual)
            if self.parameters.mutation_strategy == "SwapCuts"
            else mutate_reverse_subsequence(individual)
        )

    def __print_best_solution(self, best_solution: Individual) -> None:
        messages_per_bar: list[str] = []
        for i, bar in enumerate(best_solution.cuts_layout):
            bar_sum = sum(bar)
            waste = best_solution.bar_length - bar_sum
            message_for_bar = inspect.cleandoc(f"""
                Barra {i + 1:2d} (Tamaño: {best_solution.bar_length} unidades):
                    Cortes:  {bar}
                    Utilizado:   {bar_sum} unidades
                    Desperdicio: {waste} unidades
            """)
            messages_per_bar.append(message_for_bar)

        message = inspect.cleandoc(f"""
\n\n=================================
    Mejor solucion encontrada    
=================================
Barras requeridas: {best_solution.calculate_required_bars()}
Desperdicio total: {best_solution.calculate_total_waste()} unidades
% Desperdiciado: {best_solution.calculate_percentage_wasted():.2f}%
---------------------------------

{"\n\n".join(messages_per_bar)}
================================""")
        self.logger.log(f"\n\n{message}")

    def __save_plot_for_run(self, result: SimulationResult) -> None:
        fig, ax = create_plot_for_run(result)
        fig.savefig(f"{get_project_root()}/logs/{self.logger.file_name}.png")


def run(parameters: Parameters) -> SimulationResult:
    delay = random.random()
    sleep(delay)
    solver = CSPGeneticAlgorithm(parameters)
    result = solver.run()
    return result


def run_standalone() -> SimulationResult:
    parameters = Parameters(
        population_size=500,
        generations=3_000,
        tournament_size=20,
        mutation_rate=0.60,
        mutation_strategy="ReverseSubsequence",
        percentage_of_individuals_to_mutate=1,
        selection_size=200,
    )
    result = run(parameters)
    return result
