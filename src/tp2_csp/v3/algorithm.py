import random

from .individual import Individual

type Population = list[Individual]


class CSPGeneticAlgorithmV3:
    def __init__(
        self,
        required_cuts: dict[int, int],
        bar_length: int,
        population_size: int,
        generations: int,
        crossover_rate: float,
        mutation_rate: float,
        elitism_size: int,
        tournament_size: int,
        penalty_factor: float,
    ) -> None:
        self.bar_length = bar_length
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_size = elitism_size
        self.tournament_size = tournament_size
        self.penalty_factor = penalty_factor

        # Establish a fixed order for cut types and quantities
        self.cut_types = sorted(required_cuts.keys())
        self.required_quantities = [required_cuts[key] for key in self.cut_types]

    def run(self) -> Individual:
        """
        Runs the genetic algorithm to find the optimal cutting pattern.
        """
        print("Genetic Algorithm for Cutting Stock Problem (v3 - Flexible Chromosome)")
        print("--------------------------------------------------------------------")

        population = self._initialize_population()
        best_solution_ever = max(population, key=lambda x: x.fitness_score)

        for gen in range(self.generations):
            # Sort population by fitness (descending)
            sorted_population = sorted(population, key=lambda x: x.fitness_score, reverse=True)

            # Update the best solution ever found
            current_best = sorted_population[0]
            if current_best.fitness_score > best_solution_ever.fitness_score:
                best_solution_ever = current_best

            if gen == 0 or (gen + 1) % 20 == 0 or gen == self.generations - 1:
                valid_str = "YES" if best_solution_ever.is_valid else "NO"
                print(
                    f"Gen {gen + 1:4d}: Best Fitness = {best_solution_ever.fitness_score:10.2f} | "
                    f"Bars = {best_solution_ever.get_required_bars()} | "
                    f"Waste = {best_solution_ever.total_waste:6d} | "
                    f"Is Valid = {valid_str}"
                )

            # Create Next Generation
            next_population: Population = []

            # Elitism
            next_population.extend(sorted_population[: self.elitism_size])

            # Fill the rest of the new population
            while len(next_population) < self.population_size:
                parent1 = self._selection(sorted_population)
                parent2 = self._selection(sorted_population)

                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1, parent2  # Pass through

                next_population.append(self._mutate(child1))
                if len(next_population) < self.population_size:
                    next_population.append(self._mutate(child2))

            population = next_population

        print("\nEvolution finished.")
        if not best_solution_ever.is_valid:
            print("⚠️ WARNING: The best solution found is not a valid solution (does not meet all cut requirements).")
            print("   Consider increasing generations, population size, or the penalty_factor.")

        return best_solution_ever

    def _initialize_population(self) -> Population:
        """Creates the initial population."""
        return [
            Individual(self.bar_length, self.cut_types, self.required_quantities, self.penalty_factor)
            for _ in range(self.population_size)
        ]

    def _selection(self, population: Population) -> Individual:
        """Selects a parent using Tournament Selection."""
        tournament = random.sample(population, self.tournament_size)
        return max(tournament, key=lambda x: x.fitness_score)

    def _crossover(self, parent1: Individual, parent2: Individual) -> tuple[Individual, Individual]:
        """Performs Two-Point Crossover on the chromosomes (lists of quantities)."""
        size = len(parent1.chromosome)
        p1_chromosome = parent1.chromosome[:]
        p2_chromosome = parent2.chromosome[:]

        # Select two crossover points
        cx_point1, cx_point2 = sorted(random.sample(range(size), 2))

        # Swap the segment between the points
        p1_chromosome[cx_point1:cx_point2], p2_chromosome[cx_point1:cx_point2] = (
            p2_chromosome[cx_point1:cx_point2],
            p1_chromosome[cx_point1:cx_point2],
        )

        child1 = Individual(
            self.bar_length,
            self.cut_types,
            self.required_quantities,
            self.penalty_factor,
            p1_chromosome,
        )
        child2 = Individual(
            self.bar_length,
            self.cut_types,
            self.required_quantities,
            self.penalty_factor,
            p2_chromosome,
        )
        return child1, child2

    def _mutate(self, individual: Individual) -> Individual:
        """
        Performs Creep Mutation. A random gene (quantity) is slightly modified.
        """
        if random.random() >= self.mutation_rate:
            return individual

        chromosome = individual.chromosome[:]

        # Select a random gene to mutate
        gene_idx = random.randint(0, len(chromosome) - 1)

        # Determine a small change (+/- 1 or 2)
        change = random.choice([-2, -1, 1, 2])

        # Apply the change, ensuring the quantity doesn't go below zero
        chromosome[gene_idx] = max(0, chromosome[gene_idx] + change)

        mutated_individual = Individual(
            self.bar_length,
            self.cut_types,
            self.required_quantities,
            self.penalty_factor,
            chromosome,
        )
        return mutated_individual
