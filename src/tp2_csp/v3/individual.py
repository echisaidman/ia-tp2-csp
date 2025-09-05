import random
from math import inf

# A chromosome is a list of quantities for each cut type
type Chromosome = list[int]


class Individual:
    def __init__(
        self,
        bar_length: int,
        cut_types: list[int],
        required_quantities: list[int],
        penalty_factor: float,
        chromosome: Chromosome | None = None,
    ) -> None:
        self.bar_length = bar_length
        self.cut_types = cut_types
        self.required_quantities = required_quantities
        self.penalty_factor = penalty_factor

        if chromosome is not None:
            self.chromosome = chromosome
        else:
            self.chromosome = self._generate_random_chromosome()

        # These will be calculated during fitness evaluation
        self.fitness_score: float = -inf
        self.cuts_layout: list[list[int]] = []
        self.total_waste: int = 0
        self.is_valid: bool = False

        # Calculate fitness upon creation
        self.calculate_fitness()

    def _generate_random_chromosome(self) -> Chromosome:
        """
        Generates a random chromosome where quantities are varied around the required amounts.
        """
        chromosome: Chromosome = []
        for required_qty in self.required_quantities:
            # Generate a quantity within a range, e.g., +/- 50% of the required amount
            variation = int(required_qty * 0.5)
            lower_bound = max(0, required_qty - variation)
            upper_bound = required_qty + variation
            random_qty = random.randint(lower_bound, upper_bound)
            chromosome.append(random_qty)
        return chromosome

    def _flatten_chromosome(self) -> list[int]:
        """Converts the chromosome (quantities) into a flat list of cuts."""
        all_cuts: list[int] = []
        for i, quantity in enumerate(self.chromosome):
            cut_length = self.cut_types[i]
            all_cuts.extend([cut_length] * quantity)
        return all_cuts

    def calculate_fitness(self) -> None:
        """
        Calculates the fitness of the individual.
        Fitness = (Packing Fitness) - (Validity Penalty)
        """
        # 1. Calculate Packing Fitness based on waste (using Best-Fit Decreasing)
        all_cuts = self._flatten_chromosome()
        # Sorting cuts from largest to smallest (BFD heuristic) can improve packing
        sorted_cuts = sorted(all_cuts, reverse=True)

        cuts_layout, bar_remainders = self._pack_cuts_best_fit(sorted_cuts)
        self.cuts_layout = cuts_layout
        self.total_waste = sum(bar_remainders)

        # Packing fitness is the negative of total waste (we want to minimize waste)
        packing_fitness = -self.total_waste

        # 2. Calculate Validity Penalty
        deviation = 0
        for i, quantity in enumerate(self.chromosome):
            deviation += abs(quantity - self.required_quantities[i])

        penalty = self.penalty_factor * deviation
        self.is_valid = deviation == 0

        # 3. Final Fitness Score
        self.fitness_score = packing_fitness - penalty

    def _pack_cuts_best_fit(self, all_cuts: list[int]) -> tuple[list[list[int]], list[int]]:
        """Calculates the layout of cuts on bars using a Best-Fit heuristic."""
        bars_cuts: list[list[int]] = []
        bar_remainders: list[int] = []

        for cut in all_cuts:
            best_fit_idx = -1
            min_remainder = self.bar_length + 1

            # Find the bar with the tightest fit
            for i, remainder in enumerate(bar_remainders):
                if cut <= remainder:
                    new_remainder = remainder - cut
                    if new_remainder < min_remainder:
                        best_fit_idx = i
                        min_remainder = new_remainder

            if best_fit_idx != -1:
                # Place in the best-fitting bar
                bars_cuts[best_fit_idx].append(cut)
                bar_remainders[best_fit_idx] -= cut
            else:
                # If it doesn't fit anywhere, start a new bar
                bars_cuts.append([cut])
                bar_remainders.append(self.bar_length - cut)

        return bars_cuts, bar_remainders

    def get_total_cuts(self) -> int:
        return sum(self.chromosome)

    def get_required_bars(self) -> int:
        return len(self.cuts_layout)
